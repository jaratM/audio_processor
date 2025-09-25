#!/usr/bin/env python3
"""
Multi-GPU Processor for H100 MIG-based Parallel Audio Processing
Implements true GPU parallelism using Multi-Instance GPU (MIG) technology
"""

import os
import torch
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
import time
import gc
from functools import lru_cache

from services.audio_processor import AudioProcessor
from services.speech_segment import SpeechBatchTranscriber
from services.sentiment_analysis import SentimentAnalyzer
from utils.utils import check_gpu_availability


@dataclass
class MIGInstance:
    """Configuration for a MIG GPU instance"""
    device_id: int
    memory_gb: float
    batch_size_multiplier: float
    device: torch.device
    name: str


class MultiGPUManager:
    """Manages multiple GPU instances and their workloads"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.mig_enabled = config.get('enable_mig', False)
        self.instances: Dict[str, MIGInstance] = {}
        self.device_locks: Dict[int, threading.Lock] = {}
        
        if self.mig_enabled:
            self._setup_mig_instances()
        else:
            self._setup_single_gpu()
    
    def _setup_mig_instances(self):
        """Setup MIG instances from configuration"""
        self.logger.info("Setting up MIG instances...")
        
        mig_config = self.config.get('mig_instances', {})
        
        for instance_name, instance_config in mig_config.items():
            device_id = instance_config['device_id']
            
            # Verify the MIG instance exists
            try:
                device = torch.device(f"cuda:{device_id}")
                torch.cuda.set_device(device)
                
                # Test memory allocation
                test_tensor = torch.zeros(100, device=device)
                del test_tensor
                
                self.instances[instance_name] = MIGInstance(
                    device_id=device_id,
                    memory_gb=instance_config['memory_gb'],
                    batch_size_multiplier=instance_config['batch_size_multiplier'],
                    device=device,
                    name=instance_name
                )
                self.device_locks[device_id] = threading.Lock()
                
                self.logger.info(f"MIG instance '{instance_name}' configured on device {device_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to setup MIG instance '{instance_name}': {e}")
                raise
    
    def _setup_single_gpu(self):
        """Fallback to single GPU configuration"""
        self.logger.info("MIG disabled, using single GPU configuration")
        
        gpu_index = self.config.get('gpu_index', 0)
        device = torch.device(f"cuda:{gpu_index}")
        
        # Create virtual instances on the same device
        for instance_name in ['transcription', 'vad', 'sentiment']:
            self.instances[instance_name] = MIGInstance(
                device_id=gpu_index,
                memory_gb=20.0,  # Shared memory
                batch_size_multiplier=1.0,
                device=device,
                name=instance_name
            )
        
        self.device_locks[gpu_index] = threading.Lock()
    
    def get_instance(self, instance_name: str) -> Optional[MIGInstance]:
        """Get MIG instance by name"""
        return self.instances.get(instance_name)
    
    def get_device_lock(self, device_id: int) -> threading.Lock:
        """Get thread lock for device"""
        return self.device_locks.get(device_id, threading.Lock())


class ParallelAudioProcessor:
    """Enhanced audio processor with true GPU parallelism"""
    
    def __init__(self, config: dict, db_manager=None):
        self.config = config
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Setup GPU manager
        self.gpu_manager = MultiGPUManager(config)
        
        # Initialize processors for each GPU instance
        self.processors = {}
        self._setup_processors()
        
        # Parallel execution
        self.max_parallel_jobs = config.get('max_workers', 16)
        self.executor = ThreadPoolExecutor(max_workers=self.max_parallel_jobs)
        
    def _setup_processors(self):
        """Setup specialized processors for each MIG instance"""
        self.logger.info("Setting up specialized processors...")
        
        # Transcription processor (largest instance)
        transcription_instance = self.gpu_manager.get_instance('transcription')
        if transcription_instance:
            self.processors['transcription'] = EnhancedTranscriptionProcessor(
                self.config, transcription_instance, self.db_manager
            )
        
        # VAD processor
        vad_instance = self.gpu_manager.get_instance('vad')
        if vad_instance:
            self.processors['vad'] = EnhancedVADProcessor(
                self.config, vad_instance
            )
        
        # Sentiment processor
        sentiment_instance = self.gpu_manager.get_instance('sentiment')
        if sentiment_instance:
            self.processors['sentiment'] = EnhancedSentimentProcessor(
                self.config, sentiment_instance
            )
    
    def process_batch_parallel(self, batch_id: int, audio_files: List[Path]) -> List[Dict]:
        """Process batch with true GPU parallelism"""
        self.logger.info(f"Processing batch {batch_id + 1} with {len(audio_files)} files in parallel")
        
        try:
            # Submit all three GPU operations in parallel
            futures = {}
            
            # 1. Audio loading and chunking (CPU + VAD GPU)
            if 'vad' in self.processors:
                futures['preprocessing'] = self.executor.submit(
                    self._parallel_preprocessing, batch_id, audio_files
                )
            
            # Wait for preprocessing to complete
            preprocessing_results = futures['preprocessing'].result(timeout=300)
            if not preprocessing_results:
                return []
            
            all_chunks, file_metadata = preprocessing_results
            
            # 2. Parallel GPU processing
            parallel_futures = {}
            
            # Transcription on dedicated GPU instance
            if 'transcription' in self.processors:
                parallel_futures['transcription'] = self.executor.submit(
                    self._parallel_transcription, all_chunks
                )
            
            # Sentiment analysis on dedicated GPU instance  
            if 'sentiment' in self.processors:
                parallel_futures['sentiment'] = self.executor.submit(
                    self._parallel_sentiment, all_chunks
                )
            
            # Collect results from parallel GPU operations
            results = {}
            for operation, future in parallel_futures.items():
                try:
                    results[operation] = future.result(timeout=600)  # 10 minute timeout
                except Exception as e:
                    self.logger.error(f"Parallel {operation} failed: {e}")
                    results[operation] = None
            
            # Combine results
            final_results = self._combine_parallel_results(
                all_chunks, 
                results.get('transcription'),
                results.get('sentiment'),
                file_metadata
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Parallel batch processing failed for batch {batch_id + 1}: {e}")
            return []
    
    def _parallel_preprocessing(self, batch_id: int, audio_files: List[Path]) -> Tuple[List[Dict], Dict]:
        """Preprocess files in parallel using VAD GPU instance"""
        if 'vad' not in self.processors:
            return [], {}
        
        return self.processors['vad'].process_files_for_chunks(audio_files)
    
    def _parallel_transcription(self, chunks: List[Dict]) -> List[Dict]:
        """Transcribe chunks using dedicated transcription GPU instance"""
        if 'transcription' not in self.processors:
            return chunks
        
        return self.processors['transcription'].transcribe_chunks_batch(chunks)
    
    def _parallel_sentiment(self, chunks: List[Dict]) -> List[Dict]:
        """Analyze sentiment using dedicated sentiment GPU instance"""
        if 'sentiment' not in self.processors:
            return chunks
        
        return self.processors['sentiment'].analyze_chunks_batch(chunks)
    
    def _combine_parallel_results(self, chunks: List[Dict], transcription_results: Optional[List[Dict]], 
                                 sentiment_results: Optional[List[Dict]], metadata: Dict) -> List[Dict]:
        """Combine results from parallel GPU operations"""
        # Create result mapping by chunk ID
        transcription_map = {}
        sentiment_map = {}
        
        if transcription_results:
            for result in transcription_results:
                chunk_id = result.get('chunk_id', f"{result.get('file_name', '')}_{result.get('chunk_idx', 0)}")
                transcription_map[chunk_id] = result
        
        if sentiment_results:
            for result in sentiment_results:
                chunk_id = result.get('chunk_id', f"{result.get('file_name', '')}_{result.get('chunk_idx', 0)}")
                sentiment_map[chunk_id] = result
        
        # Combine all results
        combined_results = []
        for chunk in chunks:
            chunk_id = f"{chunk.get('file_name', '')}_{chunk.get('chunk_idx', 0)}"
            
            # Start with original chunk
            combined_chunk = chunk.copy()
            
            # Add transcription results
            if chunk_id in transcription_map:
                combined_chunk.update(transcription_map[chunk_id])
            
            # Add sentiment results
            if chunk_id in sentiment_map:
                combined_chunk.update(sentiment_map[chunk_id])
            
            combined_results.append(combined_chunk)
        
        return combined_results


class EnhancedTranscriptionProcessor:
    """Specialized transcription processor for dedicated GPU instance"""
    
    def __init__(self, config: dict, mig_instance: MIGInstance, db_manager=None):
        self.config = config
        self.mig_instance = mig_instance
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        torch.cuda.set_device(self.mig_instance.device)
        
        # Enhanced batch size for larger GPU instance
        self.batch_size = int(
            config.get('chunk_batch_size', 16) * mig_instance.batch_size_multiplier
        )
        
        # Initialize transcriber on specific device
        self.transcriber = self._setup_transcriber()
    
    def _setup_transcriber(self) -> SpeechBatchTranscriber:
        """Setup transcriber on specific GPU instance"""
        # Create device-specific config
        device_config = self.config.copy()
        device_config['gpu_index'] = self.mig_instance.device_id
        device_config['batch_size'] = self.batch_size
        
        # Set device before creating transcriber
        torch.cuda.set_device(self.mig_instance.device)
        
        transcriber = SpeechBatchTranscriber(device_config)
        
        # Load models on specific device
        if hasattr(transcriber.segmenter, 'transcription_model') and transcriber.segmenter.transcription_model:
            transcriber.segmenter.transcription_model = transcriber.segmenter.transcription_model.to(self.mig_instance.device)
        if hasattr(transcriber.segmenter, 'processor'):
            # Processor doesn't need device transfer, but VAD pipeline does
            if hasattr(transcriber.segmenter, 'vad_pipeline'):
                transcriber.segmenter.vad_pipeline = transcriber.segmenter.vad_pipeline.to(self.mig_instance.device)
        
        return transcriber
    
    def transcribe_chunks_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Transcribe chunks using dedicated GPU instance"""
        if not chunks:
            return []
        
        device_lock = torch.cuda.current_device()
        
        try:
            with torch.cuda.device(self.mig_instance.device):
                # Process in optimized batches
                results = []
                for i in range(0, len(chunks), self.batch_size):
                    batch_chunks = chunks[i:i + self.batch_size]
                    
                    # Extract audio segments for transcription
                    segments = []
                    for chunk in batch_chunks:
                        if 'agent_waveform' in chunk:
                            segments.append({
                                'segment_waveform': chunk['agent_waveform'],
                                'start': chunk.get('start_time', 0),
                                'end': chunk.get('end_time', 0),
                                'speaker': 'agent'
                            })
                        if 'client_waveform' in chunk:
                            segments.append({
                                'segment_waveform': chunk['client_waveform'], 
                                'start': chunk.get('start_time', 0),
                                'end': chunk.get('end_time', 0),
                                'speaker': 'client'
                            })
                    
                    # Transcribe segments
                    if segments:
                        transcribed = self.transcriber.segmenter.transcribe_segments_batched(
                            segments, self.config['target_sample_rate']
                        )
                        
                        # Map results back to chunks
                        for j, chunk in enumerate(batch_chunks):
                            chunk_result = chunk.copy()
                            chunk_result['chunk_id'] = f"{chunk.get('file_name', '')}_{chunk.get('chunk_idx', j)}"
                            
                            # Add transcription results
                            agent_text = ''
                            client_text = ''
                            for seg in transcribed:
                                if seg.get('speaker') == 'agent':
                                    agent_text += seg.get('text', '') + ' '
                                elif seg.get('speaker') == 'client':
                                    client_text += seg.get('text', '') + ' '
                            
                            chunk_result['agent_transcription'] = agent_text.strip()
                            chunk_result['client_transcription'] = client_text.strip()
                            chunk_result['transcription_chunk'] = (agent_text + ' ' + client_text).strip()
                            
                            results.append(chunk_result)
                    
                    # Clear GPU memory
                    torch.cuda.empty_cache()
                
                return results
                
        except Exception as e:
            self.logger.error(f"Transcription processing failed on device {self.mig_instance.device_id}: {e}")
            return chunks


class EnhancedVADProcessor:
    """Specialized VAD and preprocessing processor"""
    
    def __init__(self, config: dict, mig_instance: MIGInstance):
        self.config = config
        self.mig_instance = mig_instance
        self.logger = logging.getLogger(__name__)
        
        # Setup audio processor for this instance
        torch.cuda.set_device(self.mig_instance.device)
        self.audio_processor = AudioProcessor(config)
    
    def process_files_for_chunks(self, audio_files: List[Path]) -> Tuple[List[Dict], Dict]:
        """Process files and extract chunks using VAD"""
        all_chunks = []
        file_metadata = {}
        
        with torch.cuda.device(self.mig_instance.device):
            for file_path in audio_files:
                try:
                    # Load and process file
                    chunks, agent_waveform, client_waveform, id_enregistrement = \
                        self.audio_processor._process_single_file(file_path)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        file_metadata[str(file_path)] = {
                            'id_enregistrement': id_enregistrement,
                            'chunk_count': len(chunks)
                        }
                    
                except Exception as e:
                    self.logger.error(f"VAD processing failed for {file_path}: {e}")
        
        return all_chunks, file_metadata


class EnhancedSentimentProcessor:
    """Specialized sentiment analysis processor"""
    
    def __init__(self, config: dict, mig_instance: MIGInstance):
        self.config = config
        self.mig_instance = mig_instance
        self.logger = logging.getLogger(__name__)
        
        # Setup sentiment analyzer for this instance
        torch.cuda.set_device(self.mig_instance.device)
        
        # Adjust batch size for smaller instance
        self.batch_size = max(1, int(
            config.get('chunk_batch_size', 16) * mig_instance.batch_size_multiplier
        ))
        
        try:
            self.sentiment_analyzer = SentimentAnalyzer(config)
        except Exception as e:
            self.logger.warning(f"Could not initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def analyze_chunks_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Analyze sentiment for chunks using dedicated GPU instance"""
        if not chunks or not self.sentiment_analyzer:
            return chunks
        
        try:
            with torch.cuda.device(self.mig_instance.device):
                return self.sentiment_analyzer.analyze_batch_sentiment(chunks)
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed on device {self.mig_instance.device_id}: {e}")
            return chunks
