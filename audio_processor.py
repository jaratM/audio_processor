import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess
import io
import torchaudio
import torch
from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc
import time
from functools import lru_cache
from utils import remove_special_characters
from speech_segment import SpeechBatchTranscriber

class AudioProcessor:
    """Optimized audio processor for large-scale processing"""
    
    def __init__(self, config: dict, db_manager=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("OptimizedAudioProcessor initialized")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Database manager (optional)
        self.db_manager = db_manager
        self.transcriber = SpeechBatchTranscriber(config)
        # Memory management
        self.max_memory_usage = 0.8  # 80% of available memory
        self.chunk_cache = {}
        self.cache_size = 100
        self.chunk_duration_sec = self.config['chunk_duration_sec']
        self.overlap_sec = self.config['overlap_sec']
        self.target_sample_rate = self.config['target_sample_rate']
        # Retries and failure tracking
        self.max_retries = int(self.config.get('max_retries', 3))
        self.failed_files: List[Dict[str, Any]] = []
        # Threading
        self.io_workers = int(self.config.get('io_workers', 8))
        self.io_executor = ThreadPoolExecutor(max_workers=self.io_workers)
        
    def load_models(self):
        """Load models with memory optimization"""
        self.logger.info("Loading transcription model")
        
        # Load with memory optimization
        self.model = Wav2Vec2BertForCTC.from_pretrained(
            self.config['transcription_model'],
            torch_dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.processor = Wav2Vec2BertProcessor.from_pretrained(
            self.config['transcription_model']
        )
        
        self.model.eval()

        self.transcriber.segmenter.transcription_model = self.model
        self.transcriber.segmenter.processor = self.processor
        self.transcriber.segmenter.device = self.device
        
        self.logger.info("Transcription model loaded successfully")
    
    def load_audio(self, audio_path: Path) -> tuple:
        """Load audio with memory optimization"""
        try:
            if str(audio_path).lower().endswith('.ogg'):
                # Use subprocess for OGG files
                command = [
                    "ffmpeg", "-i", str(audio_path), 
                    "-f", "wav", "-acodec", "pcm_s16le", "-"
                ]
                proc = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                wav_bytes, _ = proc.communicate()
                waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
            else:
                # Use memory mapping for large files
                waveform, sample_rate = torchaudio.load(
                    str(audio_path)
                )
            
            return waveform, sample_rate
            
        except Exception as e:
            self.logger.error(f"Error loading audio {audio_path}: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def get_resampler(self, orig_freq: int, new_freq: int):
        """Cached resampler for efficiency"""
        return torchaudio.transforms.Resample(orig_freq, new_freq)
    
    def split_audio(self, waveform: torch.Tensor, sample_rate: int, file_name: str) -> List[Dict]:
        """Split audio into optimized chunks with memory management"""

        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = self.get_resampler(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        
        chunk_samples = int(self.chunk_duration_sec * sample_rate)
        overlap_samples = int(self.overlap_sec * sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        total_samples = waveform.shape[1]
        chunks = []
        
        # Handle stereo/mono efficiently
        if waveform.shape[0] == 2:
            agent_waveform = waveform[0].unsqueeze(0)
            client_waveform = waveform[1].unsqueeze(0)
            mixed_waveform = waveform.mean(dim=0, keepdim=True)
        else:
            # waveform is already [1, L] (mono). Keep it 2D.
            agent_waveform = waveform
            client_waveform = waveform
            mixed_waveform = waveform
        
        start = 0
        chunk_idx = 0
        
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            
            # Extract chunks efficiently
            chunk_mixed = mixed_waveform[:, start:end]
            chunk_agent = agent_waveform[:, start:end]
            chunk_client = client_waveform[:, start:end]
            
            chunks.append({
                'file_name': f'{file_name}',
                'stereo_waveform': chunk_mixed,
                'agent_waveform': chunk_agent,
                'client_waveform': chunk_client,
                'chunk_idx': chunk_idx,
                'start_time': start / sample_rate,
                'end_time': end / sample_rate
            })
            
            chunk_idx += 1
            start += step_samples
            
            if end >= total_samples:
                break
        
        return chunks, agent_waveform, client_waveform
    
    def transcribe_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Transcribe chunks with optimized batching and memory management"""
        if not chunks:
            return []
        
        batch_size = int(self.config['batch_size'])
        results = []
        
        # Optional length bucketing to reduce padding overhead
        if bool(self.config.get('enable_length_bucketing', True)):
            try:
                chunks = sorted(chunks, key=lambda c: int(c.get('stereo_waveform', c.get('agent_waveform')).shape[1]))
            except Exception:
                pass

        i = 0
        # Process in batches with simple OOM-aware adjust on CUDA
        while i < len(chunks):
            batch_end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:batch_end]
            
            try:
                # Process batch
                batch_results = self._process_single_batch(batch_chunks)
                results.extend(batch_results)
                
                # Memory cleanup
                self._cleanup_memory()
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # If OOM on GPU, reduce batch size and retry once
                if torch.cuda.is_available() and 'out of memory' in str(e).lower() and batch_size > 1:
                    torch.cuda.empty_cache()
                    new_bs = max(1, batch_size // 2)
                    self.logger.warning(f"CUDA OOM detected. Reducing batch size {batch_size} -> {new_bs} and retrying this slice")
                    batch_size = new_bs
                    continue  # retry same i with smaller batch
                # Add error results for failed batch
                for chunk in batch_chunks:
                    chunk.update({
                        'transcription_chunk': '',
                        'agent_transcription': '',
                        'client_transcription': '',
                        'error': str(e)
                    })
                results.extend(batch_chunks)
            # advance only if processed or after error handling
            i = batch_end

        return results
   
    def _process_single_batch(self, batch_chunks: List[Dict]) -> List[Dict]:
        """Process a single batch of chunks"""
        # Extract waveforms
        agent_waveforms = [chunk['agent_waveform'] for chunk in batch_chunks]
        client_waveforms = [chunk['client_waveform'] for chunk in batch_chunks]
        stereo_waveforms = [chunk['stereo_waveform'] for chunk in batch_chunks]
        
        # Convert to numpy arrays
        stereo_arrays = [wf.squeeze().numpy() for wf in stereo_waveforms]
        agent_arrays = [wf.squeeze().numpy() for wf in agent_waveforms]
        client_arrays = [wf.squeeze().numpy() for wf in client_waveforms]
        
        # Process with mixed precision if available
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                results = self._run_inference(stereo_arrays, agent_arrays, client_arrays)
        else:
            results = self._run_inference(stereo_arrays, agent_arrays, client_arrays)
        
        # Update chunks with results
        for i, chunk in enumerate(batch_chunks):
            if i < len(results):
                chunk.update(results[i])
            else:
                chunk.update({
                    'transcription_chunk': '',
                    'agent_transcription': '',
                    'client_transcription': '',
                    'error': 'Index out of range'
                })
        
        return batch_chunks
    
    def _run_inference(self, stereo_arrays: List[np.ndarray], 
                      agent_arrays: List[np.ndarray], 
                      client_arrays: List[np.ndarray]) -> List[Dict]:
        """Run inference on arrays"""
        try:
            # Process all three types
            stereo_inputs = self.processor(
                stereo_arrays, sampling_rate=16000, return_tensors="pt", padding=True
            )
            agent_inputs = self.processor(
                agent_arrays, sampling_rate=16000, return_tensors="pt", padding=True
            )
            client_inputs = self.processor(
                client_arrays, sampling_rate=16000, return_tensors="pt", padding=True
            )
            
            # Move to device
            stereo_features = stereo_inputs["input_features"].to(self.device)
            agent_features = agent_inputs["input_features"].to(self.device)
            client_features = client_inputs["input_features"].to(self.device)
            
            # Run inference
            with torch.no_grad():
                stereo_logits = self.model(input_features=stereo_features).logits
                agent_logits = self.model(input_features=agent_features).logits
                client_logits = self.model(input_features=client_features).logits
            
            # Decode predictions
            stereo_ids = torch.argmax(stereo_logits, dim=-1)
            agent_ids = torch.argmax(agent_logits, dim=-1)
            client_ids = torch.argmax(client_logits, dim=-1)
            
            stereo_texts = self.processor.batch_decode(stereo_ids)
            agent_texts = self.processor.batch_decode(agent_ids)
            client_texts = self.processor.batch_decode(client_ids)
            
            # Clean texts
            stereo_cleaned = [remove_special_characters(text) for text in stereo_texts]
            agent_cleaned = [remove_special_characters(text) for text in agent_texts]
            client_cleaned = [remove_special_characters(text) for text in client_texts]
            
            # Prepare results
            results = []
            for i in range(len(stereo_cleaned)):
                results.append({
                    'transcription_chunk': stereo_cleaned[i] if i < len(stereo_cleaned) else "",
                    'agent_transcription': agent_cleaned[i] if i < len(agent_cleaned) else "",
                    'client_transcription': client_cleaned[i] if i < len(client_cleaned) else "",
                    'error': ''
                })
            
            # Cleanup GPU memory
            del (stereo_features, agent_features, client_features,
                 stereo_logits, agent_logits, client_logits,
                 stereo_inputs, agent_inputs, client_inputs)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            # Return empty results
            return [{'transcription_chunk': '', 'agent_transcription': '', 
                    'client_transcription': '', 'error': str(e)}] * len(stereo_arrays)
    
    def _cleanup_memory(self):
        """Clean up memory after processing"""
        # Clear cache
        if len(self.chunk_cache) > self.cache_size:
            self.chunk_cache.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def process_batch(self, batch_id: int, audio_files: List[Path]) -> List[Dict]:
        """Load and chunk files in parallel, then transcribe them all in parallel"""
        
        all_chunks = []
        all_agent_waveforms = []
        all_client_waveforms = []
        all_id_enregistrements = []
        # Load and chunk files in parallel
        futures = []
        for file in audio_files:
            future = self.io_executor.submit(self._process_single_file_with_retries, file)
            futures.append((file, future))
        
        # Collect results
        for file, future in futures:
            try:
                chunks, agent_waveform, client_waveform, id_enregistrement = future.result(timeout=120)  # 2 minute timeout
                if chunks:
                    all_chunks.extend(chunks)
                    # Keep per-file tensors and IDs as single list items
                    all_agent_waveforms.append(agent_waveform)
                    all_client_waveforms.append(client_waveform)
                    all_id_enregistrements.append(id_enregistrement)
                else:
                    # Treat empty chunks as failure after retries
                    self.failed_files.append({'filename': file.name, 'path': str(file), 'error': 'empty_chunks_after_retries', 'chunks': []})
            except Exception as e:
                self.logger.error(f"Error processing file {file}: {e}")
                self.failed_files.append({'filename': file.name, 'path': str(file), 'error': str(e), 'chunks': []})

        if all_agent_waveforms:
            for id_enregistrement, agent_waveform, client_waveform in zip(all_id_enregistrements, all_agent_waveforms, all_client_waveforms):
                self._save_messages_to_database(id_enregistrement, agent_waveform, client_waveform)
                del agent_waveform, client_waveform
        # Transcribe all chunks
        if all_chunks:
            results = self.transcribe_batch(all_chunks)
        else:
            results = []
        del all_agent_waveforms, all_client_waveforms, all_id_enregistrements
        gc.collect()
        self.logger.info(f"Chunking and transcription for batch {batch_id + 1} completed")
        return results

    def _process_single_file_with_retries(self, file_path: Path) -> List[Dict]:
        """Retry wrapper around _process_single_file using max_retries from config."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                chunks, agent_waveform, client_waveform, id_enregistrement = self._process_single_file(file_path)
                if chunks:
                    return chunks, agent_waveform, client_waveform, id_enregistrement 
                else:
                    raise RuntimeError("no_chunks")
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt}/{self.max_retries} failed for {file_path}: {e}")
                time.sleep(min(5, attempt))
        self.logger.error(f"All {self.max_retries} attempts failed for {file_path}: {last_error}")
        return [], None, None, None
    
    def _process_single_file(self, file_path: Path) -> List[Dict]:
        """Process a single file with error handling"""
        try:
            waveform, sample_rate = self.load_audio(file_path)
            chunks, agent_waveform, client_waveform = self.split_audio(waveform, sample_rate, file_path.name)
            id_enregistrement = file_path.name.split('.')[0]
            
            # Save call information to database
            self._save_call_to_database(id_enregistrement, waveform)

            # Clear waveform from memory
            del waveform
            gc.collect()
            return chunks, agent_waveform, client_waveform, id_enregistrement   
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return [], None, None, None
        
    def _save_messages_to_database(self, id_enregistrement: str, agent_waveform: torch.Tensor, client_waveform: torch.Tensor):
        sr = int(self.target_sample_rate)
        transcription = []
        try:
            transcription.extend(self.transcriber.transcribe_mono(agent_waveform.cpu(), sr, 'agent'))
            transcription.extend(self.transcriber.transcribe_mono(client_waveform.cpu(), sr, 'client'))
        except Exception as e:
            self.logger.error(f"Mono transcription failed for {id_enregistrement}: {e}")
            transcription = []
        
        transcript_sorted = sorted(transcription, key=lambda x: x.get("start", 0.0))
        for i, message in enumerate(transcript_sorted):
            row = {
                'id_enregistrement': id_enregistrement,
                'text': message['text'],
                'speaker': message['speaker'],
                'order_message': i + 1      

            }
            self.db_manager.insert_message(row)

    def _save_call_to_database(self, id_enregistrement: str, waveform: torch.Tensor):
        """Save call information to database"""
        try:
            from datetime import date
            
            # Calculate duration from waveform
            duration_seconds = waveform.shape[1] / self.target_sample_rate
            # Create call record
            call_data = {
                'id_enregistrement': id_enregistrement,
                'file': '',# will be updated later,
                'duration_seconds': duration_seconds,
                'date_appel': date.today().isoformat(),
                'partenaire': self.config.get('partenaire', 'INWI'),
                'login_conseiller': self.config.get('login_conseiller', 'system'),
                'topics': '',  # Will be updated later if available
                'emotion_client_globale': '',  # Will be updated after sentiment analysis
                'ton_agent_global': ''  # Will be updated after sentiment analysis
            }
            
            # Insert call record
            _ = self.db_manager.insert_call(call_data)
            
        except Exception as e:
            self.logger.error(f"Failed to save call to database: {e}")
            # Don't raise the exception to avoid interrupting file processing 