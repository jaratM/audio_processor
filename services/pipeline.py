#!/usr/bin/env python3
"""
Optimized Audio Processing Pipeline for Thousands of Files
Implements parallel processing, memory management, and efficient batching
"""

import os
import yaml
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import pandas as pd
import torch
import torchaudio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import multiprocessing as mp
from dataclasses import dataclass
from functools import lru_cache
import gc
import psutil
import time
import json

from utils.utils import check_gpu_availability, get_system_stats
from services.audio_processor import AudioProcessor
from services.sentiment_analysis import SentimentAnalyzer
# from topics_inf import TopicClassifier

class MemoryManager:
    """Manages memory usage and prevents OOM errors"""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.current_usage = 0
        self.lock = threading.Lock()
    
    def check_memory(self) -> bool:
        """Check if we have enough memory for processing"""
        with self.lock:
            current = psutil.virtual_memory().used
            return current < self.max_memory_bytes * 0.8
    
    def wait_for_memory(self, timeout: int = 60):
        """Wait until enough memory is available"""
        start_time = time.time()
        while not self.check_memory() and (time.time() - start_time) < timeout:
            time.sleep(1)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class AudioFileScanner:
    """Efficiently scans and validates audio files"""
    
    def __init__(self, config: dict):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.get('io_workers', 32))
    
    def scan_files_parallel(self, input_dir: Path) -> List[Path]:
        """Scan audio files in parallel with validation"""
        files = list(input_dir.rglob('*.wav'))
        files.extend(input_dir.rglob('*.ogg'))
        
        # Validate files in parallel
        valid_files = []
        futures = []
        
        for file in files:
            future = self.executor.submit(self._validate_file, file)
            futures.append((file, future))
        
        for file, future in futures:
            try:
                if future.result(timeout=10):
                    valid_files.append(file)
            except Exception as e:
                logging.warning(f"Failed to validate {file}: {e}")
        
        return valid_files
    
    def _validate_file(self, file_path: Path) -> bool:
        """Validate a single audio file"""
        try:
            if not file_path.exists():
                logging.warning(f"File does not exist: {file_path}")
                return False

            # Quick header check
            info = torchaudio.info(str(file_path))

            if info.num_frames == 0:
                logging.warning(f"Empty audio file: {file_path}")
                return False

            # Validate channel count (expecting mono)
            if info.num_channels == 1:
                logging.warning(f"Mono file detected (num_channels={info.num_channels}): {file_path}")
                return False
        
            #  All checks passed
            return True

        except Exception as e:
            logging.error(f"Error validating {file_path}: {e}")
            return False

class DataProcessor:
    """Main optimized processor for thousands of audio files"""
    
    def __init__(self, config: dict = None):
        self.config = config
        # Initialize ProcessingConfig from YAML with safe caps
        self.memory_manager = MemoryManager(self.config.get('max_memory_gb', 64.0))
        self.file_scanner = AudioFileScanner(self.config)
        
        self.setup_logging()
        self.setup_models()
        
        # Processing queues
        self.file_queue = Queue(maxsize=100)
        self.chunk_queue = Queue(maxsize=500)
        self.result_queue = Queue(maxsize=100)
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'chunks_processed': 0,
            'errors': 0,
            'start_time': None,
            'files_success': 0,
            'files_failed': 0,
            'files_skipped': 0
        }
        # Paths for idempotence and intermediate artifacts
        self.output_dir = Path(self.config['output_folder'])
        self.intermediate_dir = self.output_dir / 'intermediate'
        self.processed_markers_dir = self.output_dir / 'processed_markers'
        self.temp_dir = Path(self.config.get('temp_dir', '/tmp/audio_processing'))
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.processed_markers_dir.mkdir(parents=True, exist_ok=True)
        # Per-run file statuses
        self._file_statuses: List[Dict[str, Any]] = []

    def setup_logging(self):
        """Setup logging with rotation"""
        logs_dir = Path(self.config['logs_folder'])
        logs_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"optimized_processing_{timestamp}.log"
        
        from logging.handlers import TimedRotatingFileHandler
        file_handler = TimedRotatingFileHandler(str(log_file), when='D', interval=1, backupCount=14, utc=False)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        # Clear existing handlers to avoid duplicates on re-init
        for h in list(root.handlers):
            root.removeHandler(h)
        root.addHandler(file_handler)
        root.addHandler(console_handler)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline initialized")
    
    def setup_models(self):
        """Setup models with memory optimization"""
        self.logger.info("Setting up models")
        
        # Create a config compatible with AudioProcessor
        audio_processor_config = self.config.copy()
        
        # Map optimized config keys to AudioProcessor expected keys
        audio_processor_config['batch_size'] = self.config.get('chunk_batch_size', 16)
        
        # Ensure all required keys are present
        required_keys = [
            'batch_size', 'chunk_duration_sec', 'overlap_sec', 'target_sample_rate',
            'transcription_model', 'input_folder', 'output_folder', 'logs_folder'
        ]
        
        for key in required_keys:
            if key not in audio_processor_config:
                if key == 'chunk_duration_sec':
                    audio_processor_config[key] = self.config.get('max_chunk_duration', 25)
                elif key == 'overlap_sec':
                    audio_processor_config[key] = 1  # Default overlap
                elif key == 'target_sample_rate':
                    audio_processor_config[key] = 16000  # Default sample rate
                else:
                    self.logger.warning(f"Missing config key: {key}")
        
        # Pass database manager to audio processor if available
        db_manager = getattr(self, 'db_manager', None)
        self.audio_processor = AudioProcessor(audio_processor_config, db_manager)
        self.audio_processor.load_models()
        
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentAnalyzer(audio_processor_config)
            self.logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            self.logger.warning(f"Could not initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = None
    
    def _get_file_size_mb(self, file_path: Path) -> float:

        """Get file size in MB"""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            self.logger.warning(f"Could not get size for {file_path}: {e}")
            return 0.0

    def create_file_batches(self, files: List[Path]) -> List[List[Path]]:
        """Create optimized batches of files based on size (under 30MB per batch)"""
        max_batch_size_mb = self.config.get('max_batch_size_mb', 24.0)
        max_files_per_batch = self.config.get('file_batch_size', 16)  # Safety limit
        
        batches = []
        current_batch = []
        current_batch_size_mb = 0.0
        
        # Sort files by size (largest first) to optimize batching
        files_with_sizes = [(f, self._get_file_size_mb(f)) for f in files]
        files_with_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for file_path, file_size_mb in files_with_sizes:
            # Check if adding this file would exceed size limit
            if (current_batch_size_mb + file_size_mb > max_batch_size_mb and current_batch) or \
               len(current_batch) >= max_files_per_batch:
                # Start a new batch
                batches.append([f for f, _ in current_batch])
                current_batch = [(file_path, file_size_mb)]
                current_batch_size_mb = file_size_mb
            else:
                # Add to current batch
                current_batch.append((file_path, file_size_mb))
                current_batch_size_mb += file_size_mb
        
        # Add the last batch if it has files
        if current_batch:
            batches.append([f for f, _ in current_batch])
        
        # Log batch statistics
        total_files = len(files)
        avg_batch_size = sum(len(batch) for batch in batches) / len(batches) if batches else 0
        self.logger.info(f"Created {len(batches)} file batches (max size: {max_batch_size_mb}MB)")
        self.logger.info(f"Average files per batch: {avg_batch_size:.1f}")
        
        # Log size distribution
        for i, batch in enumerate(batches):
            batch_size_mb = sum(self._get_file_size_mb(f) for f in batch)
            self.logger.debug(f"Batch {i+1}: {len(batch)} files, {batch_size_mb:.1f}MB")
        
        return batches

    def _is_already_processed(self, file_path: Path) -> bool:
        """Check processed markers or DB to ensure idempotence."""
        try:
            base = file_path.stem
            marker = self.processed_markers_dir / f"{base}.done"
            if marker.exists():
                return True
            # Optional DB check
            # db_manager = getattr(self, 'db_manager', None)
            # if db_manager is not None:
            #     existing = db_manager.get_call_by_id_enregistrement(base)
            #     if existing:
            #         return True
        except Exception:
            pass
        return False

    def _mark_file_processed(self, file_path: Path):
        """Create a marker indicating this file was processed successfully."""
        try:
            base = file_path.stem
            marker = self.processed_markers_dir / f"{base}.done"
            marker.write_text(datetime.now().isoformat())
        except Exception:
            self.logger.warning(f"Failed to create processed marker for {file_path}")

    def _mark_file_failed(self, file_path: Path, error: str = ""):
        """Create a marker indicating this file failed processing."""
        try:
            base = file_path.stem
            marker = self.processed_markers_dir / f"{base}.failed"
            content = { 'timestamp': datetime.now().isoformat(), 'error': error }
            import json
            marker.write_text(json.dumps(content))
        except Exception:
            self.logger.warning(f"Failed to create failed marker for {file_path}")

    def _cleanup_old_artifacts(self):
        """Delete temp/intermediate/markers older than retention_days; optionally audio."""
        retention_days = int(self.config.get('retention_days', 30))
        delete_processed_audio = bool(self.config.get('delete_processed_files', False))
        cutoff = datetime.now() - pd.Timedelta(days=retention_days)

        def _cleanup_dir(d: Path):
            if not d.exists():
                return
            for p in d.glob('**/*'):
                try:
                    if p.is_file():
                        mtime = datetime.fromtimestamp(p.stat().st_mtime)
                        if mtime < cutoff:
                            p.unlink(missing_ok=True)
                except Exception:
                    pass

        _cleanup_dir(self.temp_dir)
        _cleanup_dir(self.intermediate_dir)
        _cleanup_dir(self.processed_markers_dir)

        if delete_processed_audio:
            input_dir = Path(self.config['input_folder'])
            for audio in input_dir.rglob('*.wav'):
                base = audio.stem
                marker = self.processed_markers_dir / f"{base}.done"
                try:
                    if marker.exists():
                        mtime = datetime.fromtimestamp(marker.stat().st_mtime)
                        if mtime < cutoff:
                            audio.unlink(missing_ok=True)
                except Exception:
                    pass

    def process_files_parallel(self, files: List[Path]) -> List[Dict]:
        """
        Process files in parallel with memory management.
        """
        self.logger.info(f"Starting parallel processing of {len(files)} files")
        self.stats.setdefault('errors', 0)
        self.stats.setdefault('files_skipped', 0)   
        self.stats['start_time'] = datetime.now()

        # Idempotence filter
        filtered_files = [f for f in files if not self._is_already_processed(f)]
        skipped = len(files) - len(filtered_files)
        if skipped:
            self.logger.info(f"Skipping {skipped} files already processed.")
        self.stats['files_skipped'] += skipped

        # Create batches (assume your method returns an iterable of batches)
        file_batches = list(self.create_file_batches(filtered_files))
        total_batches = len(file_batches)
        if total_batches == 0:
            self.logger.info("No batches to process after filtering.")
            return []

        max_workers = self.config.get('max_workers', 32)
        # self.logger.info(f"Using max_workers={max_workers}")

        # Windowed submission to avoid huge pending queues
        MAX_IN_FLIGHT = max(4, max_workers)  # tune as needed
        MAX_ERRORS = 10                           # stop early if things are going south

        all_results = 0
        in_flight = set()

        def submit_batch(executor, batch_id, batch):
            # Wait for memory before creating work
            self.memory_manager.wait_for_memory()
            fut = executor.submit(self.process_file_batch, batch_id, batch)
            fut.batch_id = batch_id  # attach context for logging
            in_flight.add(fut)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Prime the window
            next_idx = 0
            while next_idx < min(MAX_IN_FLIGHT, total_batches):
                submit_batch(executor, next_idx, file_batches[next_idx])
                next_idx += 1

            completed = 0
            while in_flight:
                for fut in as_completed(list(in_flight)):
                    in_flight.remove(fut)
                    bid = getattr(fut, "batch_id", "?")
                    try:
                        # Per-batch timeout (5 minutes)
                        batch_results = fut.result(timeout=300)
                        all_results += batch_results                            
                    except TimeoutError:
                        self.logger.error(f"Batch {bid} timed out after 300s")
                        self.stats['errors'] += 1
                    except Exception as e:
                        # Exception with traceback
                        self.logger.exception(f"Batch {bid} failed: {e}")
                        self.stats['errors'] += 1

                    completed += 1
                    if completed % max(1, total_batches // 10) == 0:
                        self.logger.info(f"Progress: {completed}/{total_batches} batches done")

                    # Early abort if too many errors
                    if self.stats['errors'] >= MAX_ERRORS:
                        self.logger.error(f"Aborting after {self.stats['errors']} errors; "
                                        f"cancelling {len(in_flight)} pending batches")
                        for pending in in_flight:
                            pending.cancel()
                        in_flight.clear()
                        break

                    # Keep the window full
                    if next_idx < total_batches and len(in_flight) < MAX_IN_FLIGHT:
                        submit_batch(executor, next_idx, file_batches[next_idx])
                        next_idx += 1

        self.logger.info(f"Finished: {completed}/{total_batches} batches; "
                        f"errors={self.stats['errors']}; results={all_results}")
        return all_results

    # def process_files_parallel(self, files: List[Path]) -> List[Dict]:
    #     """Process files in parallel with memory management"""
    #     self.logger.info(f"Starting parallel processing of {len(files)} files")
    #     self.stats['start_time'] = datetime.now()
        
    #     # Idempotence filter: skip already processed files
    #     filtered_files = [f for f in files if not self._is_already_processed(f)]
    #     skipped = len(files) - len(filtered_files)
    #     if skipped:
    #         self.logger.info(f"Skipping {skipped} files already processed")
    #     self.stats['files_skipped'] += skipped

    #     # Create file batches
    #     file_batches = self.create_file_batches(filtered_files)
        
    #     # Process batches with thread pool
    #     with ThreadPoolExecutor(max_workers=self.processing_config.max_workers) as executor:
    #         futures = []
            
    #         for batch_id, batch in enumerate(file_batches):
    #             # Wait for memory if needed
    #             self.memory_manager.wait_for_memory()
                
    #             future = executor.submit(self.process_file_batch, batch_id, batch)
    #             futures.append(future)
            
    #         # Collect results
    #         all_results = []
    #         for future in as_completed(futures):
    #             try:
    #                 batch_results = future.result(timeout=300)  # 5 minute timeout
    #                 all_results.extend(batch_results)
    #             except Exception as e:
    #                 self.logger.error(f"Batch processing error: {e}")
    #                 self.stats['errors'] += 1
        
    #     return all_results
        
    #     self.log_results(all_results)
    
    def process_file_batch(self, batch_id: int, files: List[Path]) -> int:
        """Process a batch of files using the optimized audio processor"""
        self.logger.info(f"Processing batch {batch_id + 1} with {len(files)} files")
        
        try:
            batch_results = self.audio_processor.process_batch(batch_id, files)

            self.stats['files_processed'] += len(files)
            self.stats['chunks_processed'] += len(batch_results)
            
            # Persist intermediate transcription results if enabled
            if self.config.get('save_intermediate_results', False) and batch_results:
                self._save_intermediate_transcriptions(batch_id, batch_results)

            # Add sentiment analysis if available
            if self.sentiment_analyzer and batch_results:
                self.logger.info(f"Adding sentiment analysis to {len(batch_results)} chunks of batch {batch_id + 1}")
                batch_results = self.sentiment_analyzer.analyze_batch_sentiment(batch_results)

            self.logger.info(f"Sentiment analysis completed for {len(batch_results)} chunks of batch {batch_id + 1}")
            if self.config.get('save_sentiment_analysis', False) and batch_results:
                self._save_chunks_analysis(batch_id, batch_results)

            # Determine which files failed in this batch
            failed_paths = set()
            failed_names = set()
            try:
                if hasattr(self.audio_processor, 'failed_files') and self.audio_processor.failed_files:
                    for item in self.audio_processor.failed_files:
                        p = item.get('path'); n = item.get('filename');
                        if p: failed_paths.add(p)
                        if n: failed_names.add(n)
            except Exception:
                pass

            # Mark per-file status, create markers, and collect statuses
            for fp in files:
                failed = (str(fp) in failed_paths) or (fp.name in failed_names)
                if failed:
                    err_msg = ''
                    try:
                        # find the error message for this file if present
                        err_msg = next((it.get('error','') for it in getattr(self.audio_processor, 'failed_files', []) if it.get('path') == str(fp) or it.get('filename') == fp.name), '')
                    except Exception:
                        pass
                    self._mark_file_failed(fp, err_msg)
                    self.stats['files_failed'] += 1
                    self._file_statuses.append({'file': str(fp), 'status': 'failed', 'error': err_msg})
                    self.logger.info(f"FILE STATUS - failed: {fp.name} - {err_msg}")
                else:
                    self._mark_file_processed(fp)
                    self.stats['files_success'] += 1
                    self._file_statuses.append({'file': str(fp), 'status': 'success'})
                    # self.logger.info(f"FILE STATUS - success: {fp.name}")
            del batch_results
            gc.collect()
            return len(files)
        except Exception as e:
            self.logger.error(f"Error processing batch {batch_id + 1}: {e}")
            self.stats['errors'] += 1
            return 0
    
    def log_results(self):
        """Log processing results and statistics"""
        end_time = datetime.now()
        duration = end_time - self.stats['start_time']
        
        self.logger.info(f"Processing completed in {duration}")
        self.logger.info(f"Files processed: {self.stats['files_processed']}")
        self.logger.info(f"Chunks processed: {self.stats['chunks_processed']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        
        
        # Always write a run summary JSON and failed-calls JSON
        try:
            summary = {
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None,
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'files_processed': self.stats['files_processed'] - self.stats['files_failed'],
                'chunks_processed': self.stats['chunks_processed'],
                'errors': self.stats['errors'],
                'files_success': self.stats.get('files_success', 0),
                'files_failed': self.stats.get('files_failed', 0),
                'files_skipped': self.stats.get('files_skipped', 0),
                'config_snapshot': {
                    'file_batch_size': self.config.get('file_batch_size', 8),
                    'chunk_batch_size': self.config.get('chunk_batch_size', 16),
                    'max_workers': self.config.get('max_workers', 32),
                    'io_workers': self.config.get('io_workers', 32),
                }
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = self.output_dir / f'run_summary_{run_id}.json'
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            self.logger.info(f"Run summary written to {summary_path}")

            # Failed calls file (aggregate per-file failures)
            failed_calls = []
            try:
                if hasattr(self, 'audio_processor') and getattr(self.audio_processor, 'failed_files', None):
                    # Enrich with any chunk-level errors found in results
                    file_to_chunks: Dict[str, List[Dict[str, Any]]] = {}
                    for item in self.audio_processor.failed_files:
                        fname = item.get('filename')
                        if fname in file_to_chunks:
                            item['chunks'] = file_to_chunks[fname]
                        failed_calls.append(item)
            except Exception:
                pass
            failed_path = self.output_dir / f'failed_calls_{run_id}.json'
            with open(failed_path, 'w') as f:
                json.dump({'failed': failed_calls}, f, indent=2)
            self.logger.info(f"Failed calls written to {failed_path}")

            # File status list for the run
            statuses_path = self.output_dir / f'file_statuses_{run_id}.json'
            with open(statuses_path, 'w') as f:
                json.dump({'files': self._file_statuses}, f, indent=2)
            self.logger.info(f"Per-file statuses written to {statuses_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write run summary: {e}")
    
    def save_results(self, results: List[Dict]):
        """Save results with optimized I/O"""
        output_dir = Path(self.config['output_folder'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data efficiently
        output_data = []
        for result in results:
            # Check if this is a failed result (has error and no transcription)
            if result.get('error') and not result.get('transcription_chunk'):
                output_data.append({
                    'file_name': result['file_name'],
                    'error': result['error'],
                    'agent_transcription': '',
                    'client_transcription': '',
                    'transcription_chunk': '',
                    # Sentiment analysis columns
                    'agent_text_sentiment': '',
                    'agent_text_confidence': 0.0,
                    'agent_acoustic_sentiment': '',
                    'agent_acoustic_confidence': 0.0,
                    'agent_fusion_sentiment': '',
                    'agent_fusion_confidence': 0.0,
                    'client_text_sentiment': '',
                    'client_text_confidence': 0.0,
                    'client_acoustic_sentiment': '',
                    'client_acoustic_confidence': 0.0,
                    'client_fusion_sentiment': '',
                    'client_fusion_confidence': 0.0,
                })
            else:
                output_data.append({
                    'file_name': result['file_name'],
                    'agent_transcription': result.get('agent_transcription', ''),
                    'client_transcription': result.get('client_transcription', ''),
                    'transcription_chunk': result.get('transcription_chunk', ''),
                    'error': result.get('error', ''),
                    # Sentiment analysis columns
                    'agent_text_sentiment': result.get('agent_text_sentiment', ''),
                    'agent_text_confidence': result.get('agent_text_confidence', 0.0),
                    'agent_acoustic_sentiment': result.get('agent_acoustic_sentiment', ''),
                    'agent_acoustic_confidence': result.get('agent_acoustic_confidence', 0.0),
                    'agent_fusion_sentiment': result.get('agent_fusion_sentiment', ''),
                    'agent_fusion_confidence': result.get('agent_fusion_confidence', 0.0),
                    'client_text_sentiment': result.get('client_text_sentiment', ''),
                    'client_text_confidence': result.get('client_text_confidence', 0.0),
                    'client_acoustic_sentiment': result.get('client_acoustic_sentiment', ''),
                    'client_acoustic_confidence': result.get('client_acoustic_confidence', 0.0),
                    'client_fusion_sentiment': result.get('client_fusion_sentiment', ''),
                    'client_fusion_confidence': result.get('client_fusion_confidence', 0.0),
                })
        
        # Save with compression
        if self.config['output_format'] == 'csv':
            output_file = output_dir / f"optimized_results_{timestamp}.csv"
            df = pd.DataFrame(output_data)
            df.to_csv(output_file, index=False)
            self.logger.info(f"Results saved: {output_file}")

    def _save_intermediate_transcriptions(self, batch_id: int, batch_results: List[Dict]):
        """Persist transcription-only results for a batch to intermediate storage."""
        try:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = self.intermediate_dir / f"batch_{batch_id+1}_{timestamp}.jsonl"
            # Strip heavy tensors; keep essential fields
            def strip_chunk(c: Dict) -> Dict:
                return {
                    'file_name': c.get('file_name', ''),
                    'chunk_idx': c.get('chunk_idx', 0),
                    'start_time': c.get('start_time', 0.0),
                    'end_time': c.get('end_time', 0.0),
                    'transcription_chunk': c.get('transcription_chunk', ''),
                    'agent_transcription': c.get('agent_transcription', ''),
                    'client_transcription': c.get('client_transcription', ''),
                    'error': c.get('error', ''),
                }
            
            with open(out_path, 'w') as f:
                for c in batch_results:
                    f.write(json.dumps(strip_chunk(c), ensure_ascii=False) + "\n")
            self.logger.info(f"Intermediate batch written: {out_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write intermediate results for batch {batch_id+1}: {e}")
        
    def _save_chunks_analysis(self, batch_id: int, batch_results: List[Dict]):
        """Persist sentiment analysis results for a batch to intermediate storage."""
        try:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = self.intermediate_dir / f"batch_{batch_id+1}_chunks_analysis_{timestamp}.jsonl"
            def strip_chunk(c: Dict) -> Dict:
                return {
                    'file_name': c.get('file_name', ''),
                    'chunk_idx': c.get('chunk_idx', 0),
                    'agent_text_sentiment': c.get('agent_text_sentiment', ''),
                    'agent_text_confidence': c.get('agent_text_confidence', 0.0),
                    'agent_acoustic_sentiment': c.get('agent_acoustic_sentiment', ''),
                    'agent_acoustic_confidence': c.get('agent_acoustic_confidence', 0.0),
                    'agent_fusion_sentiment': c.get('agent_fusion_sentiment', ''),
                    'agent_fusion_confidence': c.get('agent_fusion_confidence', 0.0),
                    'client_text_sentiment': c.get('client_text_sentiment', ''),
                }
            with open(out_path, 'w') as f:
                for c in batch_results:
                    f.write(json.dumps(strip_chunk(c), ensure_ascii=False) + "\n")
            self.logger.info(f"Intermediate sentiment analysis written: {out_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write intermediate sentiment analysis for batch {batch_id+1}: {e}")
        
    def run(self):
        """Main processing loop with optimization"""
        self.logger.info("Starting optimized audio processing")
        
        # Check system requirements
        gpu_status, gpu_info = check_gpu_availability(self.config['gpu_index'])
        self.logger.info(f"GPU Status: {gpu_status}")
        self.logger.info(f"GPU Info: {gpu_info}")
        self.logger.info(f"System Stats: {get_system_stats()}")
        
        # Scan files
        input_dir = Path(self.config['input_folder'])
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
        files = self.file_scanner.scan_files_parallel(input_dir)
        # files = files[:2]
        self.logger.info(f"Found {len(files)} valid audio files")
        
        # Process files
        total_files_success = self.process_files_parallel(files)
        self.log_results()
        self.logger.info("Optimized audio processing completed successfully with %s files success" % total_files_success)

def main():
    """Entry point for optimized pipeline"""
    print("Optimized Audio Processor - Parallel Processing Pipeline")
    print("=" * 60)
    
    try:
        processor = DataProcessor()
        processor.run()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()