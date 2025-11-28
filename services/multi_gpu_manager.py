#!/usr/bin/env python3
"""
Multi-GPU Manager for distributed audio processing
Automatically distributes workload across available GPUs (1-4)
"""

import os
import torch
import logging
from typing import List, Dict, Any
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass

# CRITICAL: Set spawn method for CUDA compatibility
mp.set_start_method('spawn', force=True)


@dataclass
class GPUWorkerConfig:
    """Configuration for a single GPU worker"""
    gpu_id: int
    config: dict
    worker_id: int


class MultiGPUManager:
    """Manages multi-GPU processing with automatic GPU detection"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.available_gpus = self._detect_gpus()
        self.num_workers = len(self.available_gpus)
        
        self.logger.info(f"Multi-GPU Manager initialized with {self.num_workers} GPU(s): {self.available_gpus}")
    
    def _detect_gpus(self) -> List[int]:
        """Detect available GPUs"""
        if not torch.cuda.is_available():
            self.logger.warning("No CUDA GPUs available, using CPU")
            return [0]  # Use CPU as fallback
        
        num_gpus = torch.cuda.device_count()
        max_gpus = self.config.get('max_gpus', 4)
        num_gpus = min(num_gpus, max_gpus)
        
        available = list(range(num_gpus))
        self.logger.info(f"Detected {num_gpus} GPU(s)")
        return available if available else [0]
    
    def distribute_batches(self, file_batches: List[List[Path]]) -> List[tuple]:
        """Distribute file batches across GPUs"""
        if not file_batches:
            return []
        
        # Distribute batches round-robin across GPUs
        distributed = []
        for batch_idx, batch in enumerate(file_batches):
            gpu_id = self.available_gpus[batch_idx % self.num_workers]
            distributed.append((batch_idx, gpu_id, batch))
        
        self.logger.info(f"Distributed {len(file_batches)} batches across {self.num_workers} GPU(s)")
        return distributed
    
    def process_batches_parallel(self, file_batches: List[List[Path]], processor_instance) -> int:
        """Process batches in parallel across multiple GPUs"""
        
        if self.num_workers == 1:
            # Single GPU - use original processing
            self.logger.info("Single GPU mode - using standard processing")
            return self._process_single_gpu(file_batches, processor_instance)
        
        # Multi-GPU processing
        self.logger.info(f"Multi-GPU mode - processing on {self.num_workers} GPUs")
        from datetime import datetime
        start_time = datetime.now()
        
        distributed_batches = self.distribute_batches(file_batches)
        
        # Create worker configs
        worker_configs = []
        for batch_id, gpu_id, batch in distributed_batches:
            worker_config = GPUWorkerConfig(
                gpu_id=gpu_id,
                config=self._create_worker_config(gpu_id),
                worker_id=batch_id
            )
            worker_configs.append((batch_id, gpu_id, batch, worker_config))
        
        # Process in parallel using separate processes (one per GPU)
        results_count = 0
        
        # Group batches by GPU
        gpu_batches = {}
        for batch_id, gpu_id, batch, worker_config in worker_configs:
            if gpu_id not in gpu_batches:
                gpu_batches[gpu_id] = []
            gpu_batches[gpu_id].append((batch_id, batch, worker_config))
        
        # Launch one process per GPU using spawn context
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=self.num_workers, mp_context=ctx) as executor:
            futures = []
            
            for gpu_id, batches_for_gpu in gpu_batches.items():
                # Create worker config for this GPU
                worker_config = self._create_worker_config(gpu_id)
                
                future = executor.submit(
                    self._process_gpu_batches,
                    gpu_id,
                    batches_for_gpu,
                    worker_config  # Use worker config, not self.config
                )
                futures.append(future)
            
            # Collect results with progress tracking
            completed_workers = 0
            total_workers = len(futures)
            for future in as_completed(futures):
                try:
                    batch_count = future.result(timeout=3600)
                    results_count += batch_count
                    completed_workers += 1
                    self.logger.info(f"GPU worker {completed_workers}/{total_workers} completed: {batch_count} files processed")
                except Exception as e:
                    self.logger.error(f"GPU worker failed: {e}", exc_info=True)
                    completed_workers += 1
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Aggregate worker summaries
        self._aggregate_worker_summaries(start_time, end_time, duration)
        
        self.logger.info(f"Multi-GPU processing completed in {duration}: {results_count} total files processed across {self.num_workers} GPUs")
        return results_count
    
    def _process_single_gpu(self, file_batches: List[List[Path]], processor_instance) -> int:
        """Process all batches on single GPU (fallback)"""
        results_count = 0
        for batch_id, batch in enumerate(file_batches):
            try:
                batch_count = processor_instance.process_file_batch(batch_id, batch)
                results_count += batch_count
            except Exception as e:
                self.logger.error(f"Batch {batch_id} failed: {e}")
        return results_count
    
    def _create_worker_config(self, gpu_id: int) -> dict:
        """Create config for a specific GPU worker"""
        worker_config = self.config.copy()
        worker_config['gpu_index'] = gpu_id
        
        # Map config keys for AudioProcessor compatibility
        if 'chunk_batch_size' in worker_config and 'batch_size' not in worker_config:
            worker_config['batch_size'] = worker_config['chunk_batch_size']
        if 'max_chunk_duration' in worker_config and 'chunk_duration_sec' not in worker_config:
            worker_config['chunk_duration_sec'] = worker_config['max_chunk_duration']
        if 'overlap_sec' not in worker_config:
            worker_config['overlap_sec'] = 1
        if 'target_sample_rate' not in worker_config:
            worker_config['target_sample_rate'] = 16000
            
        return worker_config
    
    def _aggregate_worker_summaries(self, start_time, end_time, duration):
        """Aggregate summaries from all GPU workers and create comprehensive report"""
        import json
        from datetime import datetime
        
        output_dir = Path(self.config.get('output_folder', './output'))
        
        # Find all GPU worker summary files
        gpu_summaries = list(output_dir.glob('gpu_*_summary_*.json'))
        
        if not gpu_summaries:
            self.logger.warning("No GPU worker summaries found")
            return
        
        # Aggregate statistics
        total_files = 0
        total_success = 0
        total_failed = 0
        total_chunks = 0
        all_file_statuses = []
        all_failed_files = []
        gpu_stats = []
        
        for summary_file in gpu_summaries:
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    
                total_files += summary.get('files_processed', 0)
                total_success += summary.get('files_success', 0)
                total_failed += summary.get('files_failed', 0)
                total_chunks += summary.get('chunks_processed', 0)
                
                # Collect file statuses
                file_statuses = summary.get('file_statuses', [])
                all_file_statuses.extend(file_statuses)
                
                # Collect failed files
                for status in file_statuses:
                    if status.get('status') == 'failed':
                        all_failed_files.append({
                            'file': status.get('file'),
                            'error': status.get('error', ''),
                            'gpu_id': status.get('gpu_id')
                        })
                
                # Track per-GPU stats
                gpu_stats.append({
                    'gpu_id': summary.get('gpu_id'),
                    'files_processed': summary.get('files_processed', 0),
                    'files_success': summary.get('files_success', 0),
                    'files_failed': summary.get('files_failed', 0),
                    'chunks_processed': summary.get('chunks_processed', 0)
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to read GPU summary {summary_file}: {e}")
        
        # Create comprehensive summary
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        comprehensive_summary = {
            'multi_gpu_processing': True,
            'num_gpus': self.num_workers,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'total_files_processed': total_files,
            'total_files_success': total_success,
            'total_files_failed': total_failed,
            'total_chunks_processed': total_chunks,
            'gpu_stats': gpu_stats,
            'config_snapshot': {
                'file_batch_size': self.config.get('file_batch_size', 8),
                'chunk_batch_size': self.config.get('chunk_batch_size', 16),
                'max_workers': self.config.get('max_workers', 32),
                'io_workers': self.config.get('io_workers', 32),
                'max_gpus': self.config.get('max_gpus', 4),
            }
        }
        
        # Write comprehensive summary
        try:
            summary_path = output_dir / f'multi_gpu_run_summary_{run_id}.json'
            with open(summary_path, 'w') as f:
                json.dump(comprehensive_summary, f, indent=2)
            self.logger.info(f"Multi-GPU run summary written to {summary_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write comprehensive summary: {e}")
        
        # Write aggregated failed files
        try:
            failed_path = output_dir / f'multi_gpu_failed_calls_{run_id}.json'
            with open(failed_path, 'w') as f:
                json.dump({'failed': all_failed_files, 'count': len(all_failed_files)}, f, indent=2)
            self.logger.info(f"Multi-GPU failed calls written to {failed_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write failed calls: {e}")
        
        # Write aggregated file statuses
        try:
            statuses_path = output_dir / f'multi_gpu_file_statuses_{run_id}.json'
            with open(statuses_path, 'w') as f:
                json.dump({'files': all_file_statuses}, f, indent=2)
            self.logger.info(f"Multi-GPU file statuses written to {statuses_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write file statuses: {e}")
        
        # Log summary to console
        self.logger.info("="*80)
        self.logger.info("MULTI-GPU PROCESSING SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"GPUs used: {self.num_workers}")
        self.logger.info(f"Total files processed: {total_files}")
        self.logger.info(f"  - Success: {total_success}")
        self.logger.info(f"  - Failed: {total_failed}")
        self.logger.info(f"Total chunks processed: {total_chunks}")
        self.logger.info(f"\nPer-GPU Statistics:")
        for stat in gpu_stats:
            self.logger.info(f"  GPU {stat['gpu_id']}: {stat['files_processed']} files "
                           f"({stat['files_success']} success, {stat['files_failed']} failed, "
                           f"{stat['chunks_processed']} chunks)")
        self.logger.info("="*80)
    
    @staticmethod
    def _process_gpu_batches(gpu_id: int, batches: List[tuple], config: dict) -> int:
        """Process batches on a specific GPU (runs in separate process)"""
        # Set GPU for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Configure logging for this worker process
        import logging
        from pathlib import Path
        from datetime import datetime
        import json
        
        logs_dir = Path(config.get('logs_folder', './logs'))
        logs_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"gpu_{gpu_id}_{timestamp}.log"
        
        # Setup logging with both file and console handlers
        logger = logging.getLogger(f"GPU-{gpu_id}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # Clear any existing handlers
        
        # File handler
        file_handler = logging.FileHandler(str(log_file))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Console handler (will show in main log)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        
        # Import here to avoid pickling issues
        from services.audio_processor import AudioProcessor
        from services.sentiment_analysis import SentimentAnalyzer
        from services.database_manager import DatabaseManager
        
        logger.info(f"GPU {gpu_id} worker started with {len(batches)} batches")
        logger.info(f"GPU {gpu_id} logging to {log_file}")
        
        # Setup marker directories for idempotence
        output_dir = Path(config.get('output_folder', './output'))
        processed_markers_dir = output_dir / 'processed_markers'
        processed_markers_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models on this GPU
        config['gpu_index'] = 0  # After CUDA_VISIBLE_DEVICES, always use 0
        
        # Initialize database connection for this worker
        db_manager = None
        try:
            if not config.get('save_csv_results', False):
                db_manager = DatabaseManager(config)
                logger.info(f"GPU {gpu_id} database connection established")
        except Exception as e:
            logger.warning(f"GPU {gpu_id} failed to initialize database: {e}. Continuing without DB.")
            db_manager = None
        
        try:
            audio_processor = AudioProcessor(config, db_manager=db_manager)
            audio_processor.load_models()
            
            sentiment_analyzer = SentimentAnalyzer(config)
            
            # Set database manager for sentiment analyzer (for chunk saving)
            if db_manager:
                sentiment_analyzer.set_database_manager(db_manager)
            
            logger.info(f"GPU {gpu_id} models loaded successfully")
        except Exception as e:
            logger.error(f"GPU {gpu_id} failed to load models: {e}")
            if db_manager:
                db_manager.close()
            return 0
        
        # Setup intermediate directories for result persistence
        intermediate_dir = output_dir / 'intermediate'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Track file statuses for this GPU worker
        file_statuses = []
        
        # Helper functions for marking files
        def mark_file_processed(file_path: Path):
            """Create a marker indicating this file was processed successfully and remove the file."""
            try:
                base = file_path.stem
                marker = processed_markers_dir / f"{base}.done"
                marker.write_text(datetime.now().isoformat())
                
                # Remove the original audio file after marking as processed
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Removed processed file: {file_path.name}")
                
                # Remove the corresponding JSON file if it exists
                json_file = file_path.with_suffix('.json')
                if json_file.exists():
                    json_file.unlink()
                    logger.debug(f"Removed corresponding JSON file: {json_file.name}")
                
            except Exception as e:
                logger.warning(f"Failed to create processed marker or remove file {file_path}: {e}")
        
        def mark_file_failed(file_path: Path, error: str = ""):
            """Create a marker indicating this file failed processing."""
            try:
                base = file_path.stem
                marker = processed_markers_dir / f"{base}.failed"
                content = {'timestamp': datetime.now().isoformat(), 'error': error}
                marker.write_text(json.dumps(content))
            except Exception as e:
                logger.warning(f"Failed to create failed marker for {file_path}: {e}")
        
        def save_intermediate_transcriptions(batch_id: int, batch_results: list):
            """Persist transcription-only results for a batch to intermediate storage."""
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = intermediate_dir / f"gpu_{gpu_id}_batch_{batch_id}_{timestamp}.jsonl"
                
                def strip_chunk(c: dict) -> dict:
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
                logger.info(f"GPU {gpu_id} intermediate transcriptions written: {out_path}")
            except Exception as e:
                logger.warning(f"GPU {gpu_id} failed to write intermediate results for batch {batch_id}: {e}")
        
        def save_chunks_analysis(batch_id: int, batch_results: list):
            """Persist sentiment analysis results for a batch to intermediate storage."""
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = intermediate_dir / f"gpu_{gpu_id}_batch_{batch_id}_sentiment_{timestamp}.jsonl"
                
                def strip_chunk(c: dict) -> dict:
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
                        'client_text_confidence': c.get('client_text_confidence', 0.0),
                        'client_acoustic_sentiment': c.get('client_acoustic_sentiment', ''),
                        'client_acoustic_confidence': c.get('client_acoustic_confidence', 0.0),
                        'client_fusion_sentiment': c.get('client_fusion_sentiment', ''),
                        'client_fusion_confidence': c.get('client_fusion_confidence', 0.0),
                    }
                
                with open(out_path, 'w') as f:
                    for c in batch_results:
                        f.write(json.dumps(strip_chunk(c), ensure_ascii=False) + "\n")
                logger.info(f"GPU {gpu_id} sentiment analysis written: {out_path}")
            except Exception as e:
                logger.warning(f"GPU {gpu_id} failed to write sentiment analysis for batch {batch_id}: {e}")
        
        # Process batches
        results_count = 0
        files_success = 0
        files_failed = 0
        chunks_processed = 0
        total_batches = len(batches)
        
        for idx, (batch_id, batch_files, worker_config) in enumerate(batches, 1):
            try:
                logger.info(f"GPU {gpu_id} processing batch {idx}/{total_batches} (batch_id={batch_id}, {len(batch_files)} files)")
                
                # Transcription
                batch_results = audio_processor.process_batch(batch_id, batch_files)
                chunks_processed += len(batch_results)
                logger.info(f"GPU {gpu_id} transcription completed for batch {batch_id}: {len(batch_results)} chunks")
                
                # Save intermediate transcription results if enabled
                if config.get('save_intermediate_results', False) and batch_results:
                    save_intermediate_transcriptions(batch_id, batch_results)
                
                # Sentiment analysis
                if sentiment_analyzer and batch_results:
                    batch_results = sentiment_analyzer.analyze_batch_sentiment(batch_results)
                    logger.info(f"GPU {gpu_id} sentiment analysis completed for batch {batch_id}")
                    
                    # Save sentiment analysis results if enabled
                    if config.get('save_sentiment_analysis', False):
                        save_chunks_analysis(batch_id, batch_results)
                
                # Mark files as processed or failed based on audio_processor.failed_files
                failed_paths = set()
                failed_names = set()
                if hasattr(audio_processor, 'failed_files') and audio_processor.failed_files:
                    for item in audio_processor.failed_files:
                        p = item.get('path')
                        n = item.get('filename')
                        if p: failed_paths.add(p)
                        if n: failed_names.add(n)
                
                # Mark each file and track status
                for fp in batch_files:
                    failed = (str(fp) in failed_paths) or (fp.name in failed_names)
                    if failed:
                        err_msg = next((it.get('error', '') for it in audio_processor.failed_files 
                                       if it.get('path') == str(fp) or it.get('filename') == fp.name), '')
                        mark_file_failed(fp, err_msg)
                        files_failed += 1
                        file_statuses.append({'file': str(fp), 'status': 'failed', 'error': err_msg, 'gpu_id': gpu_id})
                        logger.info(f"GPU {gpu_id} FILE STATUS - failed: {fp.name} - {err_msg}")
                    else:
                        mark_file_processed(fp)
                        files_success += 1
                        file_statuses.append({'file': str(fp), 'status': 'success', 'gpu_id': gpu_id})
                        logger.debug(f"GPU {gpu_id} FILE STATUS - success: {fp.name}")
                
                results_count += len(batch_files)
                logger.info(f"GPU {gpu_id} completed batch {idx}/{total_batches} ({results_count} files processed, {files_success} success, {files_failed} failed)")
                
            except Exception as e:
                logger.error(f"GPU {gpu_id} batch {batch_id} failed: {e}", exc_info=True)
                # Mark all files in this batch as failed
                for fp in batch_files:
                    mark_file_failed(fp, str(e))
                    files_failed += 1
                    file_statuses.append({'file': str(fp), 'status': 'failed', 'error': str(e), 'gpu_id': gpu_id})
                    logger.info(f"GPU {gpu_id} marked as FAILED (batch error): {fp.name}")
        
        # Close database connection
        if db_manager:
            try:
                db_manager.close()
                logger.info(f"GPU {gpu_id} database connection closed")
            except Exception as e:
                logger.warning(f"GPU {gpu_id} failed to close database: {e}")
        
        # Write GPU worker summary
        try:
            summary = {
                'gpu_id': gpu_id,
                'total_batches': total_batches,
                'files_processed': results_count,
                'files_success': files_success,
                'files_failed': files_failed,
                'chunks_processed': chunks_processed,
                'file_statuses': file_statuses,
                'timestamp': datetime.now().isoformat()
            }
            
            summary_path = output_dir / f"gpu_{gpu_id}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"GPU {gpu_id} summary written to {summary_path}")
        except Exception as e:
            logger.warning(f"GPU {gpu_id} failed to write summary: {e}")
        
        logger.info(f"GPU {gpu_id} worker completed: {results_count} files processed ({files_success} success, {files_failed} failed, {chunks_processed} chunks)")
        logger.info(f"="*60)
        
        # Close handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        return results_count


def get_optimal_gpu_count(config: dict) -> int:
    """Determine optimal number of GPUs to use"""
    if not torch.cuda.is_available():
        return 1
    
    available = torch.cuda.device_count()
    max_gpus = config.get('max_gpus', 4)
    
    return min(available, max_gpus, 4)
