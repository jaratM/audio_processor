"""
Performance Monitoring and Optimization Utilities
"""

import time
import psutil
import torch
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import json
import os


@dataclass
class ProcessingMetrics:
    """Metrics for processing performance"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    files_processed: int = 0
    chunks_processed: int = 0
    total_audio_duration: float = 0.0
    processing_time: float = 0.0
    memory_peak_gb: float = 0.0
    gpu_memory_peak_gb: float = 0.0
    errors: int = 0
    throughput_files_per_hour: float = 0.0
    throughput_audio_per_hour: float = 0.0
    
    def calculate_throughput(self):
        """Calculate throughput metrics"""
        if self.end_time:
            duration_hours = (self.end_time - self.start_time).total_seconds() / 3600
            if duration_hours > 0:
                self.throughput_files_per_hour = self.files_processed / duration_hours
                self.throughput_audio_per_hour = self.total_audio_duration / duration_hours


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = ProcessingMetrics()
        self.batch_metrics = deque(maxlen=100)
        self.memory_history = deque(maxlen=1000)
        self.gpu_memory_history = deque(maxlen=1000)
        self.per_gpu_memory_history = defaultdict(lambda: deque(maxlen=1000))  # Per-GPU tracking
        self.custom_metrics_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Threading
        self.monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Performance tracking
        self.file_times = defaultdict(list)
        self.chunk_times = defaultdict(list)
        self.bottlenecks = defaultdict(int)
        self.callbacks: Dict[str, Callable[[], Any]] = {}
        
    def add_callback(self, name: str, func: Callable[[], Any]):
        """Add a callback to collect custom metrics"""
        self.callbacks[name] = func
        
    def start_monitoring(self):
        """Start background monitoring"""
        if self.config.get('enable_performance_monitoring', True):
            self.monitoring_thread = threading.Thread(target=self._monitor_resources)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Performance monitoring stopped")
    
    def _get_gpu_memory_nvidia_smi(self) -> Dict[int, float]:
        """Get GPU memory usage using nvidia-smi (works across processes)"""
        try:
            # query format: index, memory.used [MiB]
            # Use nounits to avoid parsing "MiB"
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            gpu_memory = {}
            for line in result.strip().split('\n'):
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0].strip())
                        used_mb = float(parts[1].strip())
                        # Convert MiB to GB
                        gpu_memory[idx] = used_mb / 1024
            return gpu_memory
        except Exception as e:
            # Don't log every second if it fails, just debug
            # self.logger.debug(f"nvidia-smi failed: {e}") 
            return {}

    def _monitor_resources(self):
        """Background thread to monitor system resources"""
        while not self._stop_monitoring.is_set():
            try:
                timestamp = datetime.now()
                
                # Monitor memory
                memory = psutil.virtual_memory()
                self.memory_history.append({
                    'timestamp': timestamp,
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                })
                
                # Monitor GPU memory
                max_gpu_memory = 0.0
                gpu_memory_map = {}
                
                # Try nvidia-smi first (works for multi-process)
                gpu_memory_map = self._get_gpu_memory_nvidia_smi()
                
                # Fallback to torch if nvidia-smi failed/empty but torch is available
                if not gpu_memory_map and torch.cuda.is_available():
                    try:
                        gpu_count = torch.cuda.device_count()
                        for i in range(gpu_count):
                            gpu_memory_map[i] = torch.cuda.memory_allocated(i) / (1024**3)
                    except Exception:
                        pass

                if gpu_memory_map:
                    gpu_index = self.config.get('gpu_index')
                    
                    for gpu_id, memory_gb in gpu_memory_map.items():
                        # If specific GPU is requested, only track that one
                        if gpu_index is not None and gpu_id != gpu_index:
                            continue
                            
                        max_gpu_memory = max(max_gpu_memory, memory_gb)
                        self.per_gpu_memory_history[gpu_id].append({
                            'timestamp': timestamp,
                            'used_gb': memory_gb
                        })
                    
                    self.gpu_memory_history.append({
                        'timestamp': timestamp,
                        'used_gb': max_gpu_memory
                    })

                # Collect custom metrics from callbacks
                for name, func in self.callbacks.items():
                    try:
                        val = func()
                        self.custom_metrics_history[name].append({
                            'timestamp': timestamp,
                            'value': val
                        })
                    except Exception as e:
                        self.logger.error(f"Error in callback {name}: {e}")

                # Update peak values
                current_memory_gb = memory.used / (1024**3)
                if current_memory_gb > self.metrics.memory_peak_gb:
                    self.metrics.memory_peak_gb = current_memory_gb
                
                if max_gpu_memory > self.metrics.gpu_memory_peak_gb:
                    self.metrics.gpu_memory_peak_gb = max_gpu_memory
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        self.metrics.end_time = datetime.now()
        self.metrics.calculate_throughput()
        
        # Calculate averages
        avg_file_time = 0
        if self.file_times:
            all_times = [t for times in self.file_times.values() for t in times]
            avg_file_time = sum(all_times) / len(all_times)
        
        avg_batch_time = 0
        if self.batch_metrics:
            avg_batch_time = sum(b['duration'] for b in self.batch_metrics) / len(self.batch_metrics)
        
        # Identify bottlenecks
        top_bottlenecks = sorted(self.bottlenecks.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = {
            'processing_time': (self.metrics.end_time - self.metrics.start_time).total_seconds(),
            'files_processed': self.metrics.files_processed,
            'chunks_processed': self.metrics.chunks_processed,
            'total_audio_duration': self.metrics.total_audio_duration,
            'throughput_files_per_hour': self.metrics.throughput_files_per_hour,
            'throughput_audio_per_hour': self.metrics.throughput_audio_per_hour,
            'memory_peak_gb': self.metrics.memory_peak_gb,
            'gpu_memory_peak_gb': self.metrics.gpu_memory_peak_gb,
            'errors': self.metrics.errors,
            'error_rate': self.metrics.errors / max(self.metrics.files_processed, 1),
            'avg_file_time': avg_file_time,
            'avg_batch_time': avg_batch_time,
            'top_bottlenecks': top_bottlenecks,
            'system_info': self._get_system_info()
        }
        
        return summary
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        # Add per-GPU peak memory if available
        if self.per_gpu_memory_history:
            per_gpu_peaks = {}
            for gpu_id, history in self.per_gpu_memory_history.items():
                if history:
                    peak = max(entry['used_gb'] for entry in history)
                    per_gpu_peaks[f'gpu_{gpu_id}_peak_gb'] = peak
            if per_gpu_peaks:
                info['per_gpu_memory_peaks'] = per_gpu_peaks
        
        return info
    
    def save_performance_report(self, output_path: str):
        """Save detailed performance report"""
        summary = self.get_performance_summary()
        
        # Add detailed metrics
        report = {
            'summary': summary,
            'batch_metrics': list(self.batch_metrics),
            'memory_history': list(self.memory_history),
            'gpu_memory_history': list(self.gpu_memory_history),
            'per_gpu_memory_history': {
                f'gpu_{gpu_id}': list(history) 
                for gpu_id, history in self.per_gpu_memory_history.items()
            },
            'custom_metrics': {
                name: list(history)
                for name, history in self.custom_metrics_history.items()
            },
            'file_times': dict(self.file_times),
            'bottlenecks': dict(self.bottlenecks)
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {output_path}")

    # Optimize resource usage based on performance data
    
    def optimize_batch_size(self, current_batch_size: int, avg_batch_time: float, 
                          memory_usage: float) -> int:
        """Optimize batch size based on performance"""
        target_batch_time = 30  # Target 30 seconds per batch
        
        if avg_batch_time > target_batch_time * 1.5:
            # Batch is too slow, reduce size
            new_size = max(1, int(current_batch_size * 0.8))
            self.logger.info(f"Reducing batch size from {current_batch_size} to {new_size}")
            return new_size
        elif avg_batch_time < target_batch_time * 0.5 and memory_usage < 0.7:
            # Batch is too fast and memory is available, increase size
            new_size = min(64, int(current_batch_size * 1.2))
            self.logger.info(f"Increasing batch size from {current_batch_size} to {new_size}")
            return new_size
        
        return current_batch_size
    
    def optimize_worker_count(self, current_workers: int, cpu_usage: float, 
                            io_wait: float) -> int:
        """Optimize worker count based on system load"""
        if cpu_usage > 0.9:
            # CPU is overloaded, reduce workers
            new_workers = max(1, int(current_workers * 0.8))
            self.logger.info(f"Reducing workers from {current_workers} to {new_workers}")
            return new_workers
        elif cpu_usage < 0.6 and io_wait < 0.1:
            # CPU is underutilized, increase workers
            new_workers = min(64, int(current_workers * 1.2))
            self.logger.info(f"Increasing workers from {current_workers} to {new_workers}")
            return new_workers
        
        return current_workers
    
    def get_memory_recommendations(self, peak_memory: float, total_memory: float) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        
        memory_usage_ratio = peak_memory / total_memory
        
        if memory_usage_ratio > 0.9:
            recommendations.append("Memory usage is very high. Consider reducing batch size.")
        elif memory_usage_ratio > 0.7:
            recommendations.append("Memory usage is high. Monitor for potential OOM errors.")
        
        if peak_memory > 16:
            recommendations.append("Peak memory usage > 16GB. Consider using memory mapping.")
        
        return recommendations
    
    def get_performance_recommendations(self, throughput: float, target_throughput: float) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if throughput < target_throughput * 0.5:
            recommendations.append("Throughput is very low. Check for bottlenecks.")
        elif throughput < target_throughput * 0.8:
            recommendations.append("Throughput is below target. Consider optimization.")
        
        return recommendations
    
    def get_current_gpu_memory(self) -> Dict[int, float]:
        """Get current GPU memory usage for all GPUs in GB"""
        # Try nvidia-smi first
        gpu_memory = self._get_gpu_memory_nvidia_smi()
        if gpu_memory:
            return gpu_memory
            
        # Fallback to torch
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for gpu_id in range(gpu_count):
                try:
                    memory_gb = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                    gpu_memory[gpu_id] = memory_gb
                except Exception as e:
                    self.logger.debug(f"Failed to get memory for GPU {gpu_id}: {e}")
                    gpu_memory[gpu_id] = 0.0
        return gpu_memory
    
    def log_gpu_memory_status(self):
        """Log current GPU memory status for all GPUs"""
        gpu_memory = self.get_current_gpu_memory()
        if gpu_memory:
            for gpu_id, memory_gb in gpu_memory.items():
                self.logger.info(f"GPU {gpu_id} memory: {memory_gb:.2f} GB")
        else:
            if torch.cuda.is_available():
                 self.logger.info("Could not retrieve GPU memory usage")
            else:
                self.logger.info("No CUDA GPUs available")
