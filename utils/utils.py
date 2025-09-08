"""
Utility functions for audio processing
"""

import psutil
import torch
from typing import Optional, Tuple

def check_gpu_availability() -> Tuple[bool, str]:
    """Check GPU availability and return status"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        
        return True, f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
    else:
        return False, "No GPU available - using CPU"

def get_gpu_memory_usage() -> Optional[float]:
    """Get current GPU memory usage percentage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return (allocated / total) * 100
    return None

def get_system_stats() -> dict:
    """Get system resource statistics"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    stats = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'memory_available_gb': memory.available / (1024**3),
        'disk_free_gb': disk.free / (1024**3)
    }
    
    # Add GPU stats if available
    gpu_memory = get_gpu_memory_usage()
    if gpu_memory is not None:
        stats['gpu_memory_percent'] = gpu_memory
    
    return stats

def pad_chunk_waveforms(waveforms):
    """Pad chunk waveforms to the same length for batch processing."""
    if not waveforms:
        return torch.empty(0)
    
    max_length = max(wf.shape[1] for wf in waveforms)
    padded_waveforms = []
    
    for waveform in waveforms:
        if waveform.shape[1] < max_length:
            padding = max_length - waveform.shape[1]
            padded = torch.nn.functional.pad(waveform, (0, padding))
        else:
            padded = waveform
        padded_waveforms.append(padded)
    
    return torch.stack(padded_waveforms)

def remove_special_characters(text):
    import re
    if text is None:
        return ""
    chars_to_remove_regex = r'[\,\?\.\!\-\;:\"%\'\»\«\؟\(\)،\.]'
    return re.sub(chars_to_remove_regex, '', text.lower())