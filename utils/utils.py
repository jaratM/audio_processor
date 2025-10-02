"""
Utility functions for audio processing
"""

import psutil
import torch
from typing import Optional, Tuple
import logging
from services.database_manager import DatabaseManager
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def check_gpu_availability(gpu_index: int = 0) -> Tuple[bool, str]:
    """Check GPU availability and return status"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_index >= gpu_count:
            return False, f"GPU index {gpu_index} not available (only {gpu_count} GPUs found)"
        
        gpu_name = torch.cuda.get_device_name(gpu_index)
        gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)  # GB
        
        return True, f"GPU {gpu_index}: {gpu_name} ({gpu_memory:.1f}GB)"
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

def load_metadata(db_manager: DatabaseManager, config: dict, logger: logging.Logger):
    """Load metadata from JSON files"""
    logger.info("Loading metadata from JSON files")


    def process_metadata_file(file_path, db_manager, logger):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                metadata_json = json.load(f)
            # Extract id_enregistrement from filename (strip extension)
            id_enregistrement = Path(file_path).stem
            # Add business type
            dest_num = metadata_json.get("DESTINATION_NUMBER")
            metadata_json["BUSINESS_TYPE"] = db_manager.business_type(dest_num)
            db_manager.insert_call_metadata(id_enregistrement, metadata_json)
            return (file_path, True, None)
        except Exception as e:
            logger.error(f"Failed to process metadata file {file_path}: {e}")
            return (file_path, False, str(e))

    # Find all JSON files in the metadata folder
    config = getattr(db_manager, "config", None)
    if config and "input_folder" in config:
        metadata_folder = config["input_folder"]
    else:
        # Fallback: try to infer from db_manager or use default
        metadata_folder = getattr(db_manager, "metadata_folder", "data/metadata")

    metadata_folder = Path(metadata_folder)
    if not metadata_folder.exists():
        logger.warning(f"Metadata folder {metadata_folder} does not exist.")
        return

    json_files = list(metadata_folder.glob("*.json"))
    if not json_files:
        logger.info(f"No metadata JSON files found in {metadata_folder}")
        return

    logger.info(f"Found {len(json_files)} metadata files. Loading concurrently...")

    results = []
    with ThreadPoolExecutor(max_workers=config.get('io_workers', 32)) as executor:
        future_to_file = {
            executor.submit(process_metadata_file, str(f), db_manager, logger): f
            for f in json_files
        }
        for future in as_completed(future_to_file):
            file_path, success, error = future.result()
            if not success:
                logger.warning(f"Failed to load metadata for {file_path}: {error}")
            results.append((file_path, success, error))

    loaded_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - loaded_count
    logger.info(f"Metadata loading complete: {loaded_count} succeeded, {failed_count} failed.")