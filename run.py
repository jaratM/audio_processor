#!/usr/bin/env python3
"""
Audio Processing Pipeline Runner
Demonstrates the pipeline for processing thousands of audio files
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.pipeline import DataProcessor
from services.performance_monitor import PerformanceMonitor
from services.database_manager import DatabaseManager
import importlib.util

# Import minio-access module
spec = importlib.util.spec_from_file_location("minio_access", "utils/minio-access.py")
minio_access = importlib.util.module_from_spec(spec)
spec.loader.exec_module(minio_access)


def setup_logging(log_level: str = "INFO"):
    """
    Sets up the logging configuration for the application.

    This function configures the logging system to output log messages to the console.
    The log level can be specified (default is "INFO"). The log format includes the timestamp,
    logger name, log level, and the message.
    
    Args:
        log_level (str): The logging level as a string (e.g., "INFO", "DEBUG", "ERROR").
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {config_path} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        sys.exit(1)


def validate_config(config: dict) -> bool:
    """Validate configuration settings"""
    required_paths = ['input_folder', 'output_folder', 'logs_folder']
    
    for path_key in required_paths:
        if path_key not in config:
            print(f"Missing required config: {path_key}")
            return False
        
        path = Path(config[path_key])
        if path_key == 'input_folder' and not path.exists():
            print(f"Input folder does not exist: {path}")
            return False
    
    # Create output directories if they don't exist
    for path_key in ['output_folder', 'logs_folder']:
        Path(config[path_key]).mkdir(parents=True, exist_ok=True)
    
    return True


def sync_minio_data(config: dict, logger, skip_sync=False):
    """Synchronize data from MinIO if enabled"""
    if skip_sync:
        logger.info("MinIO sync skipped via command line argument")
        return
    
    minio_config = config.get('minio', {})
    
    if not minio_config.get('enabled', False):
        logger.info("MinIO sync is disabled in configuration")
        return
    
    
    try:
        logger.info("Starting MinIO synchronization...")
        
        # Get input folder from main config
        local_folder = config.get('input_folder', './input')
        
        # Create MinIO sync manager instance
        minio_manager = minio_access.MinIOSyncManager(config=config)
        
        # Perform sync
        stats = minio_manager.sync_to_local(local_folder_path=local_folder)
        
        logger.info(f"MinIO sync completed: {stats['downloaded']} downloaded, "
                   f"{stats['skipped']} skipped, {stats['errors']} errors")
        
        if stats['errors'] > 0:
            logger.warning(f"MinIO sync completed with {stats['errors']} errors")
        
    except ValueError as e:
        logger.error(f"MinIO configuration error: {e}")
        logger.warning("Please check your environment variables (MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY)")
        logger.warning("Continuing with local files only...")
    except Exception as e:
        logger.error(f"MinIO synchronization failed: {e}")
        logger.warning("Continuing with local files only...")


def print_system_info(config: dict):
    """Print system information"""
    import psutil
    import torch
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    # Disk space check
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    min_free_gb = float(os.environ.get('MIN_FREE_DISK_GB', '5'))
    print(f"Disk Free: {free_gb:.1f} GB (min required: {min_free_gb:.1f} GB)")
    if free_gb < min_free_gb:
        print("Insufficient disk space. Aborting.")
        sys.exit(1)
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"GPU Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(config['gpu_index'])}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    
    print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Audio Processing Pipeline")
    parser.add_argument("--config", default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--dry-run", action="store_true",
                       help="Scan files without processing")
    parser.add_argument("--performance-report", action="store_true",
                       help="Generate detailed performance report")
    parser.add_argument("--save-mode", default="database", choices=["database", "csv"],
                       help="Where to save results: 'database' (default) or 'csv'")
    parser.add_argument("--no-minio-sync", action="store_true",
                       help="Skip MinIO synchronization even if enabled in config")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    global logger
    logger = logging.getLogger(__name__)    
    # Load and validate configuration
    config = load_config(args.config)
    if not validate_config(config):
        sys.exit(1)
    
    # Add database configuration and saving mode
    config['save_csv_results'] = (args.save_mode == 'csv')
    
    # Sync data from MinIO if enabled
    sync_minio_data(config, logger, skip_sync=args.no_minio_sync)
    
    # Print system information
    print_system_info(config)
    performance_monitor = None
    db_manager = None
    try:
        db_manager = None
        if args.save_mode == 'database':
            # Initialize database manager only in database mode
            logger.info("Initializing database...")
            db_manager = DatabaseManager(config)
            logger.info("Database initialized successfully")
        else:
            logger.info("Saving mode set to CSV. Skipping database initialization.")
        
        # Initialize performance monitor
        performance_monitor = PerformanceMonitor(config)
        performance_monitor.start_monitoring()


        # Initialize processor; attach database manager only if in database mode
        processor = DataProcessor(config)
        if db_manager is not None:
            processor.db_manager = db_manager  # Attach database manager to processor
            # Also attach to sub-components created during processor initialization
            if hasattr(processor, 'audio_processor') and processor.audio_processor:
                processor.audio_processor.db_manager = db_manager
        
        # Set database manager in sentiment analyzer for real-time saving (database mode only)
        if db_manager is not None and hasattr(processor, 'sentiment_analyzer') and processor.sentiment_analyzer:
            processor.sentiment_analyzer.set_database_manager(db_manager)
        
        if args.dry_run:
            # Just scan files
            logger.info("\n DRY RUN - Scanning files only")
            input_dir = Path(config['input_folder'])
            files = processor.file_scanner.scan_files_parallel(input_dir)
            logger.info(f"Found {len(files)} valid audio files")
            
            # Estimate processing time
            if files:
                estimated_time_per_file = 30  # seconds
                total_estimated_time = len(files) * estimated_time_per_file / 3600  # hours
                logger.info(f"Estimated processing time: {total_estimated_time:.1f} hours")
        else:
            # Retention cleanup before run
            try:
                processor._cleanup_old_artifacts()
            except Exception as e:
                print(f"Retention cleanup failed: {e}")
                
            processor.run()
            # Generate performance report
            if args.performance_report:
                # Save detailed report
                report_path = Path(config['output_folder']) / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                performance_monitor.save_performance_report(str(report_path))
        
        # Stop monitoring
        if performance_monitor is not None:
            performance_monitor.stop_monitoring()
        
        # Close database connection if used
        if db_manager is not None:
            db_manager.close()
        
    except KeyboardInterrupt:
        print("\n Processing interrupted by user")
        performance_monitor.stop_monitoring()
        if 'db_manager' in locals() and db_manager is not None:
            db_manager.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n Error during processing: {e}")
        logger.exception("Processing error")
        if 'db_manager' in locals() and db_manager is not None:
            db_manager.close()
        sys.exit(1)


if __name__ == "__main__":
    main() 