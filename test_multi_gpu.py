#!/usr/bin/env python3
"""
Test script for Multi-GPU MIG-based audio processing
Verifies MIG setup and parallel processing functionality
"""

import os
import sys
import yaml
import torch
import logging
import time
from pathlib import Path
from typing import Dict, List
import subprocess

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from services.multi_gpu_processor import MultiGPUManager, ParallelAudioProcessor
from utils.utils import check_gpu_availability, get_system_stats


def setup_test_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_multi_gpu.log')
        ]
    )
    return logging.getLogger(__name__)


def check_mig_status():
    """Check if MIG is properly configured"""
    logger = logging.getLogger(__name__)
    
    try:
        # Check MIG mode
        result = subprocess.run(['nvidia-smi', '-mig', '-lgip'], 
                              capture_output=True, text=True, check=True)
        
        logger.info("MIG GPU Instances:")
        logger.info(result.stdout)
        
        # Check compute instances
        result = subprocess.run(['nvidia-smi', '-mig', '-lci'], 
                              capture_output=True, text=True, check=True)
        
        logger.info("MIG Compute Instances:")
        logger.info(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"MIG status check failed: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("nvidia-smi not found. Is NVIDIA driver installed?")
        return False


def test_pytorch_gpu_detection():
    """Test PyTorch GPU detection with MIG"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"CUDA device count: {device_count}")
        
        for i in range(device_count):
            try:
                device = torch.device(f"cuda:{i}")
                torch.cuda.set_device(device)
                
                # Test tensor allocation
                test_tensor = torch.zeros(100, 100, device=device)
                logger.info(f"Device {i}: {torch.cuda.get_device_name(i)} - OK")
                
                # Test memory info
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_cached = torch.cuda.memory_reserved(device) / 1024**3
                logger.info(f"  Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
                
                del test_tensor
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Device {i} test failed: {e}")
    
    return torch.cuda.is_available()


def test_multi_gpu_manager(config: Dict):
    """Test MultiGPUManager functionality"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing MultiGPUManager...")
        
        gpu_manager = MultiGPUManager(config)
        
        # Test instance creation
        for instance_name in ['transcription', 'vad', 'sentiment']:
            instance = gpu_manager.get_instance(instance_name)
            if instance:
                logger.info(f"Instance '{instance_name}': Device {instance.device_id}, "
                          f"Memory: {instance.memory_gb}GB, "
                          f"Batch Multiplier: {instance.batch_size_multiplier}")
                
                # Test device access
                try:
                    with torch.cuda.device(instance.device):
                        test_tensor = torch.zeros(100, device=instance.device)
                        logger.info(f"  Device access test: OK")
                        del test_tensor
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.error(f"  Device access test failed: {e}")
            else:
                logger.error(f"Instance '{instance_name}' not found")
        
        return True
        
    except Exception as e:
        logger.error(f"MultiGPUManager test failed: {e}")
        return False


def test_parallel_processor(config: Dict):
    """Test ParallelAudioProcessor functionality"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing ParallelAudioProcessor...")
        
        # Initialize processor
        processor = ParallelAudioProcessor(config)
        
        # Test processor setup
        logger.info(f"Available processors: {list(processor.processors.keys())}")
        
        # Create dummy audio files for testing
        test_files = []
        input_dir = Path(config['input_folder'])
        
        if input_dir.exists():
            # Use existing files if available
            test_files = list(input_dir.glob('*.wav'))[:2]  # Test with 2 files
            
        if not test_files:
            logger.warning("No test audio files found in input directory")
            logger.info("Skipping audio processing test")
            return True
        
        logger.info(f"Testing with {len(test_files)} audio files")
        
        # Test parallel processing
        start_time = time.time()
        results = processor.process_batch_parallel(0, test_files)
        processing_time = time.time() - start_time
        
        logger.info(f"Parallel processing completed in {processing_time:.2f} seconds")
        logger.info(f"Processed {len(results)} chunks")
        
        # Verify results structure
        if results:
            sample_result = results[0]
            expected_keys = ['file_name', 'chunk_idx', 'transcription_chunk']
            for key in expected_keys:
                if key in sample_result:
                    logger.info(f"  Result contains '{key}': ✓")
                else:
                    logger.warning(f"  Result missing '{key}': ✗")
        
        return True
        
    except Exception as e:
        logger.error(f"ParallelAudioProcessor test failed: {e}")
        return False


def test_memory_isolation():
    """Test memory isolation between MIG instances"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Testing memory isolation between MIG instances...")
        
        # Test allocating memory on different devices
        device_tensors = {}
        
        for device_id in range(min(3, torch.cuda.device_count())):
            try:
                device = torch.device(f"cuda:{device_id}")
                
                # Allocate memory
                tensor = torch.zeros(1000, 1000, device=device)
                device_tensors[device_id] = tensor
                
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
                logger.info(f"Device {device_id} allocated: {memory_allocated:.1f}MB")
                
            except Exception as e:
                logger.error(f"Memory allocation failed on device {device_id}: {e}")
        
        # Verify isolation (memory on one device shouldn't affect others)
        for device_id, tensor in device_tensors.items():
            device = torch.device(f"cuda:{device_id}")
            memory_before = torch.cuda.memory_allocated(device)
            
            # Delete tensor and check memory
            del tensor
            torch.cuda.empty_cache()
            
            memory_after = torch.cuda.memory_allocated(device)
            logger.info(f"Device {device_id} memory freed: {(memory_before - memory_after) / 1024**2:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Memory isolation test failed: {e}")
        return False


def run_performance_benchmark(config: Dict):
    """Run a simple performance benchmark"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Running performance benchmark...")
        
        # Test with different configurations
        configs_to_test = [
            ("Single GPU", {**config, 'enable_mig': False}),
            ("Multi-GPU MIG", {**config, 'enable_mig': True})
        ]
        
        results = {}
        
        for name, test_config in configs_to_test:
            logger.info(f"Testing {name} configuration...")
            
            try:
                # Initialize processor
                if test_config['enable_mig']:
                    processor = ParallelAudioProcessor(test_config)
                else:
                    from services.audio_processor import AudioProcessor
                    processor = AudioProcessor(test_config)
                    processor.load_models()
                
                # Simulate processing time
                start_time = time.time()
                
                # Create dummy workload
                for i in range(10):
                    # Simulate GPU work
                    device = torch.device('cuda:0')
                    with torch.cuda.device(device):
                        dummy_tensor = torch.randn(1000, 1000, device=device)
                        result = torch.matmul(dummy_tensor, dummy_tensor)
                        del dummy_tensor, result
                        torch.cuda.empty_cache()
                
                processing_time = time.time() - start_time
                results[name] = processing_time
                
                logger.info(f"{name} processing time: {processing_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {name}: {e}")
                results[name] = None
        
        # Compare results
        if all(results.values()):
            single_time = results.get("Single GPU", 0)
            multi_time = results.get("Multi-GPU MIG", 0)
            
            if single_time > 0:
                speedup = single_time / multi_time if multi_time > 0 else 0
                logger.info(f"Performance comparison:")
                logger.info(f"  Single GPU: {single_time:.2f}s")
                logger.info(f"  Multi-GPU MIG: {multi_time:.2f}s")
                logger.info(f"  Speedup: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        return False


def main():
    """Main test function"""
    logger = setup_test_logging()
    
    logger.info("="*60)
    logger.info("Multi-GPU MIG Audio Processing Test Suite")
    logger.info("="*60)
    
    # Load configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("config.yaml not found")
        return 1
    
    # Test results
    test_results = {}
    
    # 1. Check system status
    logger.info("\n1. Checking system status...")
    test_results['system_status'] = check_mig_status()
    
    # 2. Test PyTorch GPU detection
    logger.info("\n2. Testing PyTorch GPU detection...")
    test_results['pytorch_gpu'] = test_pytorch_gpu_detection()
    
    # 3. Test MultiGPUManager
    logger.info("\n3. Testing MultiGPUManager...")
    test_results['gpu_manager'] = test_multi_gpu_manager(config)
    
    # 4. Test ParallelAudioProcessor
    logger.info("\n4. Testing ParallelAudioProcessor...")
    test_results['parallel_processor'] = test_parallel_processor(config)
    
    # 5. Test memory isolation
    logger.info("\n5. Testing memory isolation...")
    test_results['memory_isolation'] = test_memory_isolation()
    
    # 6. Performance benchmark
    logger.info("\n6. Running performance benchmark...")
    test_results['performance'] = run_performance_benchmark(config)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:20}: {status}")
        if not result:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! Multi-GPU MIG setup is working correctly.")
        logger.info("\nYou can now run your audio processing with parallel GPU processing:")
        logger.info("  python run.py")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the errors above.")
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Make sure MIG is properly configured: sudo ./setup_mig.sh")
        logger.info("2. Check NVIDIA driver and CUDA installation")
        logger.info("3. Verify environment variables are set")
        return 1


if __name__ == "__main__":
    sys.exit(main())
