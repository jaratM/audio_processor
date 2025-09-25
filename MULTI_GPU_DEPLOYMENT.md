# Multi-GPU MIG Deployment Guide

## 🚀 H100 Multi-Instance GPU (MIG) Audio Processing

This guide helps you deploy true GPU parallelism on your H100 using Multi-Instance GPU (MIG) technology for unprecedented audio processing performance.

## 📋 Prerequisites

- NVIDIA H100 GPU with MIG support
- NVIDIA Driver 525+ 
- CUDA 12.0+
- PyTorch with CUDA support
- Root/sudo access for MIG configuration

## 🔧 Quick Setup

### 1. Configure MIG on H100

```bash
# Make setup script executable
chmod +x setup_mig.sh

# Run MIG configuration (requires sudo)
sudo ./setup_mig.sh
```

This creates 3 optimized GPU instances:
- **Instance 0**: Transcription Model (4g.20gb) - Primary workload
- **Instance 1**: VAD Pipeline (2g.10gb) - Voice activity detection  
- **Instance 2**: Sentiment Analysis (1g.5gb) - Text and acoustic analysis

### 2. Verify MIG Setup

```bash
# Check MIG instances
nvidia-smi mig -lgip

# Check compute instances  
nvidia-smi mig -lci

# Verify configuration
python test_multi_gpu.py
```

### 3. Enable Multi-GPU Processing

Edit `config.yaml`:

```yaml
# Enable MIG processing
enable_mig: true

# MIG instance configuration
mig_instances:
  transcription:
    device_id: 0
    memory_gb: 20
    batch_size_multiplier: 2.0
  vad:
    device_id: 1
    memory_gb: 10
    batch_size_multiplier: 1.0
  sentiment:
    device_id: 2
    memory_gb: 5
    batch_size_multiplier: 0.8
```

### 4. Run Parallel Processing

```bash
# Run with multi-GPU acceleration
python run.py
```

## 📊 Performance Gains

### Before (Single GPU Sequential)
```
Transcription → VAD → Sentiment → Post-processing
     GPU          GPU      GPU         CPU
   (waiting)   (waiting) (waiting)   (active)
```

### After (Multi-GPU Parallel)
```
Transcription || VAD || Sentiment || Post-processing
   GPU-0       GPU-1     GPU-2         CPU
  (active)   (active)  (active)     (active)
```

**Expected Performance Improvements:**
- **3x GPU Utilization**: All models run simultaneously
- **2-3x Overall Throughput**: Eliminate GPU idle time
- **Better Memory Efficiency**: 80GB HBM3 optimally distributed
- **Fault Isolation**: Individual model failures don't affect others

## 🔍 Monitoring and Verification

### Check GPU Usage
```bash
# Monitor all MIG instances
nvidia-smi

# Monitor specific instance
nvidia-smi -i 0

# Watch GPU utilization
watch -n 1 nvidia-smi
```

### Performance Monitoring
```bash
# Run comprehensive test
python test_multi_gpu.py

# Check logs
tail -f logs/optimized_processing_*.log
```

### Verify Parallel Processing
Look for these log entries:
```
INFO - Initializing Multi-GPU Parallel Audio Processor
INFO - MIG instance 'transcription' configured on device 0
INFO - MIG instance 'vad' configured on device 1 
INFO - MIG instance 'sentiment' configured on device 2
INFO - Processing batch X with Y files in parallel
```

## 🛠 Configuration Options

### Batch Size Optimization
```yaml
# Adjust for your workload
mig_instances:
  transcription:
    batch_size_multiplier: 2.5  # Larger batches for bigger instance
  vad:
    batch_size_multiplier: 1.0  # Standard batches
  sentiment:
    batch_size_multiplier: 0.6  # Smaller batches for limited memory
```

### Memory Allocation
```yaml
# Fine-tune memory allocation
mig_instances:
  transcription:
    memory_gb: 25    # Increase for larger models
  vad:
    memory_gb: 8     # Reduce if not needed
  sentiment:
    memory_gb: 7     # Adjust based on sentiment model size
```

## 🔧 Troubleshooting

### Common Issues

#### 1. MIG Not Available
```bash
# Check if MIG is supported
nvidia-smi --query-gpu=mig.mode.current --format=csv

# Enable MIG if disabled
sudo nvidia-smi -mig 1
```

#### 2. PyTorch Can't See MIG Instances
```bash
# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_MIG_VISIBLE_DEVICES=0,1,2
```

#### 3. Out of Memory Errors
Reduce batch sizes in config:
```yaml
chunk_batch_size: 8    # Reduce from 16
file_batch_size: 16    # Reduce from 32
```

#### 4. Model Loading Failures
Check specific device assignment:
```bash
# Test device access
python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

### Recovery Commands

#### Disable MIG
```bash
# Reset to single GPU mode
sudo nvidia-smi -mig 0
```

#### Restore Single GPU Processing
```yaml
# In config.yaml
enable_mig: false
```

#### Clear GPU Memory
```bash
# Reset all CUDA contexts
sudo fuser -v /dev/nvidia*
sudo kill -9 <pid>
```

## 📈 Performance Tuning

### Optimal Settings for H100

```yaml
# Recommended configuration for H100
max_workers: 16
io_workers: 32
chunk_batch_size: 20
file_batch_size: 32
max_memory_gb: 500

# MIG-specific optimizations
mig_instances:
  transcription:
    device_id: 0
    memory_gb: 25
    batch_size_multiplier: 2.5
  vad:
    device_id: 1  
    memory_gb: 12
    batch_size_multiplier: 1.2
  sentiment:
    device_id: 2
    memory_gb: 8
    batch_size_multiplier: 0.8
```

### Monitoring Commands

```bash
# GPU utilization per instance
nvidia-smi dmon -s pucvmet -d 1

# Memory usage tracking
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Power consumption
nvidia-smi --query-gpu=power.draw --format=csv -l 1
```

## 🧪 Testing and Validation

### Run Full Test Suite
```bash
python test_multi_gpu.py
```

### Performance Benchmark
```bash
# Compare single vs multi-GPU
python test_multi_gpu.py --benchmark

# Time specific operations
time python run.py --max-files 100
```

### Memory Stress Test
```bash
# Test with large batches
python test_multi_gpu.py --stress-test
```

## 🚫 Disable MIG (Rollback)

If you need to return to single GPU mode:

```bash
# 1. Stop all processes using GPU
sudo pkill -f python

# 2. Disable MIG
sudo ./setup_mig.sh --disable

# 3. Update config
# Set enable_mig: false in config.yaml

# 4. Restart processing
python run.py
```

## 📞 Support

If you encounter issues:

1. **Check Logs**: `logs/optimized_processing_*.log`
2. **Run Tests**: `python test_multi_gpu.py`
3. **Verify Setup**: `nvidia-smi mig -lgip`
4. **Check Environment**: `env | grep CUDA`

## 🎯 Next Steps

Once MIG is working:

1. **Monitor Performance**: Track processing speed improvements
2. **Optimize Batch Sizes**: Fine-tune for your specific workload
3. **Scale Up**: Process larger datasets with parallel GPU power
4. **Add More Models**: Distribute additional ML models across instances

Your H100 is now configured for maximum parallel audio processing throughput! 🚀
