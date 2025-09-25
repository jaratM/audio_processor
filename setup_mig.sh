#!/bin/bash

# MIG Setup Script for H100 GPU Parallelism
# This script configures your H100 GPU for Multi-Instance GPU (MIG) mode

set -e

echo "=== H100 MIG Configuration Setup ==="
echo "Configuring GPU for parallel audio processing..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Check if H100 is available
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -q "H100" || {
    echo "Warning: H100 GPU not detected. Continuing with available GPU..."
}

# Save current MIG state
echo "Saving current GPU state..."
nvidia-smi mig -lgip > /tmp/mig_state_backup.txt 2>/dev/null || echo "No existing MIG instances found"

# Reset any existing MIG configuration
echo "Resetting MIG configuration..."
nvidia-smi mig -dci || echo "No compute instances to destroy"
nvidia-smi mig -dgi || echo "No GPU instances to destroy"

# Disable MIG temporarily to reset
nvidia-smi -mig 0 || echo "MIG already disabled"
sleep 2

# Enable MIG mode
echo "Enabling MIG mode..."
nvidia-smi -mig 1
sleep 3

# Create GPU instances optimized for audio processing
echo "Creating optimized GPU instances for audio processing..."

# Instance 0: Large instance for transcription model (primary workload)
echo "Creating Instance 0: Transcription (4g.20gb)..."
nvidia-smi mig -cgi 4g.20gb -C

# Instance 1: Medium instance for VAD pipeline  
echo "Creating Instance 1: VAD Pipeline (2g.10gb)..."
nvidia-smi mig -cgi 2g.10gb -C

# Instance 2: Small instance for sentiment analysis
echo "Creating Instance 2: Sentiment Analysis (1g.5gb)..."
nvidia-smi mig -cgi 1g.5gb -C

# Verify the configuration
echo "Verifying MIG configuration..."
nvidia-smi mig -lgip
echo ""
nvidia-smi mig -lci
echo ""

# Show GPU memory allocation
echo "GPU Memory Allocation:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,units

# Set environment variables for the application
echo "Setting up environment variables..."
cat > /etc/environment.d/99-mig-audio-processor.conf << 'EOF'
CUDA_VISIBLE_DEVICES=0,1,2
CUDA_MIG_VISIBLE_DEVICES=0,1,2
AUDIO_PROCESSOR_MIG_ENABLED=1
AUDIO_PROCESSOR_TRANSCRIPTION_GPU=0
AUDIO_PROCESSOR_VAD_GPU=1  
AUDIO_PROCESSOR_SENTIMENT_GPU=2
EOF

echo ""
echo "=== MIG Configuration Complete ==="
echo "GPU Instances Created:"
echo "  - Instance 0 (MIG-0): Transcription Model (4g.20gb)"
echo "  - Instance 1 (MIG-1): VAD Pipeline (2g.10gb)" 
echo "  - Instance 2 (MIG-2): Sentiment Analysis (1g.5gb)"
echo ""
echo "Environment variables set in /etc/environment.d/99-mig-audio-processor.conf"
echo "Please restart your session or run: source /etc/environment.d/99-mig-audio-processor.conf"
echo ""
echo "To disable MIG later, run: sudo nvidia-smi -mig 0"
echo "Backup of previous state saved in: /tmp/mig_state_backup.txt"
