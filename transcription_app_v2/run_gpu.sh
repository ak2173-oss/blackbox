#!/bin/bash
# GPU-OPTIMIZED RUN SCRIPT - RTX 5070 Ti + Ryzen AI 9 HX 370

# Set cuDNN library path
export LD_LIBRARY_PATH=$PWD/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Activate venv
source venv/bin/activate

# Run with GPU
python pipeline.py "$@"
