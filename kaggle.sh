#!/bin/bash

export CUDA_HOME=/usr/local/cuda
pip install wandb huggingface_hub python-dotenv
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install -e .

mkdir -p outputs
mkdir -p inputs/ckpt

python hf_down.py

mv ckpt/* inputs/ckpt/