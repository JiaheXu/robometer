#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/robometer
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_TF=0 USE_FLAX=0
export WANDB_DISABLED=true TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
exec .venv/bin/python -m robometer.evals.eval_server \
  model_path=/home/ubuntu/robometer/models/Robometer-4B \
  num_gpus=1 max_workers=1 server_url=0.0.0.0 server_port=8000 \
  hydra.run.dir=/home/ubuntu/robometer/deployment/runtime
