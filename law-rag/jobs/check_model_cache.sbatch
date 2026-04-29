#!/bin/bash
#SBATCH --job-name=check-gams-cache
#SBATCH --partition=gridlong
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd ~/law-rag
mkdir -p logs ~/hf-cache

export HF_HOME=$HOME/hf-cache
export TRANSFORMERS_CACHE=$HOME/hf-cache
export HF_DATASETS_CACHE=$HOME/hf-cache/datasets
export TOKENIZERS_PARALLELISM=false

apptainer exec \
  --bind $HOME:$HOME \
  --bind $PWD:$PWD \
  --pwd $PWD \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag/venv/bin/python scripts/check_model_cache.py
