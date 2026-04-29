#!/bin/bash
#SBATCH --job-name=build-faiss
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag

cd $PROJECT
mkdir -p logs index hf-cache

export HF_HOME=$PROJECT/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

echo "Running on host: $(hostname)"
echo "Project: $PROJECT"
echo "HF_HOME=$HF_HOME"

nvidia-smi

apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd $PROJECT \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag/venv/bin/python -u scripts/build_faiss_index.py \
    --input data/raw/real_estate_chunks.jsonl \
    --index-out index/real_estate.faiss \
    --metadata-out index/real_estate_metadata.pkl \
    --model BAAI/bge-m3 \
    --batch-size 8
