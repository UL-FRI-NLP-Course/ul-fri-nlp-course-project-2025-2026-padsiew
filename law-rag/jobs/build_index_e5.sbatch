#!/bin/bash
#SBATCH --job-name=build-e5
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag
cd $PROJECT
mkdir -p logs index hf-cache

export HF_HOME=$PROJECT/hf-cache
export TOKENIZERS_PARALLELISM=false

apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd $PROJECT \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag/venv/bin/python -u scripts/build_index.py \
    --input data/raw/real_estate_chunks_full.jsonl \
    --index-output index/real_estate_e5_large.faiss \
    --metadata-output index/real_estate_e5_large_metadata.pkl \
    --embed-model intfloat/multilingual-e5-large \
    --batch-size 4 \
    --max-embed-chars 2000 \
    --max-seq-length 512
