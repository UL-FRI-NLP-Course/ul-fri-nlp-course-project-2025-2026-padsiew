#!/bin/bash
#SBATCH --job-name=eval-embeds
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag
cd $PROJECT
mkdir -p logs data/outputs hf-cache

export HF_HOME=$PROJECT/hf-cache
export TOKENIZERS_PARALLELISM=false

PY=$HOME/law-rag/venv/bin/python
SIF=/d/hpc/singularity/pytorch-24.12-py3.sif

run_eval () {
  QUESTIONS=$1
  NAME=$2

  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    $SIF \
    $PY -u scripts/retrieval_eval.py \
      --questions $QUESTIONS \
      --embed-model BAAI/bge-m3 \
      --index index/real_estate_full.faiss \
      --metadata index/real_estate_full_metadata.pkl \
      --output data/outputs/retrieval_bge_${NAME}.jsonl

  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    $SIF \
    $PY -u scripts/retrieval_eval.py \
      --questions $QUESTIONS \
      --embed-model intfloat/multilingual-e5-large \
      --index index/real_estate_e5_large.faiss \
      --metadata index/real_estate_e5_large_metadata.pkl \
      --output data/outputs/retrieval_e5_${NAME}.jsonl

  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    $SIF \
    $PY -u scripts/retrieval_eval.py \
      --questions $QUESTIONS \
      --embed-model rokn/slovlo-v1 \
      --index index/real_estate_slovlo.faiss \
      --metadata index/real_estate_slovlo_metadata.pkl \
      --output data/outputs/retrieval_slovlo_${NAME}.jsonl
}

run_eval data/eval/questions_30.jsonl 30q
run_eval data/eval/questions_50_article_derived.jsonl 50q
