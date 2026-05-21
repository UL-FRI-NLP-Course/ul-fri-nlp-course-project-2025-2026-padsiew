#!/bin/bash
#SBATCH --job-name=mmr-all-50q
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag
cd $PROJECT
mkdir -p logs data/outputs hf-cache

export HF_HOME=$PROJECT/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

echo "Running on host: $(hostname)"
echo "Project: $PROJECT"
echo "HF_HOME=$HF_HOME"

nvidia-smi

run_mmr () {
  NAME=$1
  MODEL=$2
  INDEX=$3
  META=$4
  OUT=$5

  echo "================================================================"
  echo "Running MMR: $NAME"
  echo "Model: $MODEL"
  echo "Index: $INDEX"
  echo "Output: $OUT"

  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    /d/hpc/singularity/pytorch-24.12-py3.sif \
    $HOME/law-rag/venv/bin/python -u scripts/retrieval_eval_mmr.py \
      --questions data/eval/questions_50_article_derived.jsonl \
      --embed-model "$MODEL" \
      --index "$INDEX" \
      --metadata "$META" \
      --mode mmr \
      --candidate-k 30 \
      --top-k 10 \
      --lambda-mult 0.7 \
      --output "$OUT"
}

run_mmr \
  "BGE-M3" \
  "BAAI/bge-m3" \
  "index/real_estate_full.faiss" \
  "index/real_estate_full_metadata.pkl" \
  "data/outputs/mmr_bge_50q.jsonl"

run_mmr \
  "E5-large" \
  "intfloat/multilingual-e5-large" \
  "index/real_estate_e5_large.faiss" \
  "index/real_estate_e5_large_metadata.pkl" \
  "data/outputs/mmr_e5_50q.jsonl"

run_mmr \
  "Slovlo-v1" \
  "rokn/slovlo-v1" \
  "index/real_estate_slovlo.faiss" \
  "index/real_estate_slovlo_metadata.pkl" \
  "data/outputs/mmr_slovlo_50q.jsonl"

echo "Done."
