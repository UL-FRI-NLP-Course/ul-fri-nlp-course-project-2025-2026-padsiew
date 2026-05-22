#!/bin/bash
#SBATCH --job-name=bm25-hybrid
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag
cd $PROJECT
mkdir -p logs data/outputs index hf-cache

export HF_HOME=$PROJECT/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

echo "Running on host: $(hostname)"
echo "Project: $PROJECT"

# Build BM25 index if missing
if [ ! -f index/real_estate_bm25.pkl ]; then
  echo "Building BM25 index..."
  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    /d/hpc/singularity/pytorch-24.12-py3.sif \
    $HOME/law-rag/venv/bin/python -u code/build_bm25_index.py \
      --input data/raw/real_estate_chunks_full.jsonl \
      --output index/real_estate_bm25.pkl
fi

run_eval () {
  MODE=$1
  ALPHA=$2
  QFILE=$3
  OUT=$4

  echo "================================================================"
  echo "MODE=$MODE ALPHA=$ALPHA QUESTIONS=$QFILE OUT=$OUT"

  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    /d/hpc/singularity/pytorch-24.12-py3.sif \
    $HOME/law-rag/venv/bin/python -u code/experiments/retrieval_eval_bm25_hybrid.py \
      --questions "$QFILE" \
      --bm25-index index/real_estate_bm25.pkl \
      --mode "$MODE" \
      --faiss-index index/real_estate_full.faiss \
      --faiss-metadata index/real_estate_full_metadata.pkl \
      --embed-model BAAI/bge-m3 \
      --candidate-k 50 \
      --top-k 10 \
      --alpha "$ALPHA" \
      --output "$OUT"
}

for Q in questions_30 questions_50_article_derived; do
  QFILE=data/eval/${Q}.jsonl

  run_eval bm25 0.0 "$QFILE" "data/outputs/${Q}_bm25.jsonl"
  run_eval hybrid 0.25 "$QFILE" "data/outputs/${Q}_hybrid_a025.jsonl"
  run_eval hybrid 0.50 "$QFILE" "data/outputs/${Q}_hybrid_a050.jsonl"
  run_eval hybrid 0.75 "$QFILE" "data/outputs/${Q}_hybrid_a075.jsonl"
done

echo "Done."
