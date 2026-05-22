#!/bin/bash
#SBATCH --job-name=final-rag-12b-50q
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag
cd $PROJECT
mkdir -p logs data/outputs hf-cache

export HF_HOME=$PROJECT/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running FINAL RAG setup"
echo "Host: $(hostname)"
echo "Project: $PROJECT"
echo "Model: cjvt/GaMS3-12B-Instruct"
echo "Embedding: BAAI/bge-m3"
echo "Retrieval: dense-only"
echo "Prompt: p2"
echo "context_n=2"
echo "max_context_chars=1200"
echo "max_new_tokens=220"

nvidia-smi

apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd $PROJECT \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag-gemma3/venv/bin/python -u code/rag_eval_dense.py \
    --questions data/eval/questions_50_article_derived.jsonl \
    --model cjvt/GaMS3-12B-Instruct \
    --context-n 2 \
    --prompt-id p2 \
    --max-context-chars 1200 \
    --max-new-tokens 220 \
    --run-name final_rag_12b_50q_bge_dense_p2_ctx2_1200_220
