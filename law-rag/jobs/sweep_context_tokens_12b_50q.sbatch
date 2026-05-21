#!/bin/bash
#SBATCH --job-name=sweep-ctx-tok-12b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

echo "Running on host: $(hostname)"
echo "Project: $PROJECT"
echo "HF_HOME=$HF_HOME"
nvidia-smi

run_config () {
  CTX=$1
  TOK=$2
  NAME=$3

  echo "================================================================"
  echo "Running config: $NAME"
  echo "max_context_chars=$CTX"
  echo "max_new_tokens=$TOK"

  apptainer exec --nv \
    --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
    --bind $HOME:$HOME \
    --pwd $PROJECT \
    /d/hpc/singularity/pytorch-24.12-py3.sif \
    $HOME/law-rag-gemma3/venv/bin/python -u scripts/rag_eval_dense.py \
      --questions data/eval/questions_50_article_derived.jsonl \
      --model cjvt/GaMS3-12B-Instruct \
      --context-n 2 \
      --prompt-id p2 \
      --max-context-chars "$CTX" \
      --max-new-tokens "$TOK" \
      --run-name "$NAME"
}

run_config 700 160 sweep_p2_ctx2_chars700_tok160
run_config 700 220 sweep_p2_ctx2_chars700_tok220
run_config 1200 160 sweep_p2_ctx2_chars1200_tok160
run_config 1200 220 sweep_p2_ctx2_chars1200_tok220

echo "Done."
