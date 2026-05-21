#!/bin/bash
#SBATCH --job-name=rag-eval-12b
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
echo "Python: $HOME/law-rag-gemma3/venv/bin/python"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

nvidia-smi

apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd $PROJECT \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag-gemma3/venv/bin/python -u scripts/rag_eval.py \
    --questions data/eval/questions_30.jsonl \
    --model cjvt/GaMS3-12B-Instruct \
    --max-new-tokens 160 \
    --run-name rag_eval_gams3_12b
