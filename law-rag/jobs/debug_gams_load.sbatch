#!/bin/bash
#SBATCH --job-name=debug-gams-load
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag

cd $PROJECT
mkdir -p logs

export HF_HOME=$PROJECT/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility

echo "Running on host: $(hostname)"
echo "Project: $PROJECT"
echo "HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"

nvidia-smi

echo "About to run apptainer..."
date

timeout 300 apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd $PROJECT \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag/venv/bin/python -u scripts/debug_gams_load.py

echo "Apptainer finished."
date
