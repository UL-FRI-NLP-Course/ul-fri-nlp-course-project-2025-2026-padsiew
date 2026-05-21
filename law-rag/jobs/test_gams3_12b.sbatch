#!/bin/bash
#SBATCH --job-name=test-12b
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT=/d/hpc/projects/onj_fri/pad-siew/law-rag

cd $PROJECT
mkdir -p logs hf-cache

export HF_HOME=$PROJECT/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on host: $(hostname)"
echo "HF_HOME=$HF_HOME"
nvidia-smi

apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd $PROJECT \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag-gemma3/venv/bin/python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "cjvt/GaMS3-12B-Instruct"

print("CUDA:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
)

print("Model loaded.", flush=True)

prompt = """Odgovori kratko v slovenščini.

Vprašanje: Kaj pomeni služnost?
Odgovor:"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

answer = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
print("ANSWER:")
print(answer)
PY
