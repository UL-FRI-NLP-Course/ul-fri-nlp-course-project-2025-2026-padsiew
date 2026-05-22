# Natural language processing course: AIPravnik.si

Conversational AI assistant for Slovenian law.

## Running on ARNES

All the code which was used for experiments is available on the ARNES cluster at:

```text
/d/hpc/projects/onj_fri/pad-siew/law-rag
````

The GitHub repository contains the main scripts, evaluation sets, selected outputs, and report files. Large files such as models, data and full logs are not in this repo, but are available on arnes.

### 1. Log in to ARNES

```bash
ssh <username>@hpc-login4.arnes.si
```

### 2. Go to the project directory

```bash
cd /d/hpc/projects/onj_fri/pad-siew/law-rag
```

### 3. Start an interactive GPU session

```bash
srun --partition=gpu --gpus=1 --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

### 4. Set cache/environment variables

```bash
export HF_HOME=$PWD/hf-cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

If the model is not already cached, log in to HuggingFace:

```bash
hf auth login
```

Paste your HuggingFace token when prompted. The model will download automatically when first used.

### 5. Run the law RAG model on one demo question

Create a one-question file:

```bash
mkdir -p data/eval

cat > data/eval/demo_question.jsonl <<'EOF'
{"id":"demo001","question":"Kaj pomeni služnost?","type":"demo"}
EOF
```

Run the final dense setup:

```bash
apptainer exec --nv \
  --bind /d/hpc/projects/onj_fri:/d/hpc/projects/onj_fri \
  --bind $HOME:$HOME \
  --pwd /d/hpc/projects/onj_fri/pad-siew/law-rag \
  /d/hpc/singularity/pytorch-24.12-py3.sif \
  $HOME/law-rag-gemma3/venv/bin/python -u scripts/rag_eval_dense.py \
    --questions data/eval/demo_question.jsonl \
    --model cjvt/GaMS3-12B-Instruct \
    --context-n 2 \
    --prompt-id p2 \
    --max-context-chars 1200 \
    --max-new-tokens 220 \
    --run-name demo_question
```

The answer is saved to:

```text
data/outputs/
```

### 6. Run final evaluation with provided test questions

Dense RAG:

```bash
sbatch jobs/final_rag_12b_50q.sbatch
```

Hybrid BM25 + dense RAG:

```bash
sbatch jobs/final_rag_hybrid_12b_50q.sbatch
```

### Notes

GaMS3-12B can take several minutes to load on ARNES. This is expected because the model checkpoint is large and loaded from shared storage. For debugging, use GaMS-2B or a smaller question file.


The most important scripts are:

```text
code/rag/rag_eval_dense.py        Final dense RAG evaluation
code/rag/rag_eval_hybrid.py       Final hybrid BM25+dense RAG evaluation
code/rag/build_index.py           FAISS index construction
code/rag/build_bm25_index.py      BM25 index construction
code/rag/experiments/             Ablation and retrieval experiments
```