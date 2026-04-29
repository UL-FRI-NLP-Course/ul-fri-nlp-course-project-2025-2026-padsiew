import os
import time

print("DEBUG 1: script started", flush=True)
print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)

print("DEBUG 2: importing torch...", flush=True)
import torch
print("DEBUG 3: torch imported", flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("DEBUG 4: importing transformers package...", flush=True)
import transformers
print("DEBUG 5: transformers package imported", flush=True)
print("transformers version:", transformers.__version__, flush=True)

print("DEBUG 6: resolving AutoTokenizer...", flush=True)
AutoTokenizer = transformers.AutoTokenizer
print("DEBUG 7: AutoTokenizer resolved", flush=True)

print("DEBUG 8: resolving AutoModelForCausalLM...", flush=True)
AutoModelForCausalLM = transformers.AutoModelForCausalLM
print("DEBUG 9: AutoModelForCausalLM resolved", flush=True)

model_id = "cjvt/GaMS-2B-Instruct"

print("DEBUG 10: loading tokenizer...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    local_files_only=True,
)
print(f"DEBUG 11: tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

print("DEBUG 12: loading model...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)
print(f"DEBUG 13: model loaded in {time.time() - t0:.1f}s", flush=True)

if torch.cuda.is_available():
    print("GPU memory allocated GB:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), flush=True)
    print("GPU memory reserved GB:", round(torch.cuda.memory_reserved(0) / 1024**3, 2), flush=True)

prompt = "Vprašanje: Kaj je RAG sistem?\nOdgovor:"

print("DEBUG 14: tokenizing...", flush=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("DEBUG 15: generating...", flush=True)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
    )

print("DEBUG 16: generated", flush=True)
print(tokenizer.decode(output[0], skip_special_tokens=True), flush=True)
print("DEBUG DONE", flush=True)
