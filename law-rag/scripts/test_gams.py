import os
import time

print("START test_gams.py", flush=True)
print("HF_HOME:", os.environ.get("HF_HOME"), flush=True)

print("Importing torch...", flush=True)
import torch
print("Torch imported.", flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Importing transformers...", flush=True)
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Transformers classes imported.", flush=True)

model_id = "cjvt/GaMS-2B-Instruct"

print("Loading tokenizer...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
print(f"Tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

print("Loading model...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)
print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

if torch.cuda.is_available():
    print("GPU memory allocated GB:", round(torch.cuda.memory_allocated(0) / 1024**3, 2), flush=True)
    print("GPU memory reserved GB:", round(torch.cuda.memory_reserved(0) / 1024**3, 2), flush=True)

prompt = """Odgovori v slovenščini.

Vprašanje: Na kratko razloži, kaj je RAG sistem za iskanje po pravnih dokumentih.
Odgovor:"""

print("Generating...", flush=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=160,
        do_sample=False,
    )

print("ANSWER:", flush=True)
print(tokenizer.decode(output[0], skip_special_tokens=True), flush=True)
print("DONE", flush=True)
