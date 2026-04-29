print("DEBUG 1: Python started", flush=True)

print("DEBUG 2: importing torch...", flush=True)
import torch
print("DEBUG 3: torch imported", flush=True)
print("torch version:", torch.__version__, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0), flush=True)

print("DEBUG 4: importing transformers...", flush=True)
import transformers
print("DEBUG 5: transformers imported", flush=True)
print("transformers version:", transformers.__version__, flush=True)

print("DEBUG DONE", flush=True)
