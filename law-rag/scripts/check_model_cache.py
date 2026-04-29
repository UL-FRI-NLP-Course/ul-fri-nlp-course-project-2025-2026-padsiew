from huggingface_hub import snapshot_download

model = "cjvt/GaMS-2B-Instruct"

path = snapshot_download(repo_id=model)
print(f"Model is available at: {path}")
