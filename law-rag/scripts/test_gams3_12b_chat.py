import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "cjvt/GaMS3-12B-Instruct"

print("CUDA:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

print("Tokenizer class:", tokenizer.__class__, flush=True)
print("bos_token:", tokenizer.bos_token, tokenizer.bos_token_id, flush=True)
print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id, flush=True)
print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id, flush=True)
print("chat_template exists:", tokenizer.chat_template is not None, flush=True)
print("chat_template:", tokenizer.chat_template[:500] if tokenizer.chat_template else None, flush=True)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
)
print("Model loaded.", flush=True)

messages = [
    {
        "role": "user",
        "content": "Odgovori kratko v slovenščini: Kaj pomeni služnost?"
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("PROMPT:")
print(repr(prompt[:1000]), flush=True)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print("input length:", inputs["input_ids"].shape[1], flush=True)
print("last input ids:", inputs["input_ids"][0, -20:].tolist(), flush=True)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

sequences = out.sequences
generated = sequences[0][inputs["input_ids"].shape[1]:]

print("generated ids:", generated.tolist(), flush=True)
print("generated length:", len(generated), flush=True)
print("ANSWER RAW:")
print(tokenizer.decode(generated, skip_special_tokens=False), flush=True)
print("ANSWER CLEAN:")
print(tokenizer.decode(generated, skip_special_tokens=True), flush=True)
