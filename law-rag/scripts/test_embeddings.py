from sentence_transformers import SentenceTransformer
import torch

print("CUDA available:", torch.cuda.is_available())

model_id = "BAAI/bge-m3"
model = SentenceTransformer(model_id)

texts = [
    "Pogodba o zaposlitvi se lahko odpove iz poslovnega razloga.",
    "Zastaralni rok za odškodninsko terjatev je določen z zakonom.",
    "To je nepovezano besedilo o vremenu.",
]

query = "Kdaj se lahko odpove pogodba o zaposlitvi?"

emb = model.encode(texts, normalize_embeddings=True)
q = model.encode([query], normalize_embeddings=True)

scores = (q @ emb.T)[0]

for score, text in sorted(zip(scores, texts), reverse=True):
    print(float(score), text)
