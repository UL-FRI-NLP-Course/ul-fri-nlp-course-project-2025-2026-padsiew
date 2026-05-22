from sentence_transformers import SentenceTransformer, util

models = [
    "intfloat/multilingual-e5-large",
    "rokn/slovlo-v1",
]

query = "Kaj pomeni služnost?"
docs = [
    "Služnost je pravica uporabljati tujo stvar ali zahtevati opustitev določenih ravnanj.",
    "Gradbeno dovoljenje se izda, če so izpolnjeni pogoji za gradnjo.",
    "Kataster nepremičnin vodi Geodetska uprava Republike Slovenije.",
]

for model_name in models:
    print("=" * 80)
    print("MODEL:", model_name)

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.models import Transformer, Pooling

        if model_name == "rokn/slovlo-v1":
            word_embedding_model = Transformer(
                model_name,
                tokenizer_args={"use_fast": False},
            )
            pooling_model = Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            model = SentenceTransformer(model_name)

        if "e5" in model_name.lower():
            q = "query: " + query
            d = ["passage: " + x for x in docs]
        else:
            q = query
            d = docs

        q_emb = model.encode([q], normalize_embeddings=True)
        d_emb = model.encode(d, normalize_embeddings=True)

        scores = util.cos_sim(q_emb, d_emb)[0]

        for score, doc in sorted(zip(scores.tolist(), docs), reverse=True):
            print(f"{score:.4f} | {doc}")

    except Exception as e:
        print("FAILED:", repr(e))