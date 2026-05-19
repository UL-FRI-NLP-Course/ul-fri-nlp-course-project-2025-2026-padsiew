# Natural language processing course: AIPravnik.si

Conversational AI assistant tailored to Slovenian law.

### CURRENT STRUCTURE
- all the code is in the "/d/hpc/projects/onj_fri/pad-siew/law-rag" folder




## Current TODO list

Done:

```text
✅ RAG vs no-RAG
✅ 2B vs 9B vs 12B
✅ dense-only vs reranking
✅ context_n sweep
✅ 5 prompt variants
✅ output cleanup
```

Still worth doing:

```text
1. Evaluate final P2 220/1200 output quality.
2. Try MAX_CONTEXT_CHARS/MAX_NEW_TOKENS small sweep if needed.
3. Query rewriting / expansion.
4. Embedding comparison: BGE-M3 vs multilingual-e5-large.
5. Generate article-derived question set from random chunks.
```

Optional:

```text
MMR retrieval
ColBERT
QwQ-32B qualitative demo
SloBERT pooled baseline
```




# WHAT WE DID SO FAR
## 1. Initial GaMS-2B setup on ARNES
- **What we tried:** Set up GaMS-2B-Instruct in the ARNES Slurm environment using the shared PyTorch Singularity/Apptainer container and a private Python venv.
- **Result:** GaMS-2B-Instruct runs successfully on V100S GPU.
- **Problem:** Model loading is slow, and generation sometimes continues with unwanted extra text unless cleaned.
## 2. BGE-M3 embeddings + FAISS baseline
- **What we tried:** Used `BAAI/bge-m3` to embed legal chunks and FAISS for similarity search.
- **Result:** Retrieval worked. Early tests retrieved sensible Slovenian legal chunks.
- **Problem:** Dense retrieval sometimes returns semantically similar but legally less appropriate chunks.
## 3. First limited PISRS corpus
- **What we tried:** Built a real-estate/legal subset from `register-predpisov.jsonl` using `--limit 2000`.
- **Result:** Created a working first RAG demo with around 23k chunks. RAG answered simple questions like služnost, komasacija, etažna lastnina.
- **Problem:** The corpus was incomplete. Some important sources, especially cadastral law, were missing or ranked poorly.
## 4. First 15-question evaluation set
- **What we tried:** Created 15 evaluation questions and manually labeled expected source/article for retrieval evaluation.
- **Result:** Could measure Top-1, Top-3, Top-5 retrieval accuracy.
- **Problem:** Initial questions were mostly simple definitional questions. Professor suggested harder/trickier questions.
## 5. Full corpus without `--limit 2000`
- **What we tried:** Removed the `--limit 2000` restriction and rebuilt chunks/index from the full filtered PISRS file.
- **Result:** Corpus increased from about 23k chunks to about 64k/79k chunks depending on chunker version. Kataster-related questions improved a lot.
- **Problem:** More corpus coverage also introduced more competing near-miss sources. Top-1 sometimes got worse even though Top-5 improved/stayed strong.
## 6. Updated chunking / cleanup
- **What we tried:** Improved chunk generation to remove trailing structural headings and added more cadastral keywords.
- **Result:** Generated a larger updated full chunk file with about 78,904 chunks.
- **Problem:** Need to keep comparing carefully because changing chunking and filtering changes both coverage and ranking. Some answer formatting issues remain unrelated to chunking.
## 7. Dense retrieval on full 15 labeled questions
- **What we tried:** Ran RAG with full updated index, no reranker, all 15 questions labeled.
- **Result:**
Top-1: 10/15 = 66.67%
Top-3: 13/15 = 86.67%
Top-5: 15/15 = 100.00%
- **Problem:** Pure reranking made results worse. It promoted semantically related but legally less appropriate chunks, for example wrong cadastral/land-register chunks or wrong legal articles.
## 8. Pure BGE reranker experiment
- **What we tried:** Added `BAAI/bge-reranker-v2-m3` after FAISS: FAISS top-20 → reranker → top chunks to GaMS.
- **Result:**
  - Top-1: 8/15 = 53.33%
  - Top-3: 12/15 = 80.00%
  - Top-5: 14/15 = 93.33%
- **Problem:** Pure reranking made results worse. It promoted semantically related but legally less appropriate chunks, for example wrong cadastral/land-register chunks or wrong legal articles.
## 9. GaMS-9B experiment
- **What we tried:** Downloaded and tested GaMS-9B-Instruct.
- **Result:** 9B runs, but outputs were not clearly better than 2B.
- **Problem:** 9B was more verbose and more likely to generate extra sections like `References`, `###`, or contradictory fallback text. We decided 2B is the stable main model for now.
## 10. Prompt and answer cleanup
- **What we tried:** Made prompts shorter and more extractive; added cleanup for generated continuation artifacts.
- **Result:** Some outputs became shorter and less contaminated.
- **Problem:** Prompting alone does not fix retrieval errors. Some answers still get cut off or add unwanted `VIRI:`/`Sklic` sections.
## 11. Manual expected labels
- **What we tried:** Added expected source/article labels for all 15 questions.
- **Result:** Now retrieval metrics are meaningful over all 15 questions.
- **Problem:** Some questions are ambiguous or terminology-sensitive, e.g. `zemljiški kataster` vs modern `kataster nepremičnin`.
## 12. Current best system
- **Best current pipeline:**
  - Full updated PISRS real-estate corpus
  - BGE-M3 embeddings
  - FAISS dense retrieval
  - GaMS-2B-Instruct
  - No pure reranker
- **Best retrieval result so far:**
  - Top-1: 66.67%
  - Top-3: 86.67%
  - Top-5: 100.00%
- **Main weakness:** Correct source is often in top 5 but not rank 1. Answer generation also needs better cleanup and manual quality evaluation.
## Main lesson so far
- RAG works and corpus coverage matters.
- Dense retrieval has good recall.
- Naive reranking did not help.
- The next step should be structured evaluation: RAG vs no-RAG, model size, retrieval parameters, and harder questions.
## 13. Current dense retrieval baseline
- **What we tried:** Disabled the reranker and used only BGE-M3 + FAISS dense retrieval on the 30-question set.
- **Result:** Dense retrieval achieved Top-1: 20/30 = 66.67%, Top-3: 26/30 = 86.67%, Top-5: 29/30 = 96.67%.
- **Problem:** The correct source is usually in top-5, but not always first. Retrieval recall is good, but ranking/source selection still needs improvement.
## 14. Reranker comparison
- **What we tried:** Compared dense-only retrieval with BGE reranking.
- **Result:** On the 30-question set, reranking did not improve Top-1, but improved Top-3 in one run: Dense Top-3: 86.67%, Reranked Top-3: 93.33%.
- **Problem:** Reranking did not reliably put the legally primary source at rank 1. It sometimes promoted semantically similar but legally less appropriate chunks.
## 15. No-RAG vs RAG model comparison
- **What we tried:** Compared no-RAG and RAG versions of GaMS-2B, GaMS-9B, and GaMS3-12B.
- **Result:** RAG clearly improved source grounding and correctness. No-RAG models often gave plausible answers but hallucinated or misidentified legal sources.
- **Problem:** Bigger models alone did not solve grounding. Even 12B still needs retrieved legal context.
## 16. Model size comparison
- **What we tried:** Ran GaMS-2B, GaMS-9B, and GaMS3-12B on the same 30 questions.
- **Result:** 12B generally produced cleaner language, 9B was also strong, and 2B was fastest/practical.
- **Problem:** Larger models still produced artifacts or unsupported claims if retrieval/context was imperfect.
## 17. LLM-assisted evaluation
- **What we tried:** Used retrieved PISRS chunks as evidence and evaluated model answers with a rubric: correctness, grounding, hallucination, format errors, and overall score.
- **Result:** This gave a practical way to compare 6 systems across 30 questions.
- **Problem:** Some questions had ambiguous or multiple valid sources, so automatic Top-k labels are not perfect. We flagged uncertain cases for manual review.
## 18. Question dataset issue
- **What we tried:** Created a 30-question dataset with easy, hard, true/false, comparison, scenario, and tricky negative questions.
- **Result:** The dataset is realistic and useful for evaluating user-style legal QA.
- **Problem:** Some expected source labels are debatable because legal questions can have multiple valid articles. This should be stated in the report.
## 19. Planned article-derived dataset
- **What we plan:** Sample random PISRS articles/chunks and generate questions directly from them.
- **Expected benefit:** The gold source is known because the question is generated from a specific article.
- **Purpose:** This will complement the current realistic dataset with a cleaner source-grounded evaluation set.
## 20. Context-size sweep
- **What we tried:** Ran GaMS3-12B with dense retrieval while changing `context_n`: 1, 2, 3, 5, 10.
- **Result:** Early qualitative comparison suggests `context_n=2` is a good balance.
- **Problem:** Too few chunks can miss evidence; too many chunks introduce noise and source contamination.
## 21. Output cleanup
- **What we tried:** Added stop markers and cleaned generated artifacts like `SKLIC`, `VIRI`, `<eos>`, `<end_of_turn>`.
- **Result:** Outputs are cleaner and easier to evaluate.
- **Problem:** Local LLMs still sometimes continue formatting after the answer, so cleanup remains necessary.
## 22. Next retrieval improvement: MMR
- **What we plan:** Try Maximum Marginal Relevance to select diverse but relevant chunks from top-10 dense retrieval.
- **Expected benefit:** Reduce repeated or noisy context and improve answer grounding.
- **Risk:** MMR may include diverse but less relevant chunks if the diversity weight is too high.
## 23. Upcoming experiments
- **Next planned experiments:** MMR retrieval, 5 prompt variants, query rewriting/query expansion, embedding comparison: BGE-M3 vs multilingual-e5-large, optional SloBERT pooled baseline.
- **Main goal:** Improve ranking/context selection and reduce hallucinations, not just try larger models.
