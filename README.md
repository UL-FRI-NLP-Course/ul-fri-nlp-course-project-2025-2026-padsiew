# Natural language processing course: AIPravnik.si

Conversational AI assistant tailored to Slovenian law.

### CURRENT STRUCTURE
- law-rag folder contains the code we are currently using on Arnes (scripts and jobs to run the models)
- old scripts are moved to /code folder (all the latest code is now in law-rag) 
- reports for submission 1 and 2 are in the /report folder 

### Current baseline:
- Dataset: real-estate subset from PISRS/COLESLAW
- Chunks: 23,592
- Embedder: BAAI/bge-m3
- Index: FAISS IndexFlatIP
- Generator: cjvt/GaMS-2B-Instruct
- Container: /d/hpc/singularity/pytorch-24.12-py3.sif
- User venv: ~/law-rag/venv
- Shared cache: /d/hpc/projects/onj_fri/pad-siew/law-rag/hf-cache
