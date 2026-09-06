# BanglaRAG-Hybrid

*[বাংলা সংস্করণ](README.bn.md)*

A retrieval-augmented question answering system over Bengali Wikipedia.
It runs dense and sparse retrieval together, fuses and reranks the results with a
multilingual cross-encoder, and generates answers with a locally served
instruction-tuned LLM.

## Approach

```
Knowledge_Base.txt
      │
      ▼
  cleaning ──► sentence-aware chunking ──► garbage filter
      │
      ├──► E5 embeddings ──► FAISS        ┐
      │                                    ├──► RRF fusion ──► cross-encoder rerank ──► top 4
      └──► BM25 (+ title boost)           ┘                                             │
                                                                                         ▼
                                                          prompt ──► Ollama LLM ──► cleaned answer
```

| Stage | Component |
|---|---|
| Embedding | `intfloat/multilingual-e5-base` |
| Dense index | FAISS (normalized, cosine) |
| Sparse index | BM25Okapi with a Bengali-aware tokenizer |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Generator | `qwen2.5:7b`, falling back to `llama3.1:8b` |

## Setup

```bash
git clone https://github.com/<username>/BanglaRAG-Hybrid.git
cd BanglaRAG-Hybrid

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash setup_ollama.sh
```

Place the dataset in `data/` (this directory is git-ignored):

```
data/
├── Knowledge_Base.txt    # Bengali Wikipedia dump
├── train.csv             # columns: index, question, answer
└── test.csv              # columns: index, question
```

If your data lives elsewhere, change `paths.data_dir` in `config.yaml`.

## Usage

```bash
# 1. Build chunks and the FAISS index (run once)
python build_index.py --rebuild

# 2. Measure token F1 on the training split
python run_inference.py --eval --limit 100

# 3. Produce submission.csv
python run_inference.py
```

From code:

```python
from banglarag.pipeline import RAGPipeline

pipeline = RAGPipeline()
print(pipeline.answer("বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়?"))
```

## Configuration

Every tunable lives in [`config.yaml`](config.yaml) — chunk size, top-k values,
model names, the RRF constant, decoding options. No path is hardcoded anywhere
in the source.

## Repository layout

```
config.yaml           all settings
build_index.py        builds chunks and the FAISS index
run_inference.py      generates answers and evaluates
setup_ollama.sh       installs Ollama and pulls the models
banglarag/
├── config.py         config.yaml loader
├── preprocess.py     cleaning, chunking, garbage filtering
├── retrieval.py      BM25, FAISS, reranker, hybrid search
├── generate.py       prompts, Ollama client, answer cleanup
└── pipeline.py       ties everything together
notebook/             original exploratory notebook, kept for reference
```

## Design decisions

- **E5 requires asymmetric prefixes.** `passage:` at index time, `query:` at search
  time. The clean text is stored separately in `metadata["raw_text"]`, so BM25, the
  reranker and the prompt never see the prefix.
- **The reranker must be multilingual.** The English-only `bge-reranker-base`
  degrades noticeably on Bengali.
- **Title boosting is precomputed.** Otherwise every query would scan the titles of
  all 38k chunks.
- **Answers are cleaned aggressively.** Scoring is token-F1 based, so markdown,
  reasoning traces and stray punctuation cost precision directly.

## License

MIT
