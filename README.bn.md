# BanglaRAG-Hybrid

*[English version](README.md)*

বাংলা উইকিপিডিয়ার উপর ভিত্তি করে একটি retrieval-augmented প্রশ্নোত্তর সিস্টেম।
Dense ও sparse retrieval একসাথে চালিয়ে, cross-encoder দিয়ে পুনরায় সাজিয়ে,
স্থানীয়ভাবে চালানো একটি instruction-tuned LLM দিয়ে উত্তর তৈরি করে।

## পদ্ধতি

```
Knowledge_Base.txt
      │
      ▼
 পরিষ্কারকরণ ──► বাক্যভিত্তিক chunking ──► আবর্জনা ছাঁকনি
      │
      ├──► E5 embedding ──► FAISS         ┐
      │                                    ├──► RRF fusion ──► cross-encoder rerank ──► শীর্ষ ৪
      └──► BM25 (+ শিরোনাম boost)         ┘                                             │
                                                                                         ▼
                                                          prompt ──► Ollama LLM ──► পরিষ্কার উত্তর
```

| ধাপ | যা ব্যবহার করা হয়েছে |
|---|---|
| Embedding | `intfloat/multilingual-e5-base` |
| Dense index | FAISS (normalized, cosine) |
| Sparse index | BM25Okapi, বাংলা-সচেতন tokenizer |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Generator | `qwen2.5:7b`, ব্যর্থ হলে `llama3.1:8b` |

## সেটআপ

```bash
git clone https://github.com/<username>/BanglaRAG-Hybrid.git
cd BanglaRAG-Hybrid

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash setup_ollama.sh
```

ডেটা `data/` ফোল্ডারে রাখুন (এটি git-ignored):

```
data/
├── Knowledge_Base.txt    # বাংলা উইকিপিডিয়া ডাম্প
├── train.csv             # কলাম: index, question, answer
└── test.csv              # কলাম: index, question
```

`data/` অন্য জায়গায় থাকলে `config.yaml` এর `paths.data_dir` বদলে দিন।

## ব্যবহার

```bash
# ১. chunk ও FAISS ইনডেক্স তৈরি (একবারই)
python build_index.py --rebuild

# ২. train set এ token F1 মাপা
python run_inference.py --eval --limit 100

# ৩. submission.csv তৈরি
python run_inference.py
```

কোড থেকে সরাসরি:

```python
from banglarag.pipeline import RAGPipeline

pipeline = RAGPipeline()
print(pipeline.answer("বাংলাদেশের স্বাধীনতা যুদ্ধ কবে শুরু হয়?"))
```

## কনফিগারেশন

সব সেটিং [`config.yaml`](config.yaml) এ — chunk size, top-k, মডেলের নাম,
RRF ধ্রুবক, decoding অপশন। কোডের কোথাও পাথ হার্ডকোড করা নেই।

## ফাইল কাঠামো

```
config.yaml           সব সেটিং
build_index.py        chunk + FAISS ইনডেক্স তৈরি
run_inference.py      উত্তর তৈরি ও মূল্যায়ন
setup_ollama.sh       Ollama ইনস্টল ও মডেল নামানো
banglarag/
├── config.py         config.yaml লোডার
├── preprocess.py     পরিষ্কারকরণ, chunking, আবর্জনা ছাঁকনি
├── retrieval.py      BM25, FAISS, reranker, hybrid search
├── generate.py       prompt, Ollama ক্লায়েন্ট, উত্তর পরিষ্কার
└── pipeline.py       সব একসাথে জোড়া
notebook/             মূল exploratory notebook (রেফারেন্স)
```

## নকশাগত সিদ্ধান্ত

- **E5 এর asymmetric প্রিফিক্স।** index করার সময় `passage:`, খোঁজার সময় `query:`।
  পরিষ্কার লেখা `metadata["raw_text"]` এ আলাদা রাখা হয়, তাই BM25, reranker ও prompt
  কখনো ওই প্রিফিক্স দেখে না।
- **Reranker অবশ্যই multilingual।** ইংরেজি-only `bge-reranker-base` বাংলায় স্পষ্টভাবে খারাপ করে।
- **শিরোনাম boost আগে থেকে index করা।** না হলে প্রতি প্রশ্নে ৩৮ হাজার ডকুমেন্ট স্ক্যান করতে হতো।
- **উত্তর কড়াভাবে পরিষ্কার করা।** স্কোরিং token F1 ভিত্তিক, তাই markdown, reasoning trace
  বা বাড়তি বিরামচিহ্ন সরাসরি নম্বর কাটে।

## লাইসেন্স

MIT
