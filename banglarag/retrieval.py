"""Hybrid retrieval: FAISS (dense) + BM25 (sparse) -> RRF -> cross-encoder rerank।"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

TOKEN_RE = re.compile(r"[\w\u0980-\u09FF]+")


def tokenize(text: str) -> list[str]:
    """বাংলা ও ইংরেজি — দুই লিপিরই শব্দ ধরে।"""
    return TOKEN_RE.findall(text.lower())


# ==========================================================
# ১. Embedding ও Document
# ==========================================================

def load_embeddings(cfg: dict) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=cfg["model_name"],
        model_kwargs={"device": cfg["device"]},
        encode_kwargs={"normalize_embeddings": cfg["normalize"]},
    )


def to_documents(records: list[dict], cfg: dict) -> list[Document]:
    """E5 কে "passage: " প্রিফিক্স দিতে হয়।

    কিন্তু BM25, reranker আর prompt যেন ওই প্রিফিক্স না দেখে —
    তাই পরিষ্কার লেখা metadata["raw_text"] এ আলাদা রাখা হয়।
    """
    prefix = cfg["passage_prefix"]
    return [
        Document(
            page_content=prefix + r["text"],
            metadata={**r["metadata"], "raw_text": r["text"]},
        )
        for r in records
    ]


def build_or_load_faiss(docs, embeddings, index_path: Path, rebuild: bool = False) -> FAISS:
    if index_path.exists() and not rebuild:
        print(f"পুরনো FAISS ইনডেক্স লোড হচ্ছে: {index_path}")
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)

    print(f"FAISS ইনডেক্স বানানো হচ্ছে ({len(docs)} ডকুমেন্ট)...")
    store = FAISS.from_documents(docs, embeddings)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_path))
    print(f"সেভ হয়েছে: {index_path}")
    return store


# ==========================================================
# ২. BM25 (শিরোনাম boost সহ)
# ==========================================================

class BM25Index:
    """নোটবুকে প্রতি প্রশ্নে ৩৮ হাজার ডকুমেন্টের শিরোনাম স্ক্যান হতো (খুব ধীর)।

    এখানে শিরোনামের টোকেন একবারই index করা হয়, তাই boost প্রায় তাৎক্ষণিক।
    """

    def __init__(self, docs: list[Document], title_boost: float = 3.0):
        self.docs = docs
        self.title_boost = title_boost
        self.bm25 = BM25Okapi([tokenize(d.metadata["raw_text"]) for d in docs])

        self.title_postings: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(docs):
            for tok in set(tokenize(str(d.metadata.get("title", "")))):
                self.title_postings[tok].append(i)

    def search(self, query: str, top_k: int) -> list[Document]:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        for tok in tokens:
            for i in self.title_postings.get(tok, ()):
                scores[i] += self.title_boost

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.docs[i] for i in top_idx]


# ==========================================================
# ৩. Cross-encoder reranker
# ==========================================================

class Reranker:
    """bge-reranker-v2-m3 multilingual — ইংরেজি-only ভার্সন বাংলায় খারাপ করে।"""

    def __init__(self, cfg: dict):
        self.model = CrossEncoder(
            cfg["model_name"], device=cfg["device"], max_length=cfg["max_length"]
        )
        self.batch_size = cfg["batch_size"]

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        if not docs:
            return []
        pairs = [[query, d.metadata["raw_text"]] for d in docs]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:top_k]]


# ==========================================================
# ৪. Hybrid search
# ==========================================================

def _doc_key(doc: Document) -> str:
    """দুই retriever এ একই chunk চেনার জন্য স্থির পরিচয়।"""
    return f"{doc.metadata.get('title','')}|{doc.metadata.get('source','')}|{hash(doc.page_content)}"


class HybridRetriever:
    def __init__(self, vector_store, bm25: BM25Index, reranker: Reranker,
                 cfg: dict, query_prefix: str):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.cfg = cfg
        self.query_prefix = query_prefix

    def search(self, query: str) -> list[Document]:
        query = query.strip()

        # ধাপ ১ — dense (E5 এর "query: " প্রিফিক্স বাধ্যতামূলক)
        dense = self.vector_store.similarity_search(
            f"{self.query_prefix}{query}", k=self.cfg["faiss_top_k"]
        )

        # ধাপ ২ — sparse
        sparse = self.bm25.search(query, top_k=self.cfg["bm25_top_k"])

        # ধাপ ৩ — Reciprocal Rank Fusion দিয়ে দুটো মেলানো
        k = self.cfg["rrf_k"]
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for results in (dense, sparse):
            for rank, doc in enumerate(results):
                key = _doc_key(doc)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
                doc_map[key] = doc

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        candidates = [doc_map[key] for key, _ in merged[: self.cfg["rerank_top_k"]]]

        # ধাপ ৪ — cross-encoder দিয়ে চূড়ান্ত সাজানো
        return self.reranker.rerank(query, candidates, top_k=self.cfg["final_top_k"])
