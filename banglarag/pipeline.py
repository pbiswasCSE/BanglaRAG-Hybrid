"""পুরো পাইপলাইন এক জায়গায় — একটা অবজেক্ট, একটা .answer() কল।"""
from __future__ import annotations

from .config import data_path, load_config
from .generate import OllamaClient, build_prompt, clean_answer, format_context
from .preprocess import build_chunks, load_chunks, save_chunks
from .retrieval import (
    BM25Index, HybridRetriever, Reranker, build_or_load_faiss,
    load_embeddings, to_documents,
)


class RAGPipeline:
    def __init__(self, cfg: dict | None = None, rebuild: bool = False):
        self.cfg = cfg or load_config()
        cfg = self.cfg

        # ধাপ ১ — chunk (একবার বানিয়ে ডিস্কে রাখা হয়)
        chunk_file = data_path(cfg, "chunks")
        if chunk_file.exists() and not rebuild:
            records = load_chunks(chunk_file)
            print(f"ক্যাশ থেকে {len(records)} chunk লোড হয়েছে")
        else:
            records = build_chunks(data_path(cfg, "knowledge_base"), cfg["chunking"])
            save_chunks(records, chunk_file)

        # ধাপ ২ — embedding ও document
        embeddings = load_embeddings(cfg["embedding"])
        docs = to_documents(records, cfg["embedding"])

        # ধাপ ৩ — retriever
        vector_store = build_or_load_faiss(
            docs, embeddings, data_path(cfg, "faiss_index"), rebuild=rebuild
        )
        bm25 = BM25Index(docs, title_boost=cfg["retrieval"]["title_boost"])
        reranker = Reranker(cfg["reranker"])
        self.retriever = HybridRetriever(
            vector_store, bm25, reranker,
            cfg=cfg["retrieval"],
            query_prefix=cfg["embedding"]["query_prefix"],
        )

        # ধাপ ৪ — generation
        self.prompt = build_prompt()
        self.llm = OllamaClient(cfg["generation"])
        print(f"পাইপলাইন প্রস্তুত (মডেল: {self.llm.active_model})")

    def answer(self, question: str) -> str:
        docs = self.retriever.search(question)
        messages = self.prompt.format_messages(
            context=format_context(docs), question=question
        )
        return clean_answer(self.llm.chat(messages))
