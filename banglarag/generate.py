"""Prompt, Ollama কল, আর উত্তর পরিষ্কার করা।"""
from __future__ import annotations

import re
import time

import requests
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# ১. Prompt
# ==========================================================

SYSTEM_PROMPT = """তুমি একটি বাংলা প্রশ্নোত্তর AI system।

কঠোর নিয়ম:
1. শুধুমাত্র Context থেকে উত্তর দাও
2. Context-এ স্পষ্ট উত্তর না থাকলে → "জানা নেই"
3. নিজের জ্ঞান ব্যবহার করবে না
4. সর্বোচ্চ ১২ শব্দ
5. অসম্পূর্ণ বাক্য লিখবে না
6. প্রশ্ন repeat করবে না
7. অতিরিক্ত ব্যাখ্যা দেবে না
8. এক লাইনে উত্তর দাও
9. Context-এর exact শব্দ ব্যবহার করো
10. তারিখ হলে পুরো তারিখ, নাম হলে পুরো নাম

ভুল উদাহরণ:
প্রশ্ন: স্বাধীনতা যুদ্ধ কবে শুরু?
উত্তর: ৩ ডিসেম্বর ১৯৭১

সঠিক উদাহরণ:
উত্তর: ২৬ মার্চ ১৯৭১"""

HUMAN_PROMPT = """Context:
{context}

প্রশ্ন:
{question}

শুধু উত্তর:"""


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])


def format_context(docs: list[Document]) -> str:
    """নম্বর দেওয়া context ব্লক।

    page_content নয়, raw_text ব্যবহার করা হয় — কারণ page_content এ
    E5 এর "passage: " প্রিফিক্স থাকে, যেটা LLM এর চোখে পড়া উচিত নয়।
    """
    blocks = []
    for i, d in enumerate(docs, 1):
        text = d.metadata.get("raw_text", d.page_content).strip()
        blocks.append(f"[{i}] {text}")
    return "\n\n".join(blocks)


# ==========================================================
# ২. Ollama ক্লায়েন্ট (মডেল fallback সহ)
# ==========================================================

class OllamaClient:
    def __init__(self, cfg: dict):
        self.url = cfg["ollama_url"]
        self.model_chain = list(cfg["model_chain"])
        self.temperature = cfg["temperature"]
        self.num_predict = cfg["num_predict"]
        self.timeout = cfg["timeout_seconds"]
        self.max_retries = cfg["max_retries"]
        self.fallback_answer = cfg["fallback_answer"]
        self._idx = 0

    @property
    def active_model(self) -> str:
        return self.model_chain[self._idx]

    @staticmethod
    def _to_ollama_messages(messages) -> list[dict]:
        """LangChain মেসেজ -> Ollama role। system যেন system-ই থাকে।"""
        return [
            {"role": "system" if m.type == "system" else "user", "content": m.content}
            for m in messages
        ]

    def chat(self, messages) -> str:
        payload_messages = self._to_ollama_messages(messages)

        for _ in range(self.max_retries):
            try:
                res = requests.post(
                    self.url,
                    json={
                        "model": self.active_model,
                        "messages": payload_messages,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.num_predict,
                        },
                    },
                    timeout=self.timeout,
                )
                res.raise_for_status()
                return res.json()["message"]["content"]
            except Exception as exc:                      # noqa: BLE001
                print(f"সমস্যা ({self.active_model}): {exc}")
                self._idx = (self._idx + 1) % len(self.model_chain)
                print(f"পরের মডেলে যাচ্ছি: {self.active_model}")
                time.sleep(2)

        return self.fallback_answer


# ==========================================================
# ৩. উত্তর পরিষ্কার করা
# ==========================================================

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
MARKDOWN_RE = re.compile(r"[*_`#]")
ANSWER_PREFIX_RE = re.compile(r"^উত্তর\s*[:：]?\s*")


def clean_answer(text: str) -> str:
    """token F1 এ প্রতিটা বাড়তি শব্দ নম্বর কাটে, তাই কড়া পরিষ্কার।"""
    text = str(text).strip()
    text = THINK_RE.sub("", text)            # reasoning trace বাদ
    text = MARKDOWN_RE.sub("", text)         # markdown চিহ্ন বাদ
    text = ANSWER_PREFIX_RE.sub("", text).strip()
    text = text.split("\n")[0].strip()       # শুধু প্রথম লাইন
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(" ,;:-")
