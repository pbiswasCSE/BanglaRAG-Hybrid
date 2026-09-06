"""উইকিপিডিয়া ডাম্প পরিষ্কার করা, chunk করা, আবর্জনা বাদ দেওয়া।

নোটবুকের clean_article / create_chunks / is_garbage_chunk এখানে একসাথে।
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

# ==========================================================
# ধ্রুবক — regex টেবিলগুলো এক জায়গায়, যাতে সহজে বদলানো যায়
# ==========================================================

CONTENT_MARKER = "উইকিপিডিয়া, মুক্ত বিশ্বকোষ থেকে"

FOOTER_MARKERS = [
    "তথ্যসূত্র\n", "বহিঃসংযোগ\n", "আরও দেখুন\n",
    "গোপনীয়তার নীতি", "ক্রিয়েটিভ কমন্স", "উইকিমিডিয়া ফাউন্ডেশন",
]

LINE_BLOCKLIST = ["বিষয়শ্রেণীসমূহ", "আলোচনা যোগ করুন", "ওয়েব্যাক মেশিনে"]

LINE_DROP_PATTERNS = [
    r"^↑",                                # রেফারেন্সের তীর চিহ্ন
    r"^[১২৩৪৫৬৭৮৯০\d]+\.?\d*$",           # শুধু সংখ্যার লাইন
    r"^https?://",                         # খালি URL
    r"^\[\s*সম্পাদনা\s*\]$",               # [সম্পাদনা] লিংক
    r"^আইএসবিএন\s+",
    r"^978[\d\-X]+$",
    r"^(ডিওআই|পিএমআইডি|doi|pmid)\s*[:।]",
    r"^[\d\-\s]+$",
]

TITLE_RE = re.compile(r"^(.+?)\s*[-–—]\s*উইকিপিডিয়া$")
ARTICLE_SPLIT_RE = re.compile(r"\n(?=\S[^\n]{2,78}\s*[-–—]\s*উইকিপিডিয়া\n)")
SENTENCE_SPLIT_RE = re.compile(r"(?<=।)\s*")

SCORECARD_RE = re.compile(r"\(\d+\s*ওভার\)")      # ক্রিকেট স্কোরকার্ড
LEAGUE_TABLE_RE = re.compile(r"জয়\s+ড্র\s+হার")   # ফুটবল টেবিল
INFOBOX_RE = re.compile(r"স্থাপিত\s+\d+")         # infobox সারি
NUMBER_RE = re.compile(r"\d+")
BANGLA_WORD_RE = re.compile(r"[\u0980-\u09FF]{3,}")


# ==========================================================
# ১. শিরোনাম বের করা
# ==========================================================

def extract_title(article_raw: str) -> str:
    """প্রথম লাইন থেকে ' - উইকিপিডিয়া' অংশ বাদ দিয়ে শিরোনাম।"""
    first_line = article_raw.split("\n")[0].strip()
    m = TITLE_RE.match(first_line)
    return m.group(1).strip() if m else first_line[:80]


# ==========================================================
# ২. আর্টিকেল পরিষ্কার করা
# ==========================================================

def _keep_line(stripped: str) -> bool:
    if any(re.match(p, stripped, re.IGNORECASE) for p in LINE_DROP_PATTERNS):
        return False
    if any(x in stripped for x in LINE_BLOCKLIST):
        return False
    if "আর্কাইভকৃত" in stripped and "তারিখে" in stripped:
        return False
    # "English Term – বাংলা" ধরনের শব্দকোষ সারি
    if re.match(r"^[A-Za-z][A-Za-z\s\d\-]+[-–]\s*[\u0980-\u09FF]", stripped):
        return False
    return True


def clean_article(text: str) -> str:
    """উইকির নেভিগেশন, রেফারেন্স, টেবিল সরিয়ে শুধু গদ্য রাখে।"""
    # ধাপ ১ — আসল লেখা শুরুর আগের সব বাদ
    idx = text.find(CONTENT_MARKER)
    if idx != -1:
        text = text[idx + len(CONTENT_MARKER):]

    # ধাপ ২ — সবচেয়ে আগের footer marker থেকে কেটে দেওয়া
    earliest = len(text)
    for marker in FOOTER_MARKERS:
        i = text.find(marker)
        if i != -1:
            earliest = min(earliest, i)
    text = text[:earliest]

    # ধাপ ৩ — লাইন ধরে ছাঁকা (পরপর ফাঁকা লাইন একটায় নামানো)
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if kept and kept[-1] != "":
                kept.append("")
            continue
        if _keep_line(stripped):
            kept.append(stripped)
    text = "\n".join(kept)

    # ধাপ ৪ — লাইনের ভেতরের আবর্জনা
    text = re.sub(r"\[\s*[\d০-৯]*\s*\]", "", text)          # [1] [১]
    text = re.sub(r"\[\s*সম্পাদনা\s*\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[\s*[^\]]*প্রয়োজন\s*\]", "", text)     # [তথ্যসূত্র প্রয়োজন]
    text = re.sub(r"ডিওআই\s*:.*", "", text)

    # ধাপ ৫ — হাইপারলিংকের কারণে ভাঙা বাক্য জোড়া লাগানো
    text = re.sub(r'(?<=[^\n।!?,])\n(?=[^\n\s(["])', " ", text)

    # ধাপ ৬ — ইউনিকোড ও স্পেস স্বাভাবিক করা
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" ।", "।", text)

    # ধাপ ৭ — প্রতি দাঁড়ির পরে অনুচ্ছেদ বিরতি
    text = re.sub(r"।\s*(?!\n)", "।\n\n", text)

    # ধাপ ৮ — শুধু বাক্যযুক্ত অনুচ্ছেদ রাখা
    good = [
        p.strip() for p in text.split("\n\n")
        if p.strip() and "।" in p and len(p.strip()) >= 20
    ]
    return "\n\n".join(good).strip()


# ==========================================================
# ৩. আবর্জনা chunk ছাঁকনি
# ==========================================================

def is_garbage_chunk(text: str, long_chunk_chars: int = 200) -> bool:
    """chunk টা গদ্য না হয়ে টেবিল/স্কোরকার্ড হলে True।"""
    if SCORECARD_RE.search(text):
        return True
    if LEAGUE_TABLE_RE.search(text):
        return True
    if INFOBOX_RE.search(text):
        return True

    # বাংলা শব্দের চেয়ে সংখ্যা বেশি মানে প্রায় নিশ্চিতভাবে টেবিল
    if len(NUMBER_RE.findall(text)) > len(BANGLA_WORD_RE.findall(text)):
        return True

    # লম্বা লেখা কিন্তু দুইটার কম বাক্য মানে তালিকা
    sentences = [s for s in text.split("।") if s.strip()]
    if len(sentences) < 2 and len(text) > long_chunk_chars:
        return True

    return False


# ==========================================================
# ৪. Chunk বানানো
# ==========================================================

def create_chunks(
    text: str,
    chunk_size: int = 800,
    overlap: int = 1,
    min_chunk_chars: int = 30,
) -> list[str]:
    """পূর্ণ বাক্য ভরে chunk_size পর্যন্ত ভরে, তারপর overlap বাক্য পিছিয়ে পরেরটা শুরু।"""
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if len(s.strip()) >= 10]

    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        current = ""
        j = i
        while j < len(sentences) and len(current) + len(sentences[j]) + 1 <= chunk_size:
            current += (" " if current else "") + sentences[j]
            j += 1
        if current.strip() and "।" in current and len(current) >= min_chunk_chars:
            chunks.append(current.strip())
        i = max(i + 1, j - overlap)   # সবসময় সামনে এগোয়, আটকায় না
    return chunks


# ==========================================================
# ৫. পুরো knowledge base -> chunk তালিকা
# ==========================================================

def build_chunks(kb_path: Path, cfg: dict) -> list[dict]:
    """Knowledge_Base.txt পড়ে {"text", "metadata"} রেকর্ডের তালিকা ফেরত দেয়।"""
    with open(kb_path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    articles = ARTICLE_SPLIT_RE.split(raw)
    print(f"মোট আর্টিকেল পাওয়া গেছে : {len(articles)}")

    records: list[dict] = []
    skipped = 0

    for i, part in enumerate(articles):
        if len(part.strip()) < cfg["min_article_chars"]:
            skipped += 1
            continue

        title = extract_title(part)
        cleaned = clean_article(part)
        if len(cleaned) < cfg["min_cleaned_chars"]:
            skipped += 1
            continue

        for chunk in create_chunks(
            cleaned,
            chunk_size=cfg["chunk_size"],
            overlap=cfg["overlap_sentences"],
            min_chunk_chars=cfg["min_chunk_chars"],
        ):
            if is_garbage_chunk(chunk):
                continue
            records.append({
                "text": f"শিরোনাম: {title}\n\n{chunk}",
                "metadata": {"title": title, "source": f"article_{i}"},
            })

    print(f"পরিষ্কার chunk        : {len(records)}")
    print(f"বাদ দেওয়া আর্টিকেল    : {skipped}")
    return records


def save_chunks(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
