"""test.csv এর প্রশ্নের উত্তর দিয়ে submission.csv লেখে।
train.csv থাকলে --eval দিয়ে token F1 মাপা যায়।

    python run_inference.py --limit 20        # দ্রুত পরীক্ষা
    python run_inference.py                   # পুরো test set
    python run_inference.py --eval --limit 100
"""
import argparse
from collections import Counter

import pandas as pd
from tqdm import tqdm

from banglarag.config import data_path, load_config
from banglarag.pipeline import RAGPipeline
from banglarag.retrieval import tokenize


def token_f1(pred: str, gold: str) -> float:
    """শব্দ পর্যায়ে F1 — প্রতিযোগিতার স্কোরিং এর মতো।"""
    p, g = tokenize(pred), tokenize(gold)
    if not p or not g:
        return float(p == g)
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="প্রথম N টা প্রশ্ন মাত্র")
    parser.add_argument("--eval", action="store_true",
                        help="test নয়, train.csv এর উপর স্কোর মাপে")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    pipeline = RAGPipeline(cfg)

    # ---------- মূল্যায়ন মোড ----------
    if args.eval:
        df = pd.read_csv(data_path(cfg, "train"))
        if args.limit:
            df = df.head(args.limit)

        scores, exact = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="মূল্যায়ন"):
            pred = pipeline.answer(row["question"])
            gold = str(row["answer"])
            scores.append(token_f1(pred, gold))
            exact.append(float(tokenize(pred) == tokenize(gold)))

        print(f"নমুনা        : {len(scores)}")
        print(f"Token F1     : {sum(scores)/len(scores):.4f}")
        print(f"Exact match  : {sum(exact)/len(exact):.4f}")
        return

    # ---------- submission মোড ----------
    test_df = pd.read_csv(data_path(cfg, "test"))
    if args.limit:
        test_df = test_df.head(args.limit)

    answers = [pipeline.answer(q) for q in tqdm(test_df["question"], desc="উত্তর তৈরি")]

    out_path = data_path(cfg, "submission")
    pd.DataFrame({"index": test_df["index"], "answer": answers}).to_csv(
        out_path, index=False
    )
    print(f"{len(answers)} টি উত্তর লেখা হয়েছে: {out_path}")


if __name__ == "__main__":
    main()
