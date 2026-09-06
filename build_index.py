"""Knowledge base পরিষ্কার করে chunk ও FAISS ইনডেক্স বানায়। একবার চালালেই হয়।

    python build_index.py --rebuild
"""
import argparse

from banglarag.config import load_config
from banglarag.pipeline import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="config.yaml এর পাথ")
    parser.add_argument("--rebuild", action="store_true",
                        help="ক্যাশ করা chunk ও FAISS উপেক্ষা করে নতুন করে বানায়")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    RAGPipeline(cfg, rebuild=args.rebuild)
    print("ইনডেক্স তৈরি শেষ।")


if __name__ == "__main__":
    main()
