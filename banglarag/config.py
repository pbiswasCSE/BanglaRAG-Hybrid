"""config.yaml পড়ার একমাত্র জায়গা।

আপেক্ষিক পাথ repo root থেকে হিসাব হয়, তাই যেকোনো ফোল্ডার থেকে চালালেও কাজ করে।
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # data_dir আপেক্ষিক হলে repo root এর সাপেক্ষে ধরা হয়
    data_dir = Path(cfg["paths"]["data_dir"]).expanduser()
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    cfg["paths"]["data_dir"] = data_dir
    return cfg


def data_path(cfg: dict, key: str) -> Path:
    """config এর paths থেকে একটা পূর্ণ পাথ বানায়।"""
    return cfg["paths"]["data_dir"] / cfg["paths"][key]
