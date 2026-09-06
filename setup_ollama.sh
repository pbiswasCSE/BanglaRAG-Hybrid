#!/usr/bin/env bash
# Ollama ইনস্টল, সার্ভার চালু, আর মডেল নামানো।
# চালান: bash setup_ollama.sh
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama ইনস্টল হচ্ছে..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

# মডেল মেমোরিতে ধরে রাখে, তাই প্রতি প্রশ্নে দেরি কম হয়
export OLLAMA_KEEP_ALIVE="-1"

pkill ollama || true
nohup ollama serve > /tmp/ollama.log 2>&1 &
sleep 10

ollama pull qwen2.5:7b
ollama pull llama3.1:8b

echo "Ollama প্রস্তুত:"
curl -s http://127.0.0.1:11434/api/tags
