"""STEP 1 — Build train/eval JSONL files in chat-message format.

Demo source: databricks-dolly-15k (instruction data, permissive license).
To use YOUR domain data instead, write the same shape and skip this script:
    {"messages": [{"role":"user","content":...},{"role":"assistant","content":...}]}
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).resolve().parent
N_TRAIN, N_EVAL, SEED = 2000, 200, 42

SYSTEM = ("You are a precise, concise domain assistant fine-tuned for enterprise "
          "Q&A. Answer directly. If context is provided, ground the answer in it.")


def to_messages(row: dict) -> dict:
    user = row["instruction"]
    if row.get("context"):
        user = f"Context:\n{row['context']}\n\nQuestion: {row['instruction']}"
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response"]},
    ]}


def main() -> None:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [to_messages(r) for r in ds]
    random.Random(SEED).shuffle(rows)
    splits = {"train.jsonl": rows[:N_TRAIN],
              "eval.jsonl": rows[N_TRAIN:N_TRAIN + N_EVAL]}
    for name, subset in splits.items():
        path = OUT / name
        with path.open("w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path} ({len(subset)} examples)")


if __name__ == "__main__":
    main()
