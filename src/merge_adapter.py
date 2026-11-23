"""STEP 3 — Merge the LoRA adapter into the base weights so vLLM can serve one
plain Hugging Face model directory (fastest inference path).

    python src/merge_adapter.py --adapter outputs/adapter --out outputs/merged
"""
import argparse
import os

import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--adapter", default="outputs/adapter")
    ap.add_argument("--out", default="outputs/merged")
    args = ap.parse_args()

    base = yaml.safe_load(open(args.config))["model"]["base_model"]
    print(f"[merge] base={base}  adapter={args.adapter}")

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="cpu",
        token=os.getenv("HF_TOKEN"))
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)

    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"✅ merged model -> {args.out}  (point vLLM here)")


if __name__ == "__main__":
    main()
