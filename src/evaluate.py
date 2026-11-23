"""STEP 4 — Benchmark the tuned model: eval perplexity + a W&B table of sample
generations, so 'did fine-tuning help?' has a number and receipts.

    python src/evaluate.py --model outputs/merged --n_samples 5
"""
import argparse
import json
import math
import os

import torch
import wandb
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], device: str, max_len: int = 1024) -> float:
    losses, total_tokens = [], 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).to(device)
        out = model(**enc, labels=enc["input_ids"])
        n = enc["input_ids"].numel()
        losses.append(out.loss.item() * n)
        total_tokens += n
    return math.exp(sum(losses) / max(total_tokens, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/merged")
    ap.add_argument("--eval_file", default="data/eval.jsonl")
    ap.add_argument("--n_ppl", type=int, default=50)
    ap.add_argument("--n_samples", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device).eval()

    rows = [json.loads(l) for l in open(args.eval_file, encoding="utf-8")]
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False)
             for r in rows[: args.n_ppl]]

    run = wandb.init(project=os.getenv("WANDB_PROJECT", "qlora-domain-llm"),
                     job_type="evaluation", name="eval-" + args.model.split("/")[-1])

    ppl = perplexity(model, tokenizer, texts, device)
    print(f"[eval] perplexity on {len(texts)} eval docs: {ppl:.2f}")
    run.log({"eval/perplexity": ppl})

    table = wandb.Table(columns=["prompt", "generation"])
    for r in rows[: args.n_samples]:
        prompt_msgs = r["messages"][:-1]  # drop gold answer
        inputs = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(device)
        out = model.generate(inputs, max_new_tokens=200, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        table.add_data(prompt_msgs[-1]["content"][:300], gen[:600])
        print("\n---\nPROMPT:", prompt_msgs[-1]["content"][:200],
              "\nMODEL :", gen[:300])

    run.log({"eval/sample_generations": table})
    run.finish()


if __name__ == "__main__":
    main()
