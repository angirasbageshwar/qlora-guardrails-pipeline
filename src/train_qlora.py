"""STEP 2 — Fine-tune an open-weight model with QLoRA (4-bit NF4 + LoRA adapters).

Single budget GPU (default):
    python src/train_qlora.py --config configs/training_config.yaml

Multi-GPU with DeepSpeed ZeRO-3 (shards params/grads/optimizer across GPUs):
    accelerate launch --num_processes 4 src/train_qlora.py \
        --config configs/training_config.yaml --deepspeed configs/deepspeed_zero3.json

⚠️ Compatibility note (honest, not hand-waved): 4-bit bitsandbytes quantization
and ZeRO-3 parameter sharding don't mix — ZeRO-3 needs full-precision shards it
can gather/partition. So `--deepspeed` automatically switches to bf16 LoRA
(still parameter-efficient); plain QLoRA remains the single-GPU path. This is
the standard production trade-off.
"""
import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

load_dotenv()


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataset(tokenizer, file: str, max_len: int):
    """JSONL chat messages -> single 'text' field via the model's chat template."""
    ds = load_dataset("json", data_files=file, split="train")

    def fmt(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)}

    return ds.map(fmt, remove_columns=ds.column_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/training_config.yaml")
    ap.add_argument("--deepspeed", default=None,
                    help="Path to ZeRO-3 json → multi-GPU bf16-LoRA mode")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    m, d, q, t = cfg["model"], cfg["data"], cfg["qlora"], cfg["training"]
    use_qlora = args.deepspeed is None

    tokenizer = AutoTokenizer.from_pretrained(m["base_model"], token=os.getenv("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "token": os.getenv("HF_TOKEN"),
        "torch_dtype": torch.bfloat16 if t["bf16"] else torch.float16,
        "attn_implementation": "sdpa",
    }
    if use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if t["bf16"] else torch.float16,
        )
        model_kwargs["device_map"] = "auto"

    print(f"[train] loading {m['base_model']} "
          f"({'4-bit QLoRA' if use_qlora else 'bf16 LoRA + DeepSpeed ZeRO-3'})")
    model = AutoModelForCausalLM.from_pretrained(m["base_model"], **model_kwargs)
    model.config.use_cache = False
    if use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=t["gradient_checkpointing"])

    lora = LoraConfig(
        r=q["r"], lora_alpha=q["lora_alpha"], lora_dropout=q["lora_dropout"],
        target_modules=q["target_modules"], bias="none", task_type="CAUSAL_LM",
    )

    train_ds = build_dataset(tokenizer, d["train_file"], d["max_seq_length"])
    eval_ds = build_dataset(tokenizer, d["eval_file"], d["max_seq_length"])
    print(f"[train] train={len(train_ds)}  eval={len(eval_ds)}")

    os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
    sft_args = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        logging_steps=t["logging_steps"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        max_seq_length=d["max_seq_length"],
        packing=t["packing"],
        dataset_text_field="text",
        report_to="wandb",                 # loss, LR, GPU memory → W&B dashboard
        run_name=cfg["wandb"]["run_name"],
        seed=t["seed"],
        deepspeed=args.deepspeed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora,
    )

    if torch.cuda.is_available():
        print(f"[train] GPU: {torch.cuda.get_device_name(0)} | "
              f"free before train: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"✅ adapter saved -> {Path(t['output_dir']).resolve()}")


if __name__ == "__main__":
    main()
