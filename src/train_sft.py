from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import _torch_collate_batch


class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Custom collator that properly handles labels with variable lengths."""
    
    def torch_call(self, examples):
        # Separate labels from other features
        labels = [ex.pop("labels") for ex in examples]
        
        # Pad the other features (input_ids, attention_mask, etc.)
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        
        # Pad labels separately to match input_ids length
        labels_padded = []
        max_len = batch["input_ids"].shape[1]
        for label in labels:
            if len(label) < max_len:
                label = label + [-100] * (max_len - len(label))
            elif len(label) > max_len:
                label = label[:max_len]
            labels_padded.append(label)
        
        batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)
        return batch


@dataclass
class SFTConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir: Path = Path("checkpoints/sft_v3")
    train_path: Path = Path("data/sft_traces.jsonl")
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5  # Lower LR for more stable training
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 512
    save_steps: int = 200
    logging_steps: int = 20
    device: str = "auto"


def train(cfg: SFTConfig) -> None:
    dataset = load_dataset("json", data_files=str(cfg.train_path))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine execution device
    target_device = "cuda" if (cfg.device == "cuda" or (cfg.device in ("auto", "default") and torch.cuda.is_available())) else "cpu"
    use_fp16 = (target_device == "cuda")

    # Tokenize with proper label masking (no padding yet - collator will handle it)
    def tokenize_with_labels(examples):
        texts = [tokenizer.apply_chat_template(msgs, tokenize=False) for msgs in examples["messages"]]
        full_tokenized = tokenizer(texts, truncation=True, max_length=cfg.max_seq_length)
        
        prompt_texts = [
            tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            for msgs in examples["messages"]
        ]
        prompt_tokenized = tokenizer(prompt_texts, truncation=True, max_length=cfg.max_seq_length)
        
        labels = []
        for i in range(len(texts)):
            prompt_len = len(prompt_tokenized["input_ids"][i])
            full_input_ids = full_tokenized["input_ids"][i]
            full_len = len(full_input_ids)
            
            if prompt_len <= full_len:
                label = [-100] * prompt_len + full_input_ids[prompt_len:]
            else:
                label = [-100] * full_len
            
            if len(label) != full_len:
                if len(label) < full_len:
                    label.extend([-100] * (full_len - len(label)))
                else:
                    label = label[:full_len]
            
            labels.append(label)
        
        result = dict(full_tokenized)
        result["labels"] = labels
        return result
    
    tokenized_ds = dataset.map(
        tokenize_with_labels,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    dtype = torch.float16 if use_fp16 else torch.float32
    if target_device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
        )
        model.to("cpu")

    collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=use_fp16,
        max_grad_norm=1.0,
        logging_first_step=True,
        use_cpu=(target_device == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run SFT on tool-call traces for TinyLLaMA on Windows.")
    parser.add_argument("--train-path", type=Path, default=SFTConfig.train_path)
    parser.add_argument("--output-dir", type=Path, default=SFTConfig.output_dir)
    parser.add_argument("--model-name", type=str, default=SFTConfig.model_name)
    parser.add_argument("--epochs", type=int, default=SFTConfig.num_train_epochs)
    parser.add_argument("--batch", type=int, default=SFTConfig.per_device_train_batch_size)
    parser.add_argument("--grad-accum", type=int, default=SFTConfig.gradient_accumulation_steps)
    parser.add_argument("--lr", type=float, default=SFTConfig.learning_rate)
    parser.add_argument("--max-len", type=int, default=SFTConfig.max_seq_length)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware device (auto, cuda, cpu)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    dev = "cpu" if args.cpu else args.device

    cfg = SFTConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_path=args.train_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_len,
        device=dev,
    )
    train(cfg)


if __name__ == "__main__":
    cli()
