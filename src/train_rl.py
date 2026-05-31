from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer


FINAL_PATTERN = re.compile(r"FINAL_ANSWER\s*(\{.*\})", re.DOTALL | re.IGNORECASE)


@dataclass
class RLConfig:
    sft_checkpoint: Path = Path("checkpoints/sft")
    tasks_path: Path = Path("data/tasks.jsonl")
    output_dir: Path = Path("checkpoints/rlvr")
    batch_size: int = 4
    ppo_epochs: int = 3
    learning_rate: float = 1e-5
    epsilon_tool: float = 0.01  # per tool-call penalty
    generation_max_new_tokens: int = 160
    device: str = "auto"


def _format_prompt(prompt: str) -> str:
    return (
        "You are a graph reasoning assistant. Use TOOL_CALL JSON to interact with the Graph Traversal Tool, "
        "then finish with FINAL_ANSWER JSON. Keep the trace concise.\n"
        f"Task: {prompt}\n"
        "Respond with TOOL_CALL and FINAL_ANSWER."
    )


def _extract_answer_json(text: str) -> Optional[dict]:
    match = FINAL_PATTERN.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def _count_tool_calls(text: str) -> int:
    return text.count("TOOL_CALL")


def _reward(sample: dict, completion: str, epsilon_tool: float) -> float:
    answer = _extract_answer_json(completion) or {}
    tool_calls = _count_tool_calls(completion)
    penalty = -epsilon_tool * tool_calls
    gt = sample.get("ground_truth", {})
    task = sample.get("task")

    acc = -0.5  # default small negative to push for valid answers
    if task == "shortest_path":
        true_len = gt.get("length")
        pred_len = answer.get("length") or (len(answer.get("path", [])) - 1 if answer.get("path") else None)
        if true_len is None:
            acc = 0.0
        elif pred_len is None:
            acc = -0.5
        elif pred_len == true_len:
            acc = 1.0
        else:
            acc = -abs(pred_len - true_len) * 0.1
    elif task == "diameter":
        true_d = gt.get("diameter")
        pred_d = answer.get("diameter")
        if pred_d is None:
            acc = -0.5
        elif pred_d == true_d:
            acc = 1.0
        else:
            acc = -abs(pred_d - true_d) * 0.1
    elif task == "components":
        true_c = gt.get("component_count")
        pred_c = answer.get("component_count")
        if pred_c is None:
            acc = -0.5
        elif pred_c == true_c:
            acc = 1.0
        else:
            acc = -abs(pred_c - true_c) * 0.1
    return acc + penalty


def train(cfg: RLConfig) -> None:
    target_device = "cuda" if (cfg.device == "cuda" or (cfg.device in ("auto", "default") and torch.cuda.is_available())) else "cpu"
    if target_device == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"

    ckpt_path = str(cfg.sft_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if target_device == "cuda" else torch.float32
    device_map = "auto" if target_device == "cuda" else None

    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ckpt_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if target_device == "cpu":
        policy_model.to("cpu")

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        ckpt_path,
        torch_dtype=dtype,
        device_map=device_map,
    )
    if target_device == "cpu":
        ref_model.to("cpu")

    data = load_dataset("json", data_files=str(cfg.tasks_path))["train"]

    ppo_config = PPOConfig(
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        ppo_epochs=cfg.ppo_epochs,
        mini_batch_size=1,
        gradient_accumulation_steps=1,
    )
    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=None,
    )

    device = trainer.accelerator.device
    
    # Collect batch_size samples before calling trainer.step()
    batch_queries = []
    batch_responses = []
    batch_rewards = []
    
    for sample in data:
        prompt = _format_prompt(sample["prompt"])
        query_toks = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        query_ids = query_toks["input_ids"][0].to(device)
        
        response_tensors = trainer.generate(
            [query_ids],
            max_new_tokens=cfg.generation_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)[0]
        reward = torch.tensor(_reward(sample, completion, cfg.epsilon_tool)).to(device)
        
        batch_queries.append(query_ids)
        batch_responses.append(response_tensors[0])
        batch_rewards.append(reward)
        
        # When batch is full, call trainer.step()
        if len(batch_queries) >= cfg.batch_size:
            trainer.step(batch_queries, batch_responses, batch_rewards)
            batch_queries = []
            batch_responses = []
            batch_rewards = []
    
    # Process remaining samples if batch_size doesn't divide evenly
    if batch_queries:
        trainer.step(batch_queries, batch_responses, batch_rewards)

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run RLVR (PPO) fine-tuning with efficiency penalty on Windows.")
    parser.add_argument("--sft-checkpoint", type=Path, default=RLConfig.sft_checkpoint)
    parser.add_argument("--tasks-path", type=Path, default=RLConfig.tasks_path)
    parser.add_argument("--output-dir", type=Path, default=RLConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=RLConfig.batch_size)
    parser.add_argument("--ppo-epochs", type=int, default=RLConfig.ppo_epochs)
    parser.add_argument("--lr", type=float, default=RLConfig.learning_rate)
    parser.add_argument("--epsilon-tool", type=float, default=RLConfig.epsilon_tool)
    parser.add_argument("--max-new", type=int, default=RLConfig.generation_max_new_tokens)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Hardware device (auto, cuda, cpu)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    dev = "cpu" if args.cpu else args.device

    cfg = RLConfig(
        sft_checkpoint=args.sft_checkpoint,
        tasks_path=args.tasks_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.lr,
        epsilon_tool=args.epsilon_tool,
        generation_max_new_tokens=args.max_new,
        device=dev,
    )
    train(cfg)


if __name__ == "__main__":
    cli()
