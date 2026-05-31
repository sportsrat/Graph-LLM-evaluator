from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Ensure HF cache uses project workspace directory if HF_HOME is not explicitly set
if "HF_HOME" not in os.environ:
    _project_root = Path(__file__).resolve().parent.parent.parent
    _cache_dir = _project_root / ".cache" / "huggingface"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(_cache_dir)


def get_default_device() -> str:
    """Return 'cuda' if NVIDIA GPU is available, else 'cpu' for Windows."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_default_dtype(device: Optional[str] = None) -> torch.dtype:
    """Return appropriate torch dtype for device: float16 for CUDA, float32 for CPU."""
    dev = (device or get_default_device()).lower()
    if "cuda" in dev:
        return torch.float16
    return torch.float32


@dataclass
class LLMConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    max_new_tokens: int = 96
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.device is None or self.device.lower() in ("auto", "default"):
            self.device = get_default_device()
        if self.torch_dtype is None:
            self.torch_dtype = get_default_dtype(self.device)


class StubLLM:
    """Deterministic stub that returns template guidance or mock responses for testing."""

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig(model_name="stub")

    def generate_plan(self, task: str) -> str:
        return (
            f"[stub-llm] For task '{task}', think aloud, then call GTT. "
            "Prefer bfs for reachability, shortest_path for routing, diameter for spread."
        )

    def chat(self, prompt: str) -> str:
        return self.generate_plan(prompt)

    def chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Provide mock responses matching ReAct or raw LLM expectation for testing."""
        last_msg = messages[-1]["content"] if messages else ""
        
        # Check if previous message contains an observation
        obs_match = None
        for m in reversed(messages):
            if m.get("role") == "user" and "Observation:" in m.get("content", ""):
                obs_content = m["content"].replace("Observation:", "").strip()
                try:
                    obs_match = json.loads(obs_content)
                except Exception:
                    pass
                break

        if obs_match:
            if "path" in obs_match:
                return f"FINAL_ANSWER: {json.dumps({'path': obs_match.get('path'), 'length': obs_match.get('length', len(obs_match.get('path', []))-1)})}"
            elif "diameter" in obs_match:
                return f"FINAL_ANSWER: {json.dumps({'diameter': obs_match.get('diameter')})}"
            elif "component_count" in obs_match:
                return f"FINAL_ANSWER: {json.dumps({'component_count': obs_match.get('component_count')})}"
            else:
                return f"FINAL_ANSWER: {json.dumps(obs_match)}"

        # Initial tool call generation based on user prompt
        prompt_lower = last_msg.lower()
        if "shortest path" in prompt_lower or "between node" in prompt_lower:
            import re
            m = re.search(r"node\s+(\d+)\s+and\s+node\s+(\d+)", last_msg, re.IGNORECASE)
            if m:
                src, dst = int(m.group(1)), int(m.group(2))
                return f"TOOL_CALL: {json.dumps({'action': 'shortest_path', 'args': {'source': src, 'target': dst}})}"
            return 'TOOL_CALL: {"action": "shortest_path", "args": {"source": 0, "target": 1}}'
        elif "diameter" in prompt_lower:
            return 'TOOL_CALL: {"action": "graph_diameter", "args": {}}'
        elif "component" in prompt_lower:
            return 'TOOL_CALL: {"action": "connected_components", "args": {}}'

        return self.generate_plan(last_msg)


class TransformersLLM:
    """Transformers-based backend with Windows (CUDA/CPU) compatibility."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore
        except ImportError as exc:
            raise RuntimeError("transformers not installed; install per requirements.txt") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        target_device = (self.config.device or get_default_device()).lower()
        if target_device in ("auto", "default"):
            target_device = get_default_device()
        self.config.device = target_device

        target_dtype = self.config.torch_dtype or get_default_dtype(target_device)
        self.config.torch_dtype = target_dtype

        # Model loading
        try:
            if target_device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=target_dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.model.to("cpu")
        except RuntimeError as e:
            if "buffer size" in str(e) or "memory" in str(e).lower() or "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                print(f"Warning: Model issue on {target_device}, falling back to CPU float32...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                self.model.to("cpu")
                self.config.device = "cpu"
                self.config.torch_dtype = torch.float32
            else:
                raise

        if self.config.device == "cuda":
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
            )
        else:
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1,
            )

    def chat(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )
        return out[0]["generated_text"]
    
    def chat_messages(self, messages: list) -> str:
        """Format chat messages using tokenizer's chat template and generate.
        
        For models without chat templates, formats as plain text.
        """
        has_chat_template = (
            hasattr(self.tokenizer, "chat_template") 
            and self.tokenizer.chat_template is not None
        )
        
        if has_chat_template:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                has_chat_template = False
        
        if not has_chat_template:
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")
            prompt_parts.append("Assistant: ")
            prompt = "".join(prompt_parts)
        
        if hasattr(self.model, "device"):
            device = self.model.device
        elif hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            device = next(iter(self.model.hf_device_map.values()))
        else:
            device = "cpu"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_ids = [
            tid for tid in generated_ids 
            if tid != self.tokenizer.pad_token_id and tid != self.tokenizer.eos_token_id
        ]
        if not generated_ids:
            return ""
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_tiny_llama(device: Optional[str] = None) -> TransformersLLM:
    """Helper to build a TinyLlama 1.1B chat backend."""
    cfg = LLMConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device=device or get_default_device(),
        max_new_tokens=196,
        temperature=0.2,
    )
    return TransformersLLM(cfg)


def build_from_checkpoint(path: str, device: Optional[str] = None) -> TransformersLLM:
    """Load a fine-tuned checkpoint (SFT or RL) as an LLM backend."""
    cfg = LLMConfig(
        model_name=path,
        device=device or get_default_device(),
        max_new_tokens=196,
        temperature=0.0,
    )
    return TransformersLLM(cfg)
