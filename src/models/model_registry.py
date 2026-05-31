"""
Model registry for multi-model evaluation.

Defines available models and their configurations for comparison on Windows (CUDA/CPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .llm import LLMConfig, StubLLM, TransformersLLM, get_default_device


@dataclass
class ModelSpec:
    """Specification for a model to evaluate."""
    name: str  # Display name
    model_id: str  # HuggingFace model ID or local path or 'stub'
    device: str = "auto"
    max_new_tokens: int = 196
    temperature: float = 0.2
    description: str = ""
    size: str = ""  # e.g., "0B", "1.1B", "7B"
    family: str = ""  # e.g., "Stub", "TinyLlama", "Qwen", "Mistral"


# Model Registry
AVAILABLE_MODELS: Dict[str, ModelSpec] = {
    # Instant testing stub
    "stub": ModelSpec(
        name="Stub-LLM (Instant)",
        model_id="stub",
        description="Instant mock LLM for testing workflows without downloads",
        size="0B",
        family="Stub",
    ),

    # TinyLlama (baseline)
    "tinyllama-1.1b": ModelSpec(
        name="TinyLlama-1.1B",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="Small efficient model, baseline for comparison",
        size="1.1B",
        family="TinyLlama",
    ),
    
    # Qwen Models (various sizes)
    "qwen-0.5b": ModelSpec(
        name="Qwen2-0.5B",
        model_id="Qwen/Qwen2-0.5B-Instruct",
        description="Qwen2 0.5B parameter model",
        size="0.5B",
        family="Qwen",
    ),
    "qwen-1.5b": ModelSpec(
        name="Qwen2-1.5B",
        model_id="Qwen/Qwen2-1.5B-Instruct",
        description="Qwen2 1.5B parameter model",
        size="1.5B",
        family="Qwen",
    ),
    "qwen-7b": ModelSpec(
        name="Qwen2-7B",
        model_id="Qwen/Qwen2-7B-Instruct",
        description="Qwen2 7B parameter model (if available)",
        size="7B",
        family="Qwen",
        max_new_tokens=256,
    ),
    
    # Mistral Models (instruction-tuned, good chat template support)
    "mistral-7b": ModelSpec(
        name="Mistral-7B-Instruct",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        description="Mistral 7B instruction-tuned model",
        size="7B",
        family="Mistral",
        max_new_tokens=256,
        temperature=0.2,
    ),
    "mistral-8x7b": ModelSpec(
        name="Mixtral-8x7B-Instruct",
        model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="Mixtral 8x7B MoE instruction-tuned model",
        size="47B",
        family="Mistral",
        max_new_tokens=256,
        temperature=0.2,
    ),
    
    # Other popular OSS models
    "phi-2": ModelSpec(
        name="Phi-2",
        model_id="microsoft/phi-2",
        description="Microsoft Phi-2 (2.7B parameters)",
        size="2.7B",
        family="Phi",
        max_new_tokens=256,
    ),
    "gemma-2b": ModelSpec(
        name="Gemma-2B",
        model_id="google/gemma-2b-it",
        description="Google Gemma 2B instruction-tuned",
        size="2B",
        family="Gemma",
        max_new_tokens=256,
    ),
}

# Global memory cache for instantiated models
_MODEL_CACHE: Dict[str, Union[TransformersLLM, StubLLM]] = {}


def get_model_spec(model_key: str) -> Optional[ModelSpec]:
    """Get model specification by key."""
    return AVAILABLE_MODELS.get(model_key)


def list_available_models() -> List[str]:
    """List all available model keys."""
    return list(AVAILABLE_MODELS.keys())


def build_model_from_spec(spec: ModelSpec) -> Union[TransformersLLM, StubLLM]:
    """Build a TransformersLLM or StubLLM from a ModelSpec."""
    if spec.model_id == "stub" or spec.family == "Stub":
        return StubLLM()
    
    device = spec.device
    if device in ("auto", "default", None):
        device = get_default_device()
        
    config = LLMConfig(
        model_name=spec.model_id,
        device=device,
        max_new_tokens=spec.max_new_tokens,
        temperature=spec.temperature,
    )
    return TransformersLLM(config)


def build_model(model_key: str) -> Union[TransformersLLM, StubLLM]:
    """Build a model by key."""
    spec = get_model_spec(model_key)
    if spec is None:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list_available_models()}")
    return build_model_from_spec(spec)


def get_cached_model(model_key: str) -> Union[TransformersLLM, StubLLM]:
    """Retrieve or instantiate a cached model instance."""
    if model_key not in _MODEL_CACHE:
        _MODEL_CACHE[model_key] = build_model(model_key)
    return _MODEL_CACHE[model_key]


def clear_model_cache() -> None:
    """Clear in-memory model cache."""
    _MODEL_CACHE.clear()


# Predefined model sets for common comparisons
MODEL_SETS = {
    "stub": ["stub"],
    "small": ["stub", "tinyllama-1.1b", "qwen-0.5b", "qwen-1.5b"],
    "medium": ["tinyllama-1.1b", "qwen-1.5b", "phi-2", "gemma-2b"],
    "all-small": ["tinyllama-1.1b", "qwen-0.5b", "qwen-1.5b"],
    "qwen-only": ["qwen-0.5b", "qwen-1.5b", "qwen-7b"],
    "mistral-only": ["mistral-7b"],
    "mistral-all": ["mistral-7b", "mistral-8x7b"],
    "all": list(AVAILABLE_MODELS.keys()),
}


def get_model_set(set_name: str) -> List[str]:
    """Get a predefined set of models."""
    if set_name in MODEL_SETS:
        return MODEL_SETS[set_name]
    # If not a predefined set, treat as comma-separated list
    return [m.strip() for m in set_name.split(",")]
