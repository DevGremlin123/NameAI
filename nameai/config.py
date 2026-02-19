"""Pydantic configuration loader for NameAI."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class EncoderConfig(BaseModel):
    vocab_size: int = 16000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 10
    d_ff: int = 2816
    max_seq_len: int = 512


class DecoderConfig(BaseModel):
    vocab_size: int = 76
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 7
    d_ff: int = 1792
    max_seq_len: int = 64


class GenerationConfig(BaseModel):
    max_length: int = 32
    temperature: float = 0.85
    top_k: int = 40
    top_p: float = 0.90
    num_candidates: int = 40
    num_names: int = 10


class ModelConfig(BaseModel):
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    dropout: float = 0.1
    generation: GenerationConfig = GenerationConfig()


class FilterConfig(BaseModel):
    use_blocklist: bool = True
    blocklist_path: str = "data/curated/slop_blocklist.txt"
    min_pronounceability: float = 0.4
    min_phonaesthetic_score: float = 0.3
    min_uniqueness: float = 0.5
    diversity_threshold: float = 0.6


class InferenceModelConfig(BaseModel):
    checkpoint_path: str = "checkpoints/best.pt"
    bpe_model: str = "data/bpe_tokenizer.model"
    device: str = "auto"


class InferenceConfig(BaseModel):
    model: InferenceModelConfig = InferenceModelConfig()
    generation: GenerationConfig = GenerationConfig()
    filtering: FilterConfig = FilterConfig()


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_model_config(path: str | Path = "configs/model.yaml") -> ModelConfig:
    return ModelConfig(**load_yaml(path))


def load_inference_config(path: str | Path = "configs/inference.yaml") -> InferenceConfig:
    return InferenceConfig(**load_yaml(path))
