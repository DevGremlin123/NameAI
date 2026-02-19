"""Pydantic configuration loader for NameAI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class GenerationConfig(BaseModel):
    max_new_tokens: int = 32
    num_beams: int = 1
    do_sample: bool = True
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.92
    repetition_penalty: float = 1.2


class ModelConfig(BaseModel):
    base_model: str = "google/flan-t5-small"
    prompt_prefix: str = "Generate a creative brand name for: "
    generation: GenerationConfig = GenerationConfig()


class TrainingParams(BaseModel):
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    bf16: bool = True
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42


class DataConfig(BaseModel):
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/val.jsonl"
    max_input_length: int = 256
    max_target_length: int = 32
    prompt_prefix: str = "Generate a creative brand name for: "


class CheckpointConfig(BaseModel):
    output_dir: str = "checkpoints"
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True


class LoggingConfig(BaseModel):
    wandb_project: str = "nameai"
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 250


class TrainingConfig(BaseModel):
    base_model: str = "google/flan-t5-small"
    training: TrainingParams = TrainingParams()
    data: DataConfig = DataConfig()
    checkpointing: CheckpointConfig = CheckpointConfig()
    logging: LoggingConfig = LoggingConfig()


class FilterConfig(BaseModel):
    use_blocklist: bool = True
    blocklist_path: str = "data/curated/slop_blocklist.txt"
    min_pronounceability: float = 0.4
    min_phonaesthetic_score: float = 0.3
    min_uniqueness: float = 0.5
    diversity_threshold: float = 0.6


class InferenceModelConfig(BaseModel):
    checkpoint_path: str = "checkpoints/best"
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


def load_training_config(path: str | Path = "configs/training.yaml") -> TrainingConfig:
    return TrainingConfig(**load_yaml(path))


def load_inference_config(path: str | Path = "configs/inference.yaml") -> InferenceConfig:
    return InferenceConfig(**load_yaml(path))
