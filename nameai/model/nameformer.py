"""NameFormer: Flan-T5-Small fine-tuned for creative brand name generation.

Instead of training a transformer from scratch, we fine-tune Google's
Flan-T5-Small (77M params) which already understands English. We only
need to teach it the mapping: description → creative brand name.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


BASE_MODEL = "google/flan-t5-small"
PROMPT_PREFIX = "Generate a creative brand name for: "


class NameFormer:
    """Wrapper around fine-tuned Flan-T5 for name generation."""

    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str = BASE_MODEL, device: str = "auto") -> "NameFormer":
        """Load from a HuggingFace model ID or local checkpoint directory."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = T5Tokenizer.from_pretrained(path)
        model = T5ForConditionalGeneration.from_pretrained(path)
        return cls(model, tokenizer, device)

    def save(self, path: str | Path) -> None:
        """Save model and tokenizer to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def generate_names(
        self,
        description: str,
        num_return: int = 10,
        max_length: int = 32,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.92,
        num_beams: int = 1,
        do_sample: bool = True,
        repetition_penalty: float = 1.2,
    ) -> list[str]:
        """Generate brand name candidates for a business description."""
        prompt = PROMPT_PREFIX + description
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate more candidates than needed to allow filtering
        num_candidates = num_return * 4

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=num_candidates,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
            )

        names = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            if text and text not in names:
                names.append(text)

        return names

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()
