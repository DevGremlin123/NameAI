"""Name generation pipeline: load model → generate → filter → rank."""

from __future__ import annotations

from nameai.config import InferenceConfig, load_inference_config
from nameai.inference.filter import filter_and_rank, load_blocklist
from nameai.model.nameformer import NameFormer


class NameGenerator:
    """End-to-end name generation from a business description."""

    def __init__(self, model: NameFormer, config: InferenceConfig) -> None:
        self.model = model
        self.config = config
        self.model.eval_mode()

        self.blocklist = set()
        if config.filtering.use_blocklist:
            self.blocklist = load_blocklist(config.filtering.blocklist_path)

    def generate(self, description: str, num_names: int | None = None) -> list[dict]:
        """Generate and rank creative names for a business description."""
        num_names = num_names or self.config.generation.max_new_tokens
        gen = self.config.generation
        filt = self.config.filtering

        # Generate raw candidates
        raw_names = self.model.generate_names(
            description,
            num_return=num_names * 4,
            max_length=gen.max_new_tokens,
            temperature=gen.temperature,
            top_k=gen.top_k,
            top_p=gen.top_p,
            repetition_penalty=gen.repetition_penalty,
            do_sample=gen.do_sample,
            num_beams=gen.num_beams,
        )

        # Filter and rank
        results = filter_and_rank(
            raw_names,
            blocklist=self.blocklist,
            min_pronounceability=filt.min_pronounceability,
            min_phonaesthetic=filt.min_phonaesthetic_score,
            min_uniqueness=filt.min_uniqueness,
            diversity_threshold=filt.diversity_threshold,
        )

        return results[:num_names]

    @classmethod
    def from_config(cls, inference_config_path: str = "configs/inference.yaml") -> "NameGenerator":
        """Load from config file."""
        config = load_inference_config(inference_config_path)
        model = NameFormer.from_pretrained(config.model.checkpoint_path, device=config.model.device)
        return cls(model, config)
