"""Name generation pipeline: encode description → autoregressive decode → filter → rank."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nameai.config import InferenceConfig, load_inference_config
from nameai.inference.filter import filter_and_rank, load_blocklist
from nameai.inference.sampler import apply_temperature, apply_top_k, apply_top_p
from nameai.model.nameformer import NameFormer
from nameai.tokenizer.bpe_tokenizer import BPETokenizer
from nameai.tokenizer.char_tokenizer import CharTokenizer


class NameGenerator:
    """End-to-end name generation from a business description."""

    def __init__(self, model: NameFormer, bpe_tok: BPETokenizer, char_tok: CharTokenizer,
                 config: InferenceConfig, device: torch.device) -> None:
        self.model = model
        self.bpe_tok = bpe_tok
        self.char_tok = char_tok
        self.config = config
        self.device = device
        self.model.eval()

        self.blocklist: set[str] = set()
        if config.filtering.use_blocklist:
            self.blocklist = load_blocklist(config.filtering.blocklist_path)

    @torch.no_grad()
    def _generate_candidates(self, description: str, num_return: int,
                             max_length: int, temperature: float,
                             top_k: int, top_p: float) -> list[str]:
        """Generate name candidates via autoregressive character-level decoding."""
        # Encode description with BPE
        src_ids = self.bpe_tok.encode(description, add_special=True)
        src_ids = self.bpe_tok.pad_sequence(src_ids, 512)
        src_ids = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        src_mask = (src_ids == 0)

        # Expand for batch generation
        src_ids = src_ids.expand(num_return, -1)
        src_mask = src_mask.expand(num_return, -1)

        # Encode once
        encoder_out = self.model.encode(src_ids, src_mask)

        # Start with BOS token
        bos_id = self.char_tok.special_tokens["<bos>"]
        eos_id = self.char_tok.special_tokens["<eos>"]
        tgt_ids = torch.full((num_return, 1), bos_id, dtype=torch.long, device=self.device)

        finished = torch.zeros(num_return, dtype=torch.bool, device=self.device)

        for _ in range(max_length):
            logits = self.model.decode_step(tgt_ids, encoder_out, src_mask)

            # Apply sampling
            logits = apply_temperature(logits, temperature)
            logits = apply_top_k(logits, top_k)
            logits = apply_top_p(logits, top_p)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Don't update finished sequences
            next_token = next_token.masked_fill(finished.unsqueeze(1), 0)
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            # Check for EOS
            finished = finished | (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        # Decode to strings
        names = []
        for i in range(num_return):
            ids = tgt_ids[i].tolist()
            name = self.char_tok.decode(ids)
            name = name.strip()
            if name and len(name) >= 2:
                names.append(name)

        return names

    def generate(self, description: str, num_names: int | None = None) -> list[dict]:
        """Generate and rank creative names for a business description."""
        gen = self.config.generation
        filt = self.config.filtering
        num_names = num_names or gen.num_names

        # Over-generate candidates
        num_candidates = gen.num_candidates
        raw_names = self._generate_candidates(
            description,
            num_return=num_candidates,
            max_length=gen.max_length,
            temperature=gen.temperature,
            top_k=gen.top_k,
            top_p=gen.top_p,
        )

        # Deduplicate
        seen = set()
        unique_names = []
        for n in raw_names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                unique_names.append(n)

        # Filter and rank
        results = filter_and_rank(
            unique_names,
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

        if config.model.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.model.device)

        bpe_tok = BPETokenizer(config.model.bpe_model)
        char_tok = CharTokenizer()

        model = NameFormer.load(
            config.model.checkpoint_path,
            device=str(device),
            enc_vocab_size=bpe_tok.vocab_size,
            dec_vocab_size=char_tok.vocab_size,
        )

        return cls(model, bpe_tok, char_tok, config, device)
