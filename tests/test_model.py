"""Smoke tests for NameFormer (custom from-scratch architecture)."""

import pytest
import torch


def test_model_builds():
    from nameai.model.nameformer import NameFormer
    model = NameFormer(enc_vocab_size=16000, dec_vocab_size=76)
    params = model.count_parameters()
    assert params["total"] > 80_000_000  # ~85M params
    assert params["encoder"] > 0
    assert params["decoder"] > 0


def test_forward_pass():
    from nameai.model.nameformer import NameFormer
    model = NameFormer(enc_vocab_size=1000, dec_vocab_size=76,
                       enc_n_layers=2, dec_n_layers=2,
                       enc_d_model=64, dec_d_model=48,
                       enc_d_ff=128, dec_d_ff=96,
                       enc_n_heads=4, dec_n_heads=4)
    src = torch.randint(1, 1000, (2, 16))
    tgt = torch.randint(1, 76, (2, 8))
    logits = model(src, tgt)
    assert logits.shape == (2, 8, 76)


def test_mlm_forward():
    from nameai.model.nameformer import NameFormer
    model = NameFormer(enc_vocab_size=1000, dec_vocab_size=76,
                       enc_n_layers=2, dec_n_layers=2,
                       enc_d_model=64, dec_d_model=48,
                       enc_d_ff=128, dec_d_ff=96,
                       enc_n_heads=4, dec_n_heads=4)
    tokens = torch.randint(1, 1000, (2, 16))
    logits = model.forward_mlm(tokens)
    assert logits.shape == (2, 16, 1000)


def test_char_tokenizer():
    from nameai.tokenizer.char_tokenizer import CharTokenizer
    tok = CharTokenizer()
    text = "Netflix"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_scoring():
    from nameai.scoring.phonaesthetics import phonaesthetic_score
    from nameai.scoring.pronounceability import pronounceability_score
    from nameai.scoring.uniqueness import uniqueness_score

    for name in ["Netflix", "Spotify", "Discord", "Stripe"]:
        assert phonaesthetic_score(name) > 0.3
        assert pronounceability_score(name) > 0.5
        assert uniqueness_score(name) > 0.3

    assert pronounceability_score("xzqpf") < 0.5


def test_filter():
    from nameai.inference.filter import load_blocklist, contains_blocked_word
    blocklist = load_blocklist("data/curated/slop_blocklist.txt")
    assert len(blocklist) > 0
    assert contains_blocked_word("TechForge", blocklist)
    assert not contains_blocked_word("Spotify", blocklist)
