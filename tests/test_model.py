"""Smoke tests for NameFormer (Flan-T5 based)."""

import pytest


def test_model_loads():
    from nameai.model.nameformer import NameFormer, BASE_MODEL
    model = NameFormer.from_pretrained(BASE_MODEL, device="cpu")
    params = model.count_parameters()
    assert params["total"] > 0
    assert params["total"] == params["trainable"]


def test_generate_names():
    from nameai.model.nameformer import NameFormer, BASE_MODEL
    model = NameFormer.from_pretrained(BASE_MODEL, device="cpu")
    names = model.generate_names(
        "A music streaming service",
        num_return=3,
        max_length=20,
    )
    assert len(names) > 0
    for name in names:
        assert isinstance(name, str)
        assert len(name) > 0


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
