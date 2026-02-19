"""CLI interface: `nameai generate "A music streaming service"`"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(package_name="nameai")
def main() -> None:
    """NameAI — Generate creative brand names from descriptions."""
    pass


@main.command()
@click.argument("description")
@click.option("-n", "--num", default=10, help="Number of names to generate")
@click.option("--temperature", "-t", default=0.85, help="Sampling temperature")
@click.option("--top-k", default=40, help="Top-k sampling")
@click.option("--top-p", default=0.90, help="Nucleus sampling threshold")
@click.option("--checkpoint", default=None, help="Model checkpoint path")
@click.option("--inference-config", default="configs/inference.yaml", help="Inference config path")
def generate(
    description: str,
    num: int,
    temperature: float,
    top_k: int,
    top_p: float,
    checkpoint: str | None,
    inference_config: str,
) -> None:
    """Generate creative brand names for a business description."""
    from nameai.config import load_inference_config
    from nameai.inference.generator import NameGenerator

    console.print(f"\n[bold]Description:[/bold] {description}\n")

    with console.status("[bold green]Loading model..."):
        cfg = load_inference_config(inference_config)
        cfg.generation.temperature = temperature
        cfg.generation.top_k = top_k
        cfg.generation.top_p = top_p

        if checkpoint:
            cfg.model.checkpoint_path = checkpoint

        generator = NameGenerator.from_config(inference_config)

    with console.status("[bold green]Generating names..."):
        results = generator.generate(description, num_names=num)

    if not results:
        console.print("[red]No names passed quality filters. Try adjusting parameters.[/red]")
        return

    table = Table(title="Generated Names", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Name", style="bold cyan", min_width=15)
    table.add_column("Overall", justify="center", width=8)
    table.add_column("Sound", justify="center", width=8)
    table.add_column("Pronounce", justify="center", width=10)
    table.add_column("Unique", justify="center", width=8)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r["name"],
            f"{r['overall']:.2f}",
            f"{r['phonaesthetics']:.2f}",
            f"{r['pronounceability']:.2f}",
            f"{r['uniqueness']:.2f}",
        )

    console.print(table)


@main.command()
@click.argument("name")
def score(name: str) -> None:
    """Score a name on phonaesthetics, pronounceability, and uniqueness."""
    from nameai.scoring.phonaesthetics import phonaesthetic_score
    from nameai.scoring.pronounceability import pronounceability_score
    from nameai.scoring.uniqueness import uniqueness_score

    console.print(f"\n[bold]Scoring:[/bold] {name}\n")

    ph = phonaesthetic_score(name)
    pr = pronounceability_score(name)
    un = uniqueness_score(name)
    overall = (ph * pr * un) ** (1 / 3)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Bar", min_width=20)

    for label, val in [
        ("Phonaesthetics", ph),
        ("Pronounceability", pr),
        ("Uniqueness", un),
        ("Overall", overall),
    ]:
        bar = _score_bar(val)
        color = "green" if val >= 0.6 else "yellow" if val >= 0.4 else "red"
        table.add_row(label, f"[{color}]{val:.3f}[/{color}]", bar)

    console.print(table)


@main.command()
def info() -> None:
    """Show model architecture information."""
    from nameai.model.nameformer import NameFormer
    from nameai.tokenizer.bpe_tokenizer import BPETokenizer
    from nameai.tokenizer.char_tokenizer import CharTokenizer

    console.print("\n[bold]NameFormer[/bold] (custom encoder-decoder, trained from scratch)\n")

    char_tok = CharTokenizer()
    model = NameFormer(enc_vocab_size=16000, dec_vocab_size=char_tok.vocab_size)
    params = model.count_parameters()
    console.print(f"  Encoder params: {params['encoder']:>12,}")
    console.print(f"  Decoder params: {params['decoder']:>12,}")
    console.print(f"  Total params:   {params['total']:>12,}")
    console.print(f"\n  Encoder: 10 layers, d=512, BPE input (16K vocab)")
    console.print(f"  Decoder: 7 layers, d=384, character output ({char_tok.vocab_size} chars)")
    console.print(f"\n  Phase 1: Wikipedia MLM pre-training (learns English)")
    console.print(f"  Phase 2: Fine-tune on (description -> brand name) pairs\n")


def _score_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    empty = width - filled
    if value >= 0.6:
        color = "green"
    elif value >= 0.4:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{'#' * filled}[/{color}][dim]{'-' * empty}[/dim]"


if __name__ == "__main__":
    main()
