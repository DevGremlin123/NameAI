"""Build the training dataset for Phase 2 fine-tuning.

The custom NameFormer learns English from Wikipedia pre-training (Phase 1).
This script builds the (description → brand name) pairs for Phase 2.

Sources:
1. Our hand-curated iconic brands (~150+, highest quality)
2. HackerNoon startup dataset (real names + real descriptions)
3. Any additional data in data/raw/

Usage:
    python -m nameai.data.build_dataset
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def load_curated() -> list[dict]:
    """Load hand-curated iconic brand entries (gold standard)."""
    records = []
    curated_path = Path("data/curated/iconic_brands.jsonl")
    if not curated_path.exists():
        return records
    with open(curated_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                r["source"] = "curated"
                r.setdefault("quality", 1.0)
                records.append(r)
    print(f"  Curated brands: {len(records)}")
    return records


def download_startup_dataset() -> list[dict]:
    """Download startup names + descriptions from HuggingFace."""
    try:
        from datasets import load_dataset

        print("  Downloading HackerNoon startup dataset...")
        ds = load_dataset("HackerNoon/where-startups-trend", split="train")
        records = []
        for row in tqdm(ds, desc="  Processing startups", leave=False):
            name = (row.get("company") or "").strip()
            desc = (row.get("companyDescription") or "").strip()
            if not name or not desc or len(desc.split()) < 3:
                continue
            # Skip very long or generic descriptions
            if len(desc) > 300:
                desc = desc[:300].rsplit(" ", 1)[0]
            records.append({
                "name": name,
                "description": desc,
                "quality": 0.6,
                "source": "hackernoon",
            })
        print(f"  Startups loaded: {len(records)}")
        return records

    except Exception as e:
        print(f"  Startup dataset failed: {e}")
        return []


def load_raw_files() -> list[dict]:
    """Load any JSONL/CSV files from data/raw/."""
    records = []
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return records
    for f in raw_dir.glob("*.jsonl"):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("name") and r.get("description"):
                        r.setdefault("quality", 0.5)
                        r["source"] = f.stem
                        records.append(r)
                except json.JSONDecodeError:
                    continue
        print(f"  Raw {f.name}: {len(records)} total")
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate by name, keeping highest quality."""
    seen: dict[str, dict] = {}
    for r in records:
        key = r["name"].lower().strip()
        if key not in seen or r.get("quality", 0) > seen[key].get("quality", 0):
            seen[key] = r
    return list(seen.values())


def oversample_curated(records: list[dict], target_ratio: float = 0.15) -> list[dict]:
    """Oversample curated entries so they make up ~target_ratio of the dataset.

    This teaches the model that *these* are the kinds of names we want.
    """
    curated = [r for r in records if r.get("source") == "curated"]
    others = [r for r in records if r.get("source") != "curated"]

    if not curated or not others:
        return records

    # How many copies of curated do we need?
    target_curated = int(len(others) * target_ratio / (1 - target_ratio))
    copies_needed = max(1, target_curated // len(curated))
    oversampled = curated * copies_needed

    print(f"  Oversampled curated {copies_needed}x ({len(curated)} -> {len(oversampled)})")
    return others + oversampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training dataset")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true", help="Skip HuggingFace downloads")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...\n")
    all_records: list[dict] = []

    # 1. Curated (gold standard)
    all_records.extend(load_curated())

    # 2. Startups (real descriptions)
    if not args.skip_download:
        all_records.extend(download_startup_dataset())

    # 3. Any raw files
    all_records.extend(load_raw_files())

    # Deduplicate
    print(f"\n  Before dedup: {len(all_records):,}")
    all_records = deduplicate(all_records)
    print(f"  After dedup:  {len(all_records):,}")

    # Oversample curated
    all_records = oversample_curated(all_records)

    # Shuffle and split
    random.shuffle(all_records)
    split_idx = int(len(all_records) * (1 - args.val_ratio))
    train_records = all_records[:split_idx]
    val_records = all_records[split_idx:]

    print(f"\n  Train: {len(train_records):,}")
    print(f"  Val:   {len(val_records):,}")

    # Write
    for split_name, records in [("train", train_records), ("val", val_records)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Wrote {out_path}")

    # Stats
    print("\nSource breakdown:")
    sources: dict[str, int] = {}
    for r in all_records:
        src = r.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {count:,}")


if __name__ == "__main__":
    main()
