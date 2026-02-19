"""Data cleaning and processing pipeline.

Cleans raw data from various sources into training-ready JSONL format.

Usage:
    python -m nameai.data.clean
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


def normalize_text(text: str) -> str:
    """Normalize unicode, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_name(name: str) -> str | None:
    """Clean and validate a brand name. Returns None to filter out."""
    name = normalize_text(name)
    if not name:
        return None
    name = re.sub(r"[™®©]", "", name).strip()
    if len(name) < 2 or len(name) > 30:
        return None
    if name.isdigit():
        return None
    if not re.match(r"^[a-zA-Z0-9 &\-_.'+!]+$", name):
        return None
    return name


def clean_description(desc: str) -> str | None:
    """Clean and validate a description."""
    desc = normalize_text(desc)
    if not desc:
        return None
    if len(desc.split()) < 3:
        return None
    if len(desc) > 500:
        desc = desc[:500].rsplit(" ", 1)[0]
    return desc


def process_jsonl(input_path: Path) -> list[dict]:
    """Process a JSONL file into clean records."""
    results = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            name = record.get("name") or record.get("brand") or record.get("company")
            desc = record.get("description") or record.get("desc") or record.get("about")

            if not name or not desc:
                continue

            name = clean_name(str(name))
            desc = clean_description(str(desc))

            if name and desc:
                results.append({
                    "name": name,
                    "description": desc,
                    "quality": float(record.get("quality", 0.5)),
                })
    return results


def process_csv(input_path: Path) -> list[dict]:
    """Process a CSV file with name,description columns."""
    import csv

    results = []
    with open(input_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (
                row.get("name") or row.get("brand") or row.get("company")
                or row.get("Name") or row.get("Brand") or row.get("Company")
            )
            desc = (
                row.get("description") or row.get("desc") or row.get("about")
                or row.get("Description") or row.get("Desc") or row.get("About")
            )
            if not name or not desc:
                continue

            name = clean_name(str(name))
            desc = clean_description(str(desc))

            if name and desc:
                results.append({
                    "name": name,
                    "description": desc,
                    "quality": 0.5,
                })
    return results


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove duplicates, keeping highest quality."""
    seen: dict[str, dict] = {}
    for r in records:
        key = r["name"].lower()
        if key not in seen or r["quality"] > seen[key]["quality"]:
            seen[key] = r
    return list(seen.values())


def train_val_split(
    records: list[dict], val_ratio: float = 0.05, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Split into train/val."""
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    split_idx = int(len(records) * (1 - val_ratio))
    return records[:split_idx], records[split_idx:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and process training data")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    # Curated data
    curated_dir = Path("data/curated")
    if curated_dir.exists():
        for f in curated_dir.glob("*.jsonl"):
            records = process_jsonl(f)
            all_records.extend(records)
            print(f"  Curated {f.name}: {len(records)} records")

    # Raw data
    if input_dir.exists():
        for f in input_dir.glob("*.jsonl"):
            records = process_jsonl(f)
            all_records.extend(records)
            print(f"  Raw JSONL {f.name}: {len(records)} records")
        for f in input_dir.glob("*.csv"):
            records = process_csv(f)
            all_records.extend(records)
            print(f"  Raw CSV {f.name}: {len(records)} records")

    print(f"\nTotal before dedup: {len(all_records)}")
    all_records = deduplicate(all_records)
    print(f"Total after dedup: {len(all_records)}")

    train_records, val_records = train_val_split(all_records, args.val_ratio)
    print(f"Train: {len(train_records)}, Val: {len(val_records)}")

    for split_name, records in [("train", train_records), ("val", val_records)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
