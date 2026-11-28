"""Export the canonical JSON split manifest to Minh Khang's CSV interface."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def export_manifest(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source), Path(destination)
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    seen = set()
    for split in ("train", "val", "test"):
        for relative_path in payload.get(f"{split}_files", []):
            normalized = Path(relative_path).as_posix()
            parts = normalized.split("/")
            if len(parts) != 3 or parts[0] not in {"train", "test"} or parts[1] not in {"neg", "pos"}:
                raise ValueError(f"Unexpected review path in split manifest: {relative_path}")
            label = 0 if parts[1] == "neg" else 1
            if normalized in seen:
                raise ValueError(f"Review appears in more than one split: {normalized}")
            seen.add(normalized)
            rows.append((split, normalized, label))
    if not rows:
        raise ValueError(f"No train/val/test file lists found in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "path", "label"])
        writer.writerows(rows)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/split_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/khang_split.csv"))
    args = parser.parse_args()
    output = export_manifest(args.manifest, args.output)
    print(f"[manifest] Exported shared split manifest: {output}")


if __name__ == "__main__":
    main()
