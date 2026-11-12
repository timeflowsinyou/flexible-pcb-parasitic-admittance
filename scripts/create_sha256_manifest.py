#!/usr/bin/env python3
"""Create or verify a non-mutating SHA-256 inventory of files under data/.

Usage:
  python scripts/create_sha256_manifest.py --write
  python scripts/create_sha256_manifest.py --verify results/audits/<manifest>.json

Never modifies data files. Requires only the Python standard library.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AUDIT = ROOT / "results" / "audits"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory() -> list[dict]:
    rows = []
    for path in sorted(p for p in DATA.rglob("*") if p.is_file() and p.name not in {".gitkeep", "README.md"}):
        rows.append({"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)})
    return rows


def write() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows = inventory()
    now = datetime.now(timezone.utc)
    destination = AUDIT / ("data_manifest_" + now.strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    content = {
        "schema_version": "1.0",
        "created_utc": now.isoformat(),
        "repository_relative_root": ".",
        "scope": "data/",
        "file_count": len(rows),
        "warning": "No experimental data is included until data sources have been reviewed and added.",
        "files": rows,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved audit: {destination.relative_to(ROOT)} (files: {len(rows)})")
    if len(rows) == 0:
        print("WARNING: No data files found. This is not a validated experimental-data release.")
    return 0


def verify(manifest_path: Path) -> int:
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior = {row["path"]: (row["size_bytes"], row["sha256"]) for row in obj["files"]}
    current = {row["path"]: (row["size_bytes"], row["sha256"]) for row in inventory()}
    missing = sorted(set(prior) - set(current))
    extra = sorted(set(current) - set(prior))
    changed = sorted(k for k in set(prior) & set(current) if prior[k] != current[k])
    print(f"Expected files: {len(prior)}; Current files: {len(current)}")
    for label, group in (("MISSING", missing), ("EXTRA", extra), ("CHANGED", changed)):
        for name in group:
            print(f"{label}: {name}")
    if missing or extra or changed:
        print("Verification FAILED")
        return 1
    print("Verification PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--write", action="store_true")
    commands.add_argument("--verify", type=Path, metavar="MANIFEST")
    args = parser.parse_args()
    if args.write:
        return write()
    return verify(args.verify.resolve())


if __name__ == "__main__":
    sys.exit(main())
