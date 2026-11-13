"""Lightweight integrity checks for the packaged research artifacts.

This script performs no optimization and writes no scientific results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_EDGES = ((1, 2), (1, 8), (2, 3), (2, 9), (3, 4), (3, 10),
                  (4, 5), (4, 11), (5, 6), (5, 12), (6, 7), (6, 13),
                  (7, 14), (8, 9), (9, 10), (10, 11), (11, 12),
                  (12, 13), (13, 14))
CLOCKWISE_NODES = (1, 2, 3, 4, 5, 6, 7, 14, 13, 12, 11, 10, 9, 8)
PAIRS = tuple((CLOCKWISE_NODES[i], CLOCKWISE_NODES[j])
              for i in range(14) for j in range(i + 1, 14))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-provenance", action="store_true",
                        help="also compare packaged hashes with provenance.csv")
    args = parser.parse_args()

    ident_path = ROOT / "data/identification/real_91pair_measurements.npz"
    valid_path = ROOT / "data/validation/V1_measurements.npz"
    with np.load(ident_path, allow_pickle=False) as data:
        require(data["Z_measured"].shape == (804, 91), "identification archive shape")
        require(np.isfinite(data["Z_measured"]).all(), "non-finite identification values")
        require(float(data["frequency_hz"]) == 100_000.0, "identification frequency")
        require(data["pair_ids"].tolist() == [f"{u}-{v}" for u, v in PAIRS],
                "identification pair order")
        # The historical fit deliberately selects the first 644 states.
        require(data["Z_measured"][:644].shape == (644, 91), "historical subset shape")

    with np.load(valid_path, allow_pickle=False) as data:
        require(data["Z_measured"].shape == (22, 10, 91), "validation archive shape")
        require(data["known_pressed_mask"].shape == (22, 19), "pressed-mask shape")
        require(np.isfinite(data["Z_measured"]).all(), "non-finite validation values")
        validation_pairs = [tuple(map(int, value.split("-"))) for value in data["pair_ids"]]
        require([frozenset(value) for value in validation_pairs] ==
                [frozenset(value) for value in PAIRS],
                "validation unordered-pair order")
        require(len(data["frequencies_hz"]) == 10, "validation frequency count")

    raw_ident = sorted((ROOT / "data/Load_label").glob("*_Measurement*.txt"))
    require(len(raw_ident) == 14, "raw historical acquisition-file count")
    ident_audit = json.loads(
        (ROOT / "data/identification/extraction_audit.json").read_text(encoding="utf-8"))
    require(ident_audit["n_samples"] == 804, "raw historical archive state count")
    ident_hashes = {row["file"]: row["sha256"] for row in ident_audit["files"]}
    require(all(sha256(path) == ident_hashes.get(path.name) for path in raw_ident),
            "raw historical file hash mismatch")

    raw_valid = sorted((ROOT / "data/Multifrequency").glob("*.txt"))
    require(len(raw_valid) == 22, "raw multifrequency sweep count")
    validation_manifest = pd.read_csv(ROOT / "data/validation/V1_manifest.csv")
    validation_hashes = dict(zip(validation_manifest["file_name"],
                                 validation_manifest["sha256"]))
    require(all(sha256(path) == validation_hashes.get(path.name) for path in raw_valid),
            "raw multifrequency file hash mismatch")

    for tag in ("05", "30", "70"):
        folder = ROOT / f"parameters/selected_weak_s2_prior/init_{tag}pct"
        parasitic = pd.read_csv(folder / "parasitic_parameters.csv")
        baseline = pd.read_csv(folder / "baseline_sensor_parameters.csv")
        require(len(parasitic) == 91, f"{tag}% parasitic parameter count")
        require(len(baseline) == 19, f"{tag}% baseline parameter count")
        require(np.isfinite(parasitic[["Gp_S", "Cp_F"]]).all().all(),
                f"{tag}% non-finite parasitic parameters")
        require(np.isfinite(baseline[["G_S", "B_S", "Z_abs_ohm"]]).all().all(),
                f"{tag}% non-finite baseline parameters")

    holdout = pd.read_csv(ROOT / "results/validation/E0_E2_five_frequency_holdout_detail.csv")
    selected = holdout[holdout["scenario"] == "E2_weak_S2_prior"]
    require(len(selected) == 630, "selected-model held-out evaluation count")
    require(set(selected["n_holdout"]) == {45, 46}, "complementary held-out sizes")

    no_para = pd.read_csv(ROOT / "results/no_parasitics/overall_metrics.csv")
    require(set(no_para["method"]) == {"E2_fixed_parasitics", "NoParasitics_recalibrated"},
            "no-parasitics method labels")

    if args.verify_provenance:
        manifest = ROOT / "docs/provenance.csv"
        require(manifest.exists(), "docs/provenance.csv is missing")
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                packaged = ROOT / row["repository_path"]
                require(packaged.exists(), f"missing packaged file: {packaged}")
                require(sha256(packaged) == row["sha256"], f"hash mismatch: {packaged}")

    print(json.dumps({
        "status": "PASS",
        "historical_states_used": 644,
        "historical_archive_states": 804,
        "historical_raw_files": len(raw_ident),
        "validation_sweeps": 22,
        "validation_raw_files": len(raw_valid),
        "frequencies_per_sweep": 10,
        "terminal_pairs": len(PAIRS),
        "sensing_elements": len(PHYSICAL_EDGES),
        "heldout_evaluations": len(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
