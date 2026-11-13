"""Build docs/provenance.csv and verify copied files byte-for-byte.

The source root is configurable so the packaged repository remains relocatable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def mappings(source: Path, load_label_root: Path,
             multifrequency_root: Path) -> list[tuple[Path, Path, str]]:
    rows: list[tuple[Path, Path, str]] = []
    scientific = [
        "experiment_I5_checkpointed_convergence.py", "experiment_I5_S2_E1_E2_E3.py",
        "V1_extract_multifrequency_91pairs_v1_1.py", "V2_2_two_offset_hypotheses.py",
        "V2_4_fixed_parasitic_intrinsic_drift.py", "V2_6_validate_22_sweeps_fixed_parasitics.py",
        "V2_7_publication_validation_v1_1.py", "validate_E2_locality_multifrequency.py",
        "E2_vs_NoParasitics_control.py", "rebuild_Fig2_Fig3_Fig4_E2_windows_fixed_download.py",
        "E2_loaded_recovery_figures_phase_legend.py",
    ]
    rows += [(source / name, ROOT / "src/scientific" / name, "scientific source")
             for name in scientific]
    ident = source / "real91_extract/run_20261103T145251_758871Z"
    rows += [(ident / name, ROOT / "data/identification" / name, "historical measurement artifact")
             for name in ("real_91pair_measurements.npz", "sample_manifest.csv", "extraction_audit.json")]
    validation = source / "V1_multifrequency_audit/run_20261105T113619_277534Z"
    rows += [(validation / name, ROOT / "data/validation" / name, "independent validation artifact")
             for name in ("V1_measurements.npz", "V1_manifest.csv", "V1_extraction_audit.json")]
    rows += [(path, ROOT / "data/Load_label" / path.name,
              "raw historical measurement file")
             for path in sorted(load_label_root.glob("*_Measurement*.txt"))]
    rows += [(path, ROOT / "data/Multifrequency" / path.name,
              "raw independent validation sweep")
             for path in sorted(multifrequency_root.glob("*.txt"))]
    for tag in ("05", "30", "70"):
        original = source / f"experiment_I5_S2_E1_E2_E3/E2_weak_S2_prior/init_{tag}pct"
        packaged = ROOT / f"parameters/selected_weak_s2_prior/init_{tag}pct"
        rows += [(original / name, packaged / name, "selected fixed-parasitics parameter artifact")
                 for name in ("parasitic_parameters.csv", "baseline_sensor_parameters.csv", "fit_summary.json")]
    final = source / "Chapter4_E2_final_validation/run_20261108T173034_721361Z"
    locality = source / "E2_locality_and_multifrequency_validation/run_20261108T161407_229602Z"
    for name in ("E0_E2_five_frequency_holdout_detail.csv", "E0_E2_five_frequency_paired.csv"):
        rows.append((locality / name, ROOT / "results/validation" / name, "validation result"))
    for name in ("E2_verified_frequency_summary.csv", "E2_sensorwise_summary.csv",
                 "E2_S16_repeatability_summary.csv", "validation_report.json"):
        rows.append((final / name, ROOT / "results/validation" / name, "validation result"))
    control = source / "E2_vs_NoParasitics_control/run_20261110T163257_799182Z"
    for name in ("overall_metrics.csv", "metrics_by_frequency.csv", "metrics_by_condition.csv",
                 "paired_e2_nopara_holdout.csv", "run_audit.json"):
        rows.append((control / name, ROOT / "results/no_parasitics" / name, "no-parasitics result"))
    fig234 = source / "Chapter4_E2_Figures_Updated/run_20261109T135226_686207Z"
    for name in ("Fig3_frequency_source.csv", "Fig4_sensorwise_source.csv", "figure_build_audit.json"):
        rows.append((fig234 / name, ROOT / "results/figure_sources" / name, "figure source"))
    for stem in ("Fig2_terminal_pair_compensation", "Fig3_frequency_validation", "Fig4_sensorwise_holdout"):
        for suffix in ("png", "pdf", "svg"):
            name = f"{stem}.{suffix}"
            rows.append((fig234 / name, ROOT / "figures/publication" / name, "publication figure"))
    fig56 = source / "E2_loaded_recovery_figures_corrected/run_20261110T151759_451311Z"
    sources56 = ("Fig_loaded_recovery_sensor6_100000Hz_init30pct_fold0_91pair_data.csv",
                 "Fig_loaded_recovery_sensor2_16_100000Hz_init30pct_fold0_91pair_data.csv",
                 "case_summary.csv", "run_audit.json")
    rows += [(fig56 / name, ROOT / "results/figure_sources" / name, "figure source") for name in sources56]
    for stem in ("Fig_loaded_recovery_sensor6_100000Hz_init30pct_fold0",
                 "Fig_loaded_recovery_sensor2_16_100000Hz_init30pct_fold0"):
        for suffix in ("png", "pdf"):
            name = f"{stem}.{suffix}"
            rows.append((fig56 / name, ROOT / "figures/publication" / name, "publication figure"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=Path(r"D:\IEEE_TIM_related\Simulation_Experiment"))
    parser.add_argument("--load-label-root", type=Path,
                        default=Path(r"D:\IEEE_TIM_related\Data\Load_label"))
    parser.add_argument("--multifrequency-root", type=Path,
                        default=Path(r"D:\IEEE_TIM_related\Data\MultiFrequency_Measurements\2026_11_05"))
    args = parser.parse_args()
    output = ROOT / "docs/provenance.csv"
    records = []
    for original, packaged, role in mappings(
            args.source_root, args.load_label_root, args.multifrequency_root):
        if not original.is_file() or not packaged.is_file():
            raise FileNotFoundError(f"Missing source or packaged file: {original} / {packaged}")
        source_hash = digest(original)
        packaged_hash = digest(packaged)
        if source_hash != packaged_hash:
            raise RuntimeError(f"Copy differs from source: {packaged}")
        records.append((packaged.relative_to(ROOT).as_posix(), str(original), source_hash,
                        packaged.stat().st_size, role,
                        "review before public release" if "measurement" in role or "validation sweep" in role else ""))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("repository_path", "original_source", "sha256", "bytes", "role", "release_review"))
        writer.writerows(records)
    print(f"PASS: verified and recorded {len(records)} byte-identical research files in {output}")


if __name__ == "__main__":
    main()
