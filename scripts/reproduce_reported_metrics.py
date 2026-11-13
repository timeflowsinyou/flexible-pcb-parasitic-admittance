"""Re-aggregate the manuscript metrics from packaged per-evaluation tables."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOL = 5e-12


def close(actual: float, expected: float, label: str) -> None:
    if abs(actual - expected) > TOL:
        raise AssertionError(f"{label}: {actual:.15g} != {expected:.15g}")


def main() -> None:
    detail = pd.read_csv(ROOT / "results/validation/E0_E2_five_frequency_holdout_detail.csv")
    selected = detail[detail["scenario"] == "E2_weak_S2_prior"]
    metrics = {
        "relative_Z_L2": float(selected["relative_Z_L2"].mean()),
        "relative_Y_L2": float(selected["relative_Y_L2"].mean()),
        "magnitude_MAPE": float(selected["magnitude_MAPE"].mean()),
        "phase_MAE_deg": float(selected["phase_MAE_deg"].mean()),
    }
    expected = {
        "relative_Z_L2": 0.07248932352514488,
        "relative_Y_L2": 0.11221025025086662,
        "magnitude_MAPE": 0.051012205181862974,
        "phase_MAE_deg": 1.4585956612044686,
    }
    for key in expected:
        close(metrics[key], expected[key], key)

    no_para = pd.read_csv(ROOT / "results/no_parasitics/paired_e2_nopara_holdout.csv")
    np_metrics = {
        "relative_Z_L2": float(no_para["relative_Z_L2_NoParasitics_recalibrated"].mean()),
        "magnitude_MAPE": float(no_para["magnitude_MAPE_NoParasitics_recalibrated"].mean()),
        "phase_MAE_deg": float(no_para["phase_MAE_deg_NoParasitics_recalibrated"].mean()),
    }
    close(np_metrics["relative_Z_L2"], 0.30721563956543607, "no-parasitics relative_Z_L2")
    close(np_metrics["magnitude_MAPE"], 0.21694665987847198, "no-parasitics MAPE")
    close(np_metrics["phase_MAE_deg"], 1.4613390296959967, "no-parasitics phase")

    print(json.dumps({
        "status": "PASS",
        "aggregation": "arithmetic mean over 21 loaded sweeps x 5 frequencies x 2 folds x 3 starts",
        "n_evaluations": len(selected),
        "selected_fixed_parasitics": metrics,
        "no_parasitics_recalibrated": np_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
