# -*- coding: utf-8 -*-
"""Create 1-2 loaded-case impedance-recovery figures based on E2.

Purpose
-------
Generate publication-style figures, analogous to Fig. 2, but for LOADED
measurement conditions. Each figure shows, across all 91 terminal pairs:
    - directly measured impedance,
    - impedance after subtraction of the fixed E2 parasitic admittances,
    - PARASITIC-FREE intrinsic-network forward impedance after fitting the
      known active sensing element(s).

The E2 original held-out error is evaluated against the full forward response
INCLUDING parasitics. It is calculated and saved separately, but this full
measurement-space prediction is not substituted for the intrinsic forward
curve in the figure.

Important:
- READ ONLY: no existing data, checkpoints, or prior results are modified.
- Uses the same independent 22-sweep dataset and the same E2 parameter set.
- Uses the same held-out loaded-state protocol as the existing validation:
  fit active-element parameters on 45/46 training pairs and evaluate on the
  complementary held-out pairs.

Default selection
-----------------
By default, the script produces TWO figures at 100 kHz, restart 0.30, fold 0:
    1) one representative single-sensor loaded case, selected automatically as
       the single-sensor condition whose held-out complex-Z error is closest to
       the median across S1...S19 at that frequency/fold/restart;
    2) the combined S2-S16 loaded case.

You may switch to manual selection at the top of the script.

Outputs
-------
A fresh timestamped run folder under:
    D:/IEEE_TIM_related/Simulation_Experiment/E2_loaded_recovery_figures/
containing:
    - one PNG and one PDF per selected case,
    - case_summary.csv
    - per_case_91pair_data_*.csv
    - run_audit.json

Run with Spyder F5.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================= User options =========================
BASE = Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
if not BASE.is_dir():
    BASE = Path(__file__).resolve().parent

E2_VALIDATOR = BASE / 'validate_E2_locality_multifrequency.py'
OUT_ROOT = BASE / 'E2_loaded_recovery_figures_corrected'

# Fixed validation setting for the new figures.
RESTART_FRACTION = 0.30
FREQUENCY_HZ = 100000.0
FOLD = 0                    # train on even-indexed (0-based parity) pairs, test on odd; 1 reverses
SELECTION_MODE = 'auto'     # 'auto' or 'manual'

# Used only if SELECTION_MODE == 'manual'.
# Example conditions: 'sensor13', 'sensor16', 'sensor2_16'
MANUAL_CASES = [
    {'condition': 'sensor13', 'frequency_hz': 100000.0, 'restart_fraction': 0.30, 'fold': 0},
    {'condition': 'sensor2_16', 'frequency_hz': 100000.0, 'restart_fraction': 0.30, 'fold': 0},
]
# ===============================================================


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(x):
    if isinstance(x, dict):
        return {str(k): strict_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [strict_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return strict_json(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def import_file(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError('Missing required file: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def nearest_freq_index(freqs_hz: np.ndarray, f_target: float) -> int:
    idx = int(np.argmin(np.abs(freqs_hz - float(f_target))))
    if not np.isclose(freqs_hz[idx], float(f_target), rtol=2e-7, atol=1e-6):
        raise ValueError(f'Frequency {f_target} Hz not found exactly in extracted data.')
    return idx


def pair_labels_from_pairs(pairs):
    return [f'{int(a)}-{int(b)}' for a, b in pairs]


def heldout_indices(n_pairs: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(n_pairs, dtype=int)
    train = np.flatnonzero(all_idx % 2 == int(fold))
    test = np.flatnonzero(all_idx % 2 != int(fold))
    return train, test


def compute_loaded_case_result(v26, v24, v22, i5, A, C,
                               Z, masks, labels, names, noload_idx,
                               pair_labels, condition, frequency_hz,
                               restart_fraction, fold, e2_params):
    """Reconstruct one loaded case using the audited E2 workflow."""
    loaded_idx_candidates = np.flatnonzero(labels == str(condition))
    if len(loaded_idx_candidates) == 0:
        raise ValueError(f'Condition not found in independent dataset: {condition}')
    if len(loaded_idx_candidates) > 1:
        # Keep deterministic behaviour. This only occurs for sensor16.
        loaded_idx = int(loaded_idx_candidates[0])
    else:
        loaded_idx = int(loaded_idx_candidates[0])

    freq_vec = np.asarray(v26.import_file(v26.EXTRACTOR, 'tmp_extract_for_freqs').EXPECTED_FREQUENCIES_HZ, dtype=float)
    fi = nearest_freq_index(freq_vec, frequency_hz)

    if float(restart_fraction) not in e2_params:
        raise ValueError(f'Restart fraction {restart_fraction} not available in E2 parameters.')
    base100, yp100 = e2_params[float(restart_fraction)]

    yp = v24.frozen_yp_at_frequency(yp100, float(frequency_hz))
    prior = v24.ref_baseline_at_frequency(base100, float(frequency_hz))

    baseline, base_diag = v24.calibrate_baseline(
        v22, i5, A, C,
        1 / Z[noload_idx, fi],
        prior, yp,
    )
    if not base_diag['success']:
        raise RuntimeError(f'Unloaded baseline calibration failed for f={frequency_hz} Hz.')

    train, test = heldout_indices(91, fold)
    pressed = np.flatnonzero(masks[loaded_idx])
    yfit, fit_diag = v24.fit_pressed(
        v22, i5, A, C,
        Z[loaded_idx, fi],
        baseline, yp,
        pressed,
        train,
    )
    if yfit is None or not fit_diag['fit_success']:
        raise RuntimeError(f'Loaded-state fit failed for {condition}, f={frequency_hz}, fold={fold}.')

    z_meas = np.asarray(Z[loaded_idx, fi], dtype=np.complex128)
    # IMPORTANT: distinguish TWO different forward quantities.
    # 1) Intrinsic-network response excludes the parasitics and must be
    #    compared with the directly compensated observed impedance.
    # 2) Complete measurement response includes the parasitics and must be
    #    compared with the raw measurement for the original held-out metric.
    y_intrinsic_forward = np.asarray(
        v22.intrinsic_y(i5, A, C, yfit), dtype=np.complex128
    )
    z_intrinsic_forward = 1.0 / y_intrinsic_forward
    z_model_measured = 1.0 / (y_intrinsic_forward + yp)

    # Reproduce the original E2 validation prediction (with parasitics).
    z_model_measured_reference = np.asarray(
        v22.predict_z(i5, A, C, yfit, yp), dtype=np.complex128
    )
    if not np.allclose(z_model_measured, z_model_measured_reference,
                       rtol=1e-8, atol=1e-7):
        raise RuntimeError('Forward-model equivalence check failed.')

    y_meas = 1.0 / z_meas
    y_comp = y_meas - yp
    if not (np.isfinite(y_comp).all() and (np.abs(y_comp) > 1e-30).all()):
        raise ValueError('Compensated admittance contains zero or invalid values.')
    z_comp = 1.0 / y_comp

    # Original E2 held-out prediction score: measured vs FULL model.
    metrics_holdout = v26.metrics(z_meas[test], z_model_measured[test])
    metrics_all = v26.metrics(z_meas, z_model_measured)
    # Additional diagnostic: compensated measurement vs intrinsic forward.
    comp_vs_forward_holdout = v26.metrics(z_comp[test], z_intrinsic_forward[test])
    comp_vs_forward_all = v26.metrics(z_comp, z_intrinsic_forward)

    pair_index = np.arange(1, 92, dtype=int)
    df = pd.DataFrame({
        'pair_index_1based': pair_index,
        'pair': pair_labels,
        'is_train_pair': np.isin(np.arange(91), train),
        'is_holdout_pair': np.isin(np.arange(91), test),
        'Z_measured_real_ohm': z_meas.real,
        'Z_measured_imag_ohm': z_meas.imag,
        'Z_compensated_real_ohm': z_comp.real,
        'Z_compensated_imag_ohm': z_comp.imag,
        'Z_intrinsic_forward_real_ohm': z_intrinsic_forward.real,
        'Z_intrinsic_forward_imag_ohm': z_intrinsic_forward.imag,
        'Z_full_model_measured_real_ohm': z_model_measured.real,
        'Z_full_model_measured_imag_ohm': z_model_measured.imag,
        'abs_Z_measured_ohm': np.abs(z_meas),
        'abs_Z_compensated_ohm': np.abs(z_comp),
        'abs_Z_intrinsic_forward_ohm': np.abs(z_intrinsic_forward),
        'abs_Z_full_model_measured_ohm': np.abs(z_model_measured),
        'phase_Z_measured_deg': np.angle(z_meas, deg=True),
        'phase_Z_compensated_deg': np.angle(z_comp, deg=True),
        'phase_Z_intrinsic_forward_deg': np.angle(z_intrinsic_forward, deg=True),
        'phase_Z_full_model_measured_deg': np.angle(z_model_measured, deg=True),
    })

    active_sensors = [f'S{idx+1}' for idx in pressed]
    condition_type = 'single_loaded' if len(pressed) == 1 else 'multi_loaded'

    return {
        'condition': str(condition),
        'source_file': str(names[loaded_idx]),
        'condition_type': condition_type,
        'active_sensors': ','.join(active_sensors),
        'frequency_hz': float(frequency_hz),
        'restart_fraction': float(restart_fraction),
        'fold': int(fold),
        'n_train_pairs': int(len(train)),
        'n_holdout_pairs': int(len(test)),
        'baseline_diag': base_diag,
        'fit_diag': fit_diag,
        'metrics_holdout': metrics_holdout,
        'metrics_all91_measured_vs_predicted': metrics_all,
        'metrics_compensated_vs_intrinsic_holdout': comp_vs_forward_holdout,
        'metrics_all91_compensated_vs_intrinsic': comp_vs_forward_all,
        'dataframe': df,
    }


def select_cases_auto(v26, v24, v22, i5, A, C,
                      Z, masks, labels, names, noload_idx,
                      pair_labels, restart_fraction, frequency_hz,
                      fold, e2_params):
    """Objective default selection: one median-error single-sensor case + S2-S16."""
    records = []
    single_conditions = sorted({str(x) for x in labels if str(x).startswith('sensor') and '_' not in str(x)})
    for cond in single_conditions:
        result = compute_loaded_case_result(
            v26, v24, v22, i5, A, C,
            Z, masks, labels, names, noload_idx,
            pair_labels, cond, frequency_hz,
            restart_fraction, fold, e2_params
        )
        records.append({
            'condition': cond,
            'relative_Z_L2_holdout': float(result['metrics_holdout']['relative_Z_L2']),
            'magnitude_MAPE_holdout': float(result['metrics_holdout']['magnitude_MAPE']),
            'phase_MAE_deg_holdout': float(result['metrics_holdout']['phase_MAE_deg']),
        })

    tab = pd.DataFrame(records).sort_values('relative_Z_L2_holdout').reset_index(drop=True)
    target = float(tab['relative_Z_L2_holdout'].median())
    rep_idx = int(np.argmin(np.abs(tab['relative_Z_L2_holdout'].to_numpy() - target)))
    representative_single = str(tab.loc[rep_idx, 'condition'])

    cases = [
        {
            'condition': representative_single,
            'frequency_hz': float(frequency_hz),
            'restart_fraction': float(restart_fraction),
            'fold': int(fold),
            'selection_reason': 'single-sensor case closest to median held-out complex-Z error',
        }
    ]
    if 'sensor2_16' in set(labels.astype(str)):
        cases.append({
            'condition': 'sensor2_16',
            'frequency_hz': float(frequency_hz),
            'restart_fraction': float(restart_fraction),
            'fold': int(fold),
            'selection_reason': 'combined loaded case',
        })
    return cases, tab


def plot_case(result: dict, out_dir: Path) -> None:
    df = result['dataframe'].copy()
    x = df['pair_index_1based'].to_numpy(dtype=float)
    holdout = df['is_holdout_pair'].to_numpy(dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.6), sharex=True)

    # Magnitude
    axes[0].plot(x, df['abs_Z_measured_ohm'].to_numpy() / 1e3, label='Measured', linewidth=1.6)
    axes[0].plot(x, df['abs_Z_compensated_ohm'].to_numpy() / 1e3, label='Compensated', linewidth=1.6)
    axes[0].plot(x, df['abs_Z_intrinsic_forward_ohm'].to_numpy() / 1e3,
                 label='Intrinsic forward model', linewidth=1.6)
    axes[0].scatter(x[holdout], (df.loc[holdout, 'abs_Z_intrinsic_forward_ohm'].to_numpy() / 1e3),
                    marker='o', s=18, facecolors='none', edgecolors='k', linewidths=0.8,
                    label='Held-out pairs')
    axes[0].set_ylabel('|Z| / kΩ')
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.25)

    # Phase
    axes[1].plot(x, df['phase_Z_measured_deg'].to_numpy(), linewidth=1.6)
    axes[1].plot(x, df['phase_Z_compensated_deg'].to_numpy(), linewidth=1.6)
    axes[1].plot(x, df['phase_Z_intrinsic_forward_deg'].to_numpy(), linewidth=1.6)
    axes[1].scatter(x[holdout], df.loc[holdout, 'phase_Z_intrinsic_forward_deg'].to_numpy(),
                    marker='o', s=18, facecolors='none', edgecolors='k', linewidths=0.8)
    axes[1].set_ylabel('φ / °')
    axes[1].set_xlabel('Terminal-pair index')
    axes[1].grid(alpha=0.25)

    # Show the shared legend in the upper-right corner of the PHASE subplot.
    # The labeled handles come from the magnitude subplot, including the
    # hollow-circle held-out marker; both subplots use the same series.
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].legend(handles, labels, loc='upper center', fontsize=8)

    hz = result['frequency_hz']
    # Keep performance metrics outside the figure: the held-out metric refers
    # to measured vs FULL (parasitic-aware) prediction, whereas the green
    # curve here is the intrinsic-only forward impedance.
    # fig.suptitle(f"Loaded condition: {result['active_sensors']} | "
    #              f"{hz/1000:.1f} kHz", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    stem = (f"Fig_loaded_recovery_{result['condition']}_"
            f"{int(round(hz))}Hz_init{int(round(100*result['restart_fraction'])):02d}pct_"
            f"fold{result['fold']}")
    fig.savefig(out_dir / (stem + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / (stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)

    df.to_csv(out_dir / (stem + '_91pair_data.csv'), index=False)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_ROOT / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out_dir.mkdir(parents=False, exist_ok=False)

    e2val = import_file(E2_VALIDATOR, 'loaded_recovery_e2_validator')
    v26 = import_file(e2val.V26, 'loaded_recovery_v26')
    ext, v24, v22, prev, i5, A, C, old, pairs, ckpt = e2val.get_models(v26)
    e2_params, e2_hashes = e2val.e2_parameters(pairs)

    summary_path = e2val.V27_ROOT / 'V2_7_summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError('Missing V2.7 summary JSON: ' + str(summary_path))
    summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))

    Z, masks, labels, names, noload_idx, loaded_idx, sel, freqs = e2val.load_raw(ext, summary, pairs)
    pair_labels = pair_labels_from_pairs(pairs)

    # Case selection
    selection_table = None
    if SELECTION_MODE.lower() == 'auto':
        cases, selection_table = select_cases_auto(
            v26, v24, v22, i5, A, C,
            Z, masks, labels, names, noload_idx,
            pair_labels, RESTART_FRACTION, FREQUENCY_HZ,
            FOLD, e2_params
        )
    elif SELECTION_MODE.lower() == 'manual':
        cases = []
        for item in MANUAL_CASES:
            cases.append({
                'condition': str(item['condition']),
                'frequency_hz': float(item.get('frequency_hz', FREQUENCY_HZ)),
                'restart_fraction': float(item.get('restart_fraction', RESTART_FRACTION)),
                'fold': int(item.get('fold', FOLD)),
                'selection_reason': 'manual selection',
            })
    else:
        raise ValueError("SELECTION_MODE must be 'auto' or 'manual'.")

    # Reconstruct and plot selected cases.
    summary_rows = []
    for case in cases:
        res = compute_loaded_case_result(
            v26, v24, v22, i5, A, C,
            Z, masks, labels, names, noload_idx,
            pair_labels,
            case['condition'], case['frequency_hz'],
            case['restart_fraction'], case['fold'],
            e2_params,
        )
        plot_case(res, out_dir)
        summary_rows.append({
            'condition': res['condition'],
            'source_file': res['source_file'],
            'condition_type': res['condition_type'],
            'active_sensors': res['active_sensors'],
            'frequency_hz': res['frequency_hz'],
            'restart_fraction': res['restart_fraction'],
            'fold': res['fold'],
            'n_train_pairs': res['n_train_pairs'],
            'n_holdout_pairs': res['n_holdout_pairs'],
            'selection_reason': case.get('selection_reason', ''),
            'heldout_relative_Z_L2_pct': 100 * res['metrics_holdout']['relative_Z_L2'],
            'heldout_relative_Y_L2_pct': 100 * res['metrics_holdout']['relative_Y_L2'],
            'heldout_magnitude_MAPE_pct': 100 * res['metrics_holdout']['magnitude_MAPE'],
            'heldout_phase_MAE_deg': res['metrics_holdout']['phase_MAE_deg'],
            'all91_measured_vs_predicted_relative_Z_L2_pct': 100 * res['metrics_all91_measured_vs_predicted']['relative_Z_L2'],
            'holdout_compensated_vs_intrinsic_relative_Z_L2_pct': 100 * res['metrics_compensated_vs_intrinsic_holdout']['relative_Z_L2'],
            'holdout_compensated_vs_intrinsic_relative_Y_L2_pct': 100 * res['metrics_compensated_vs_intrinsic_holdout']['relative_Y_L2'],
            'all91_compensated_vs_intrinsic_relative_Z_L2_pct': 100 * res['metrics_all91_compensated_vs_intrinsic']['relative_Z_L2'],
            'baseline_relative_Y_residual_pct': 100 * float(res['baseline_diag']['relative_Y_residual']),
            'baseline_n_components_at_bound': int(res['baseline_diag']['n_components_at_bound']),
            'loaded_fit_train_cost': float(res['fit_diag']['fit_train_cost']),
            'loaded_n_active_at_bound': int(res['fit_diag']['n_at_bound']),
        })

    pd.DataFrame(summary_rows).to_csv(out_dir / 'case_summary.csv', index=False)
    if selection_table is not None:
        selection_table.to_csv(out_dir / 'auto_single_sensor_screening_100k_fold0_init30pct.csv', index=False)

    audit = {
        'script': str(Path(__file__).resolve()),
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_mode': SELECTION_MODE,
        'default_restart_fraction': RESTART_FRACTION,
        'default_frequency_hz': FREQUENCY_HZ,
        'default_fold': FOLD,
        'selected_cases': summary_rows,
        'source_files': {
            'E2_validator': str(E2_VALIDATOR),
            'V2_6_script': str(e2val.V26),
            'V2_7_summary': str(summary_path),
            'I5_checkpoint': str(ckpt),
        },
        'source_hashes': {
            'E2_validator_sha256': sha256(E2_VALIDATOR),
            'V2_6_script_sha256': sha256(Path(e2val.V26)),
            'V2_7_summary_sha256': sha256(summary_path),
            'I5_checkpoint_sha256': sha256(Path(ckpt)),
            **e2_hashes,
        },
        'notes': [
            'Each loaded figure shows all 91 terminal pairs; active-element parameters were fitted only on the training subset defined by fold parity.',
            'Held-out pair locations are marked by hollow circles on the intrinsic forward-predicted curves.',
            'The green intrinsic forward response excludes Yp; its correct comparator is the compensated measurement (orange).',
            'The original E2 held-out error is measured vs the FULL parasitic-aware model, NOT green vs orange.',
            'Measured, compensated, and INTRINSIC-forward curves are displayed in both magnitude and phase.',
            'The script is read-only with respect to all existing checkpoints, raw measurements, and prior validation outputs.',
        ],
    }
    (out_dir / 'run_audit.json').write_text(json.dumps(strict_json(audit), indent=2), encoding='utf-8')

    print('Done. Output folder:')
    print(out_dir)


if __name__ == '__main__':
    main()
