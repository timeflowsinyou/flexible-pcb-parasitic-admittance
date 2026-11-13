# -*- coding: utf-8 -*-
"""
E2 vs No-Parasitics: matched 91-terminal-pair held-out validation.

RUN: Place in D:/IEEE_TIM_related/Simulation_Experiment and press Spyder F5.
REQUIRES: the previously used E2, V2.6, V2.4, I.5 and acquisition files
at their ORIGINAL paths. All source data/checkpoints/scripts are read only.

Two control modes (select below):

  'quick_ablation' (default)
      - Keep the E2 historical unloaded intrinsic reference as the STARTING
        point / quadratic prior for both E2 and No-Parasitics.
      - No-Parasitics Yp = 0, then RE-FIT no-load and loaded intrinsic sensors.
      - Suitable as a rapid diagnostic only: the No-Parasitics prior was
        estimated historically under the E2 parasitic model.

  'historical_refit' (recommended for a journal ablation)
      - Re-fit an independent No-Parasitics intrinsic reference from the SAME
        first 644 historical 100-kHz measurements, setting Yp = 0 in the
        historical inverse network model.
      - Uses the original known-loading supports, positivity parameterization,
        5200 Adam epochs and three independently seeded initial sensor states.
      - Re-calibrates no-load intrinsic sensors separately at each frequency,
        then re-fits loaded sensors with 45/46 train-pair splits.
      - Historical refit is computationally substantial; checkpoints are saved
        every CHECKPOINT_EVERY epochs for restart-safe continuation.

Both modes:
  * E2 has the ORIGINAL frozen Gp,Cp and same independently measured 22 scans.
  * E2 and No-Parasitics receive the SAME new 91-pair unloaded calibration,
    frequency points, known loading locations, loaded-pair folds, solver,
    parameter bounds, and per-row error metrics.
  * A complementary 45/46 or 46/45 loaded-pair split is held out; the 91
    unloaded pair measurements are NOT held out.
  * The resulting 630 paired rows are NOT 630 independent hardware experiments.
  * S1...S19, S16 repeat and S2+S16 are all included: no exclusions.
  * Each run writes outputs into its own timestamped directory. No edits to
    existing inputs or historical outputs.

Outputs:
  paired_e2_nopara_holdout.csv            (630 matched rows)
  overall_metrics.csv
  metrics_by_frequency.csv
  metrics_by_condition.csv
  noload_calibration_diagnostics.csv
  historical_nopara_reference.csv       (historical_refit only)
  historical_refit_log.csv              (historical_refit only)
  Fig_E2_vs_NoParasitics.png/.pdf
  run_audit.json

Interpretation: This control tests incremental *prediction* utility of the
path-dependent E2 model, not physical accuracy of individual Gp/Cp values.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================== USER OPTIONS ===================
BASE = Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
if not BASE.is_dir():
    BASE = Path(__file__).resolve().parent
E2_VALIDATOR = BASE / 'validate_E2_locality_multifrequency.py'
OUT_ROOT = BASE / 'E2_vs_NoParasitics_control'

# First run quick_ablation, then switch to historical_refit for the paper.
CONTROL_MODE = 'quick_ablation'     # 'quick_ablation' | 'historical_refit'

# For historical_refit only:
HISTORICAL_EPOCHS = 5200
HISTORICAL_LR = 0.025
CHECKPOINT_EVERY = 200
# Match the numerical identification prior used by the E2 reference fit.
E2_HISTORICAL_PRIOR_WEIGHT = 1e-4
SENSOR_START_SEEDS = {0.05: 24, 0.30: 25, 0.70: 26}
# Resume the latest incomplete run of the same mode. To start a new run,
# set AUTO_RESUME = False (keeps previous runs untouched).
AUTO_RESUME = True

# Compare published E2 7.248932% mean complex-Z held-out error.
CHECK_E2_PUBLISHED_METRICS = True
EXPECTED_E2_Z = 0.07248932
EXPECTED_E2_MAG = 0.05101221
EXPECTED_E2_PHASE_DEG = 1.458596
E2_CHECK_TOL_ABS = 2.0e-4  # absolute metric units; robust to last digits
# ====================================================

METRICS = ('relative_Z_L2', 'relative_Y_L2', 'magnitude_MAPE', 'phase_MAE_deg')
FRACTIONS = (0.05, 0.30, 0.70)
VERSION = '1.0.0'
PRIMARY_HZ = (46415.8883361278, 100000.0, 215443.4690031884,
              464158.88336127775, 1000000.0)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def json_write(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False,
                              allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def import_py(path, name):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing required dependency: {path}')
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def new_or_resume_run(mode):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    signature = {'script_version': VERSION, 'mode': mode,
                 'epochs': HISTORICAL_EPOCHS, 'lr': HISTORICAL_LR,
                 'historical_prior_weight': E2_HISTORICAL_PRIOR_WEIGHT,
                 'seeds': {str(k): v for k, v in SENSOR_START_SEEDS.items()},
                 'primary_hz': list(PRIMARY_HZ)}
    if AUTO_RESUME and mode == 'historical_refit':
        for folder in sorted(OUT_ROOT.glob('run_*'), reverse=True):
            cfg = folder / 'run_config.json'
            if cfg.is_file() and not (folder / 'RUN_COMPLETE').is_file():
                if json.loads(cfg.read_text(encoding='utf-8')) == signature:
                    print('RESUMING:', folder, flush=True)
                    return folder, signature
    folder = OUT_ROOT / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    folder.mkdir(parents=False, exist_ok=False)
    json_write(folder / 'run_config.json', signature)
    return folder, signature


def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def train_nopara_historical(i5, out_dir, old_models, config):
    """No-parasitics historical 644-state inverse optimization.

    Shares the original I.5 forward physics and known-load support but removes
    every Gp/Cp trainable parameter (Yp identically zero).
    """
    import torch
    device = i5.DEVICE
    z_hist, y_hist, manifest, mask = i5.load_inputs()
    if y_hist.shape != (644, 91) or mask.shape != (644, 19):
        raise ValueError('Expected original I.5 644 historical states / 91 pairs / 19 sensors.')
    if not (i5.NPZ_PATH.is_file() and i5.MANIFEST_PATH.is_file() and i5.AUDIT_PATH.is_file()):
        raise FileNotFoundError('Historical input files missing.')
    source_hashes = {str(p): hash_file(p) for p in (
        i5.NPZ_PATH, i5.MANIFEST_PATH, i5.AUDIT_PATH)}
    # Use the same archived-reference construction as the E2 identification:
    # the median of the other 18 historical I.5 unloaded reference elements.
    old_lookup = {float(fr): np.asarray(base, dtype=np.complex128)
                  for fr, base, _ in old_models}
    if set(old_lookup) != set(FRACTIONS):
        raise ValueError('Unexpected archived I.5 restart definitions.')

    # Run-level source audit prevents resuming on a silently changed source file.
    audit_path = out_dir / 'historical_input_hashes.json'
    if audit_path.is_file():
        if json.loads(audit_path.read_text(encoding='utf-8')) != source_hashes:
            raise ValueError('Historical source hashes changed; refusing to resume.')
    else:
        json_write(audit_path, source_hashes)

    A_t, C_t = i5.incidence_matrices(device)
    target = torch.as_tensor(y_hist, dtype=torch.complex128, device=device)
    denom = torch.mean(torch.abs(target) ** 2)
    active_rows_np, active_cols_np = np.nonzero(mask)

    class NoParasiticsHistoricalModel(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            rng = np.random.default_rng(int(seed))
            base_g = np.full(19, 3e-6) * rng.lognormal(0.0, 0.15, 19)
            base_b = np.full(19, 1e-5) * rng.lognormal(0.0, 0.15, 19)
            self.base_g_log = torch.nn.Parameter(torch.tensor(np.log(base_g / 1e-6),
                                              dtype=torch.float64, device=device))
            self.base_b_log = torch.nn.Parameter(torch.tensor(np.log(base_b / 1e-6),
                                              dtype=torch.float64, device=device))
            self.delta_g = torch.nn.Parameter(torch.zeros(len(active_rows_np),
                                              dtype=torch.float64, device=device))
            self.delta_b = torch.nn.Parameter(torch.zeros(len(active_rows_np),
                                              dtype=torch.float64, device=device))
            self.register_buffer('active_rows', torch.tensor(active_rows_np,
                                                 dtype=torch.long, device=device))
            self.register_buffer('active_cols', torch.tensor(active_cols_np,
                                                 dtype=torch.long, device=device))

        def forward(self):
            gg = self.base_g_log.unsqueeze(0).expand(644, -1).clone()
            bb = self.base_b_log.unsqueeze(0).expand(644, -1).clone()
            gg[self.active_rows, self.active_cols] += self.delta_g
            bb[self.active_rows, self.active_cols] += self.delta_b
            sensor = torch.complex(torch.exp(gg) * 1e-6,
                                   torch.exp(bb) * 1e-6)
            _, intrinsic_y = i5.forward_y_sensor_to_intrinsic(A_t, C_t, sensor)
            return intrinsic_y   # Explicit Yp = 0.

    references = {}
    log_rows = []
    for frac in FRACTIONS:
        seed = int(SENSOR_START_SEEDS[frac])
        path = out_dir / f'historical_nopara_init_{int(frac*100):02d}pct.pt'
        model = NoParasiticsHistoricalModel(seed)
        archived_base = old_lookup[frac]
        other = np.array([j for j in range(19) if j != 1])
        prior_g = float(np.median(archived_base[other].real))
        prior_b = float(np.median(archived_base[other].imag))
        if prior_g <= 0 or prior_b <= 0:
            raise ValueError('Invalid I.5 archived reference prior.')
        prior_g_log = float(np.log(prior_g / 1e-6))
        prior_b_log = float(np.log(prior_b / 1e-6))
        # As in E2, initialize the same weakly constrained baseline component
        # at the archived reference. This is an optimizer control, not a result.
        with torch.no_grad():
            model.base_g_log[1] = prior_g_log
            model.base_b_log[1] = prior_b_log
        optimizer = torch.optim.Adam(model.parameters(), lr=HISTORICAL_LR)
        epoch_start = 0
        history = []
        if path.exists():
            # Only load checkpoints previously saved within this own output run.
            try:
                saved = torch.load(path, map_location='cpu', weights_only=False)
            except TypeError:
                saved = torch.load(path, map_location='cpu')
            expected = {'seed': seed, 'frac_tag': frac, 'epochs': HISTORICAL_EPOCHS,
                        'lr': HISTORICAL_LR, 'prior_weight': E2_HISTORICAL_PRIOR_WEIGHT,
                        'prior_g_log': prior_g_log, 'prior_b_log': prior_b_log,
                        'source_hashes': source_hashes}
            if saved.get('signature') != expected:
                raise ValueError('Historical checkpoint signature mismatch: ' + str(path))
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            epoch_start = int(saved['epoch'])
            history = list(saved['history'])
            print(f'Historical NoParasitics restart {frac}: resume from epoch {epoch_start}', flush=True)

        signature = {'seed': seed, 'frac_tag': frac, 'epochs': HISTORICAL_EPOCHS,
                     'lr': HISTORICAL_LR, 'prior_weight': E2_HISTORICAL_PRIOR_WEIGHT,
                        'prior_g_log': prior_g_log, 'prior_b_log': prior_b_log,
                        'source_hashes': source_hashes}
        for epoch in range(epoch_start + 1, HISTORICAL_EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            predicted = model()
            data_loss = torch.mean(torch.abs(predicted - target) ** 2) / denom
            weak_prior = ((model.base_g_log[1] - prior_g_log)**2 +
                          (model.base_b_log[1] - prior_b_log)**2)
            loss = data_loss + E2_HISTORICAL_PRIOR_WEIGHT * weak_prior
            if not torch.isfinite(loss):
                raise FloatingPointError(f'NoParasitics historical loss became invalid at epoch {epoch}.')
            loss.backward()
            optimizer.step()
            if epoch % CHECKPOINT_EVERY == 0 or epoch == HISTORICAL_EPOCHS:
                with torch.no_grad():
                    error = float(torch.sqrt(torch.mean(torch.abs(model()-target) ** 2) / denom).item())
                history.append({'fraction_tag': frac, 'seed': seed,
                                'epoch': epoch, 'RMS_Y_error': error})
                save_checkpoint(path, {'signature': signature, 'epoch': epoch,
                                       'model': model.state_dict(),
                                       'optimizer': optimizer.state_dict(),
                                       'history': history})
                print(f'Historical no-parasitic seed={seed}, epoch={epoch}/{HISTORICAL_EPOCHS}, '
                      f'in-sample Y RMS={100*error:.3f}%', flush=True)

        with torch.no_grad():
            base = (torch.exp(model.base_g_log).cpu().numpy() +
                    1j * torch.exp(model.base_b_log).cpu().numpy()) * 1e-6
            err = float(torch.sqrt(torch.mean(torch.abs(model()-target) ** 2) / denom).item())
        if not (np.isfinite(base).all() and np.all(base.real > 0) and np.all(base.imag > 0)):
            raise FloatingPointError('NoParasitics historical parameter estimate invalid.')
        references[frac] = base
        log_rows.extend(history)
        print(f'Historical NoParasitics restart {frac:.2f}: '
              f'final in-sample Y RMS={err*100:.3f}%', flush=True)

    pd.DataFrame(log_rows).to_csv(out_dir / 'historical_refit_log.csv', index=False)
    prm_rows = []
    for frac, base in references.items():
        for idx, value in enumerate(base):
            prm_rows.append({'restart_fraction_E2_paired': frac,
                             'historical_nopara_start_seed': SENSOR_START_SEEDS[frac],
                             'sensing_element': f'S{idx+1}',
                             'G_S': float(value.real),
                             'B_S_at_100k': float(value.imag),
                             'equivalent_C_F_at_100k': float(value.imag/(2*np.pi*1e5))})
    pd.DataFrame(prm_rows).to_csv(out_dir / 'historical_nopara_reference.csv', index=False)
    return references, source_hashes


def build_rows(v26, v24, v22, i5, A, C, raw, e2_params, no_params,
               out_dir):
    """Perform matched 630-by-630 E2 and No-Parasitics validation."""
    Z, masks, labels, names, noload, loaded, sel, freqs = raw
    if len(loaded) != 21 or len(sel) != 5 or Z.shape != (22, 10, 91):
        raise ValueError('Unexpected 22-sweep shape or loaded-condition count.')
    if not np.allclose(freqs[sel], PRIMARY_HZ, rtol=2e-7, atol=0):
        raise ValueError('Selected frequencies changed.')

    rows = []
    baseline_rows = []
    source_f = [float(freqs[fi]) for fi in sel]
    folds = [(0, np.flatnonzero(np.arange(91) % 2 == 0),
              np.flatnonzero(np.arange(91) % 2 != 0)),
             (1, np.flatnonzero(np.arange(91) % 2 == 1),
              np.flatnonzero(np.arange(91) % 2 != 1))]

    for frac in FRACTIONS:
        if frac not in e2_params or frac not in no_params:
            raise ValueError('Missing model corresponding to E2 restart tag '+str(frac))
        base_e2_100k, yp100 = e2_params[frac]
        base_no_100k = no_params[frac]
        if np.asarray(base_no_100k).shape != (19,) or np.any(np.asarray(base_no_100k).real <= 0) or np.any(np.asarray(base_no_100k).imag <= 0):
            raise ValueError('No-Parasitics historical baseline invalid for '+str(frac))

        for fi, f in zip(sel, source_f):
            yp_e2 = v24.frozen_yp_at_frequency(yp100, f)
            yp_np = np.zeros(91, dtype=np.complex128)
            variants = (
                ('E2_fixed_parasitics', base_e2_100k, yp_e2),
                ('NoParasitics_recalibrated', base_no_100k, yp_np),
            )
            for method, base100, yp in variants:
                prior = v24.ref_baseline_at_frequency(base100, f)
                baseline, bd = v24.calibrate_baseline(
                    v22, i5, A, C, 1 / Z[noload, fi], prior, yp)
                baseline_rows.append({
                    'method': method, 'restart_fraction': frac,
                    'frequency_hz': f,
                    'baseline_success': bool(bd['success']),
                    'n_components_at_bound': int(bd['n_components_at_bound']),
                    'baseline_relative_Y_residual': float(bd['relative_Y_residual']),
                    'baseline_max_abs_log_change': float(bd['max_abs_log_change']),
                })
                if not bd['success']:
                    raise RuntimeError(f'Unloaded calibration failed: {method}, f={f}, restart={frac}')
                for ix in loaded:
                    pressed = np.flatnonzero(masks[ix])
                    for fold, train, test in folds:
                        fit, diag = v24.fit_pressed(
                            v22, i5, A, C, Z[ix, fi], baseline, yp, pressed, train)
                        if fit is None or not diag['fit_success']:
                            raise RuntimeError(f'Loaded fit failed: {method} {names[ix]} '
                                               f'f={f}, restart={frac}, fold={fold}')
                        zhat = np.asarray(v22.predict_z(i5, A, C, fit, yp), dtype=complex)
                        measured = Z[ix, fi]
                        metric = v26.metrics(measured[test], zhat[test])
                        # Exact same error metric as V2.6/E2.
                        if any(not np.isfinite(metric[k]) for k in METRICS):
                            raise FloatingPointError('Non-finite held-out metric')
                        rows.append({
                            'method': method,
                            'restart_fraction': frac,
                            'frequency_hz': f,
                            'source_file': str(names[ix]),
                            'condition': str(labels[ix]),
                            'fold': fold,
                            'n_train': int(len(train)),
                            'n_heldout': int(len(test)),
                            'n_pressed': int(len(pressed)),
                            'n_active_at_bound': int(diag['n_at_bound']),
                            **metric,
                        })
            print(f'VALIDATED: restart={frac:.2f}, f={f/1000:.1f} kHz '
                  f'(E2 and No-Parasitics)', flush=True)

    full = pd.DataFrame(rows)
    expected_rows = 2*3*5*21*2
    keys = ['method','restart_fraction','frequency_hz','source_file','fold']
    if len(full) != expected_rows or full.duplicated(keys).any() or full[list(METRICS)].isna().any().any():
        raise ValueError('Expected exactly 1260 unique successful model rows.')
    if set(full.method.unique()) != {'E2_fixed_parasitics', 'NoParasitics_recalibrated'}:
        raise ValueError('Model labels missing.')

    pairs_keys = ['restart_fraction','frequency_hz','source_file','condition','fold']
    wide = full.pivot(index=pairs_keys, columns='method', values=list(METRICS))
    if wide.isna().any().any() or len(wide) != 630:
        raise ValueError('Expected 630 perfectly matched E2/NoParasitics comparisons.')
    wide.columns = [f'{metric}_{method}' for metric, method in wide.columns]
    paired = wide.reset_index()
    for metric in METRICS:
        paired[f'delta_{metric}_NoPara_minus_E2'] = (
            paired[f'{metric}_NoParasitics_recalibrated'] -
            paired[f'{metric}_E2_fixed_parasitics'])

    full.to_csv(out_dir / 'all_model_holdout_rows.csv', index=False)
    paired.to_csv(out_dir / 'paired_e2_nopara_holdout.csv', index=False)
    pd.DataFrame(baseline_rows).to_csv(out_dir / 'noload_calibration_diagnostics.csv', index=False)
    return full, paired, pd.DataFrame(baseline_rows)


def summarize(full, paired, out_dir):
    overall = full.groupby('method', as_index=False).agg(
        n_evaluations=('relative_Z_L2','size'),
        relative_Z_L2=('relative_Z_L2','mean'),
        relative_Y_L2=('relative_Y_L2','mean'),
        magnitude_MAPE=('magnitude_MAPE','mean'),
        phase_MAE_deg=('phase_MAE_deg','mean'))
    overall['relative_Z_L2_pct'] = 100*overall['relative_Z_L2']
    overall['relative_Y_L2_pct'] = 100*overall['relative_Y_L2']
    overall['magnitude_MAPE_pct'] = 100*overall['magnitude_MAPE']
    overall.to_csv(out_dir / 'overall_metrics.csv', index=False)

    freq = full.groupby(['method','frequency_hz'], as_index=False).agg(
        n_evaluations=('relative_Z_L2','size'),
        relative_Z_L2=('relative_Z_L2','mean'),
        relative_Y_L2=('relative_Y_L2','mean'),
        magnitude_MAPE=('magnitude_MAPE','mean'),
        phase_MAE_deg=('phase_MAE_deg','mean'))
    freq.to_csv(out_dir / 'metrics_by_frequency.csv', index=False)

    condition = full.groupby(['method','condition'], as_index=False).agg(
        n_evaluations=('relative_Z_L2','size'),
        relative_Z_L2=('relative_Z_L2','mean'),
        relative_Y_L2=('relative_Y_L2','mean'),
        magnitude_MAPE=('magnitude_MAPE','mean'),
        phase_MAE_deg=('phase_MAE_deg','mean'))
    condition.to_csv(out_dir / 'metrics_by_condition.csv', index=False)

    pair_summary = {}
    for metric in METRICS:
        diff = paired[f'delta_{metric}_NoPara_minus_E2'].to_numpy(float)
        pair_summary[metric] = {
            'mean_delta_no_parasitics_minus_E2': float(np.mean(diff)),
            'median_delta': float(np.median(diff)),
            'fraction_E2_better': float(np.mean(diff > 0)),
            'fraction_no_parasitics_better': float(np.mean(diff < 0)),
            'fraction_exactly_equal': float(np.mean(diff == 0)),
        }

    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    axis_x = np.arange(5)
    for label, display in [
        ('E2_fixed_parasitics', 'E2: fixed identified parasitics'),
        ('NoParasitics_recalibrated', 'No parasitics: recalibrated')]:
        sub = freq[freq['method']==label].sort_values('frequency_hz')
        if len(sub) != 5:
            raise ValueError('Missing frequency-specific metrics for '+label)
        ax.plot(axis_x, 100*sub['relative_Z_L2'].to_numpy(),
                marker='o', linewidth=1.7, label=display)
    ax.set_xticks(axis_x, ['46.4','100','215.4','464.2','1000'])
    ax.set_xlabel('Frequency / kHz')
    ax.set_ylabel('Held-out complex-Z relative L2 / %')
    ax.grid(alpha=0.2)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / 'Fig_E2_vs_NoParasitics.png', dpi=300, bbox_inches='tight')
    fig.savefig(out_dir / 'Fig_E2_vs_NoParasitics.pdf', bbox_inches='tight')
    plt.close(fig)

    print('\nOVERALL HELD-OUT RESULTS:', flush=True)
    print(overall.to_string(index=False), flush=True)
    print('\nPAIRED DIFFERENCES (NoParasitics - E2):', flush=True)
    for k,v in pair_summary.items():
        print('  '+k+': mean delta='+format(v['mean_delta_no_parasitics_minus_E2'], '.6f')+
              ', fraction E2 better='+format(v['fraction_E2_better'], '.3f'), flush=True)
    return overall, pair_summary


def main():
    if CONTROL_MODE not in ('quick_ablation', 'historical_refit'):
        raise ValueError('Invalid CONTROL_MODE')
    out_dir, config = new_or_resume_run(CONTROL_MODE)
    validator = import_py(E2_VALIDATOR, 'nopara_e2_validator')
    v26 = import_py(validator.V26, 'nopara_v26')
    ext, v24, v22, prev, i5, A, C, old, pairs, checkpoint = validator.get_models(v26)
    e2_params, e2_hashes = validator.e2_parameters(pairs)

    v27_summary_path = validator.V27_ROOT / 'V2_7_summary.json'
    if not v27_summary_path.is_file():
        raise FileNotFoundError('Missing V2.7 source audit: '+str(v27_summary_path))
    v27_summary = json.loads(v27_summary_path.read_text(encoding='utf-8-sig'))
    raw = validator.load_raw(ext, v27_summary, pairs)
    Z, masks, labels, names, noload, loaded, sel, freqs = raw
    if Counter(labels) != Counter({'noloads':1,'sensor2_16':1,
                                 **{f'sensor{i}':(2 if i==16 else 1) for i in range(1,20)}}):
        raise ValueError('Unexpected independent load conditions')
    if len(e2_params) != 3:
        raise ValueError('Expected three E2 checkpoints')

    if CONTROL_MODE == 'quick_ablation':
        no_params = {f: np.asarray(e2_params[f][0], dtype=np.complex128).copy()
                     for f in FRACTIONS}
        hist_hashes = {}
        provenance_note = ('Diagnostic ablation: NoParasitics inherits the E2 '
                           'historical intrinsic reference as its no-load prior. '
                           'This is NOT a fully independently optimized historical control.')
    else:
        no_params, hist_hashes = train_nopara_historical(i5, out_dir, old, config)
        provenance_note = ('Historical-refit control: NoParasitics intrinsic '
                           'references independently fitted to the original '
                           '644 states with zero parasitic admittance. '
                           'Three distinct initial sensor seeds are used; the '
                           'numerical identification prior has the same coefficient '
                           'and archived-reference construction as E2.')

    full, paired, diagnostics = build_rows(v26,v24,v22,i5,A,C,raw,e2_params,
                                          no_params,out_dir)
    overall, pair_summary = summarize(full,paired,out_dir)

    e2 = overall.set_index('method').loc['E2_fixed_parasitics']
    check = {'E2_Z_matches': bool(abs(float(e2.relative_Z_L2)-EXPECTED_E2_Z) < E2_CHECK_TOL_ABS),
             'E2_MAPE_matches': bool(abs(float(e2.magnitude_MAPE)-EXPECTED_E2_MAG) < E2_CHECK_TOL_ABS),
             'E2_phase_matches': bool(abs(float(e2.phase_MAE_deg)-EXPECTED_E2_PHASE_DEG) < 0.03)}
    audit = {
        'version': VERSION,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'mode': CONTROL_MODE,
        'status': 'COMPLETE' if all(check.values()) else 'E2_REFERENCE_CHECK_FAILED',
        'provenance_note': provenance_note,
        'E2_check': check,
        'n_loaded_sweeps': 21,
        'n_unloaded_sweeps': 1,
        'n_frequencies': 5,
        'n_comparisons_per_model': 630,
        'n_physical_independent_loaded_sweeps': 21,
        'unloaded_calibration_pairs': 91,
        'loaded_fit_pairs': [46,45],
        'loaded_heldout_pairs': [45,46],
        'known_active_sensors': True,
        'metric_aggregation': 'arithmetic mean of per-sweep, per-frequency, per-fold, per-initialization errors',
        'additional_notes': [
            'Yp is identically zero in the NoParasitics model at all frequencies.',
            'E2 frozen Gp,Cp are unchanged and reconstructed from E2 CSVs.',
            'NoParasitics unloaded and loaded intrinsic parameters are re-fitted, not reused from E2 predictions.',
            'No-load calibration uses all 91 measured pairs for both methods.',
            'Only loaded-state parameters are fit on training terminal pairs; 45 or 46 loaded terminal pairs are held out.',
            'The 630 paired rows are repeated numerical evaluations, not 630 physical experiments.',
            'The comparison tests held-out predictive value, not physical absolute parasitic-component accuracy.',
        ],
        'input_sources': {'E2_validator': str(E2_VALIDATOR), 'V2_6': str(validator.V26),
                          'V2_4': str(validator.V26.parent/'V2_4_fixed_parasitic_intrinsic_drift.py'),
                          'V2_7_summary': str(v27_summary_path),
                          'I5_checkpoint': str(checkpoint)},
        'input_sha256': {'E2_validator':hash_file(E2_VALIDATOR),
                         'V2_6':hash_file(Path(validator.V26)),
                         'V2_7_summary':hash_file(v27_summary_path),
                         'I5_checkpoint':hash_file(Path(checkpoint)),
                         **e2_hashes, **hist_hashes},
        'paired_difference_summary': pair_summary,
        'overall': overall.to_dict(orient='records'),
        'n_unloaded_components_at_bound_by_method': diagnostics.groupby('method')['n_components_at_bound'].agg(['mean','max']).to_dict(orient='index')
    }
    json_write(out_dir/'run_audit.json', audit)
    if CHECK_E2_PUBLISHED_METRICS and not all(check.values()):
        raise RuntimeError('E2 results failed publication cross-check. Inspect output and audit; do not cite.')
    (out_dir/'RUN_COMPLETE').write_text(datetime.now(timezone.utc).isoformat(), encoding='utf-8')
    print('\nSUCCESS. Outputs:', out_dir, flush=True)


if __name__ == '__main__':
    main()
