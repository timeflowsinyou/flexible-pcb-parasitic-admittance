# -*- coding: utf-8 -*-
r"""
Chapter 4 figure-only generator (Windows/Spyder, final selected E2 parameters).

Place this script in D:\IEEE_TIM_related\Simulation_Experiment and run once:
    %runfile D:/IEEE_TIM_related/Simulation_Experiment/rebuild_Fig2_Fig3_Fig4_E2_windows_fixed_download.py --wdir

Reads EXISTING calculated CSVs, does not re-fit parameters or alter original files.
Finds CSVs automatically in the project directory and timestamped subfolders.
Does not use old E0/V2.7 per-sensor validation data.

Fig. 2: 100-kHz unloaded: measured / compensated / forward response.
Fig. 3: frequency-specific calibration, complex Z and complex Y held-out errors.
        Prefers the audited post-hoc correction applied symmetrically to BEFORE
        and AFTER metrics. Otherwise explicitly labels and uses UNFILTERED data.
Fig. 4: loaded sensor-resolved held-out complex-Z prediction error (E2 only).

Quantity/unit on plot axes:  |Z| / kΩ, Phase / °, Frequency / kHz,
                             Relative error / %, etc.

Outputs: a new timestamped folder under Chapter4_E2_Figures_Updated\run_*
         Fig2_*.png/.pdf/.svg, Fig3_*.png/.pdf/.svg,
         Fig4_*.png/.pdf/.svg, source tables and audit.json.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # file-only figures; works in Spyder without pop-up windows
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MultipleLocator
import numpy as np
import pandas as pd

# ----------------------------- USER SETTINGS ----------------------------
# No /mnt/data path anywhere: use the folder containing this script.
BASE = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE / 'Chapter4_E2_Figures_Updated'

# Optional: set an explicit file path only if your existing CSV is stored
# outside BASE. Default None means automatic discovery.
FIG2_CSV_OVERRIDE = None
FIG3_CSV_OVERRIDE = None
FIG4_CSV_OVERRIDE = None

# If true, refuse to generate a main Fig. 3 from unfiltered results when an
# audited corrected frequency summary is missing. Default false: show honest,
# clearly marked unfiltered results instead of inventing a correction.
REQUIRE_AUDITED_CORRECTED_FIG3 = False

# Main paper figures sized for insertion into A4 at approx. 170-180 mm width.
FIGURE_WIDTH_IN = 7.15
SAVE_DPI = 450

# Internal identification tag only in filename/audit, not in figure labels.
MAIN_RESTART = 0.30
APPROX_FREQS_KHZ = (46.4, 100.0, 215.4, 464.2, 1000.0)

EXPECTED_PAIR_NODES = (1, 2, 3, 4, 5, 6, 7, 14, 13, 12, 11, 10, 9, 8)
EXPECTED_PAIRS = [(EXPECTED_PAIR_NODES[i], EXPECTED_PAIR_NODES[j])
                  for i in range(13) for j in range(i + 1, 14)]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def figure_style():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.labelsize': 12.2,
        'axes.titlesize': 12.0,
        'xtick.labelsize': 10.7,
        'ytick.labelsize': 10.7,
        'legend.fontsize': 10.3,
        'lines.linewidth': 1.8,
        'axes.linewidth': 0.8,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'grid.color': '#B4B4B4',
        'grid.alpha': 0.22,
        'grid.linewidth': 0.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.facecolor': 'white',
    })


def safe_plot(fig, out, stem):
    for ext in ('png', 'pdf', 'svg'):
        kw = {'bbox_inches': 'tight', 'pad_inches': 0.08}
        if ext == 'png':
            kw['dpi'] = SAVE_DPI
        fig.savefig(out / f'{stem}.{ext}', **kw)
    plt.close(fig)


def _files_by_exact_name(name):
    """Resolve from known output locations first; no guessed filename suffixes."""
    direct = BASE / name
    results = [direct] if direct.is_file() else []
    # Search project subfolders, not all other drives or virtual /mnt paths.
    for p in BASE.rglob(name):
        if p.is_file() and p not in results and OUTPUT_ROOT not in p.parents:
            results.append(p)
    return sorted(results, key=lambda p: p.stat().st_mtime, reverse=True)


def find_input(possible_names, override=None):
    if override is not None:
        p = Path(override).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f'Explicit input does not exist: {p}')
        return p
    for filename in possible_names:
        choices = _files_by_exact_name(filename)
        if choices:
            # Prefer a valid root-level CSV, then latest timestamped result.
            direct = BASE / filename
            return direct if direct.is_file() else choices[0]
    raise FileNotFoundError(
        'Required E2 result CSV not found. Tried names: ' + ', '.join(possible_names)
        + '\nSearch root: ' + str(BASE)
        + '\nRun Chapter4_validation_E2_one_run_corrected.py first if necessary.')


def check_columns(df, required, path):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'{path.name}: missing columns: {sorted(missing)}')


def check_complex_values(*arrays):
    if not all(np.isfinite(np.asarray(x)).all() for x in arrays):
        raise ValueError('Non-finite curve values; STOP instead of plotting.')


def pair_identity(label):
    nums = [int(v) for v in re.findall(r'\d+', str(label))]
    if len(nums) != 2:
        raise ValueError('Invalid pair label: ' + str(label))
    return tuple(sorted(nums))


def read_fig2(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    if {'measured_Z_abs_ohm', 'compensated_Z_abs_ohm', 'forward_Z_abs_ohm'}.issubset(df.columns):
        # Earlier dedicated F2 script; already magnitude + phase values.
        check_columns(df, ['pair_index_1based', 'pair_label',
                           'measured_phase_deg', 'compensated_phase_deg',
                           'forward_phase_deg'], path)
        if 'restart_fraction' in df.columns:
            df = df[np.isclose(df['restart_fraction'].astype(float), MAIN_RESTART)]
        df = df.sort_values('pair_index_1based')
        idx = df.pair_index_1based.to_numpy(dtype=int)
        labels = df.pair_label.to_numpy()
        magnitudes = [df[k].to_numpy(dtype=float) / 1000 for k in
                      ('measured_Z_abs_ohm','compensated_Z_abs_ohm','forward_Z_abs_ohm')]
        phases = [df[k].to_numpy(dtype=float) for k in
                  ('measured_phase_deg','compensated_phase_deg','forward_phase_deg')]
    else:
        # Output from integrated one-run validation script.
        check_columns(df, ['restart_fraction','pair_index_1based','pair',
                           'measured_Z_real','measured_Z_imag',
                           'compensated_Z_real','compensated_Z_imag',
                           'forward_Z_real','forward_Z_imag'], path)
        df = df[np.isclose(df.restart_fraction.astype(float), MAIN_RESTART)]
        df = df.sort_values('pair_index_1based')
        idx = df.pair_index_1based.to_numpy(dtype=int)
        labels = df.pair.to_numpy()
        cols = ('measured', 'compensated', 'forward')
        z = [df[f'{name}_Z_real'].to_numpy(dtype=float)
             + 1j * df[f'{name}_Z_imag'].to_numpy(dtype=float) for name in cols]
        magnitudes = [np.abs(x) / 1000 for x in z]
        phases = [np.angle(x, deg=True) for x in z]
    if len(df) != 91 or not np.array_equal(idx, np.arange(1, 92)):
        raise ValueError(f'Fig. 2 requires exactly 91 ordered pairs, got {len(df)} from {path}')
    for i, label in enumerate(labels):
        if pair_identity(label) != tuple(sorted(EXPECTED_PAIRS[i])):
            raise ValueError(f'Fig. 2 pair order mismatch at index {i+1}: {label!r}')
    check_complex_values(*magnitudes, *phases)
    return idx, magnitudes, phases


def fig2(path, out):
    idx, mags, phases = read_fig2(path)
    names = ('Measured', 'Compensated', 'Intrinsic forward model')
    colors = ('#2878B5', '#EF8123', '#259649')
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_IN, 5.35),
                                  sharex=True, gridspec_kw={'hspace': 0.16})
    for z, ph, lab, color in zip(mags, phases, names, colors):
        ax0.plot(idx, z, color=color, label=lab)
        ax1.plot(idx, ph, color=color, label=lab)
    ax0.set_yscale('log')
    ax0.set_ylabel(r'$|Z|\,/\,\mathrm{k}\Omega$')
    ax1.set_ylabel(r'$\varphi\,/\,{}^\circ$')
    ax1.set_xlabel('Terminal-pair index')
    ax1.set_xlim(1, 91)
    ax1.set_xticks(np.arange(10, 91, 10))
    ax0.set_ylim(bottom=max(1.0, min(m.min() for m in mags) * 0.78))
    ax0.legend(ncol=3, loc='upper center', frameon=False, columnspacing=0.95,
               bbox_to_anchor=(0.5, 1.015))
    for ax in (ax0, ax1):
        ax.grid(True, which='major')
        ax.tick_params(direction='in', top=False, right=False)
    fig.align_ylabels()
    fig.subplots_adjust(left=0.105, right=0.99, top=0.965, bottom=0.11)
    safe_plot(fig, out, 'Fig2_terminal_pair_compensation')
    return {'pairs': len(idx), 'plotted_restart_fraction': MAIN_RESTART}


def read_fig3(path):
    df = pd.read_csv(path, encoding='utf-8-sig').sort_values('frequency_hz')
    check_columns(df, ['frequency_hz'], path)
    if {'before_relative_Z_L2', 'after_relative_Z_L2',
        'before_relative_Y_L2', 'after_relative_Y_L2'}.issubset(df.columns):
        df = df.rename(columns={
            'before_relative_Z_L2': 'before_Z', 'after_relative_Z_L2': 'after_Z',
            'before_relative_Y_L2': 'before_Y', 'after_relative_Y_L2': 'after_Y'})
    check_columns(df, ['before_Z','after_Z','before_Y','after_Y'], path)
    if len(df) != 5 or df.frequency_hz.nunique() != 5:
        raise ValueError(f'Expected 5 distinct validation frequencies in {path}')
    for col in ['before_Z','after_Z','before_Y','after_Y']:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Invalid Fig. 3 data in {col}')
    return df


def _create_unfiltered_summary_from_paired(paired_path, out):
    d = pd.read_csv(paired_path, encoding='utf-8-sig')
    check_columns(d, ['frequency_hz','before_Z','after_Z','before_Y','after_Y'], paired_path)
    if len(d) != 630:
        raise ValueError(f'Expected 630 E2 validation rows, found {len(d)}')
    grouped = d.groupby('frequency_hz', as_index=False)[
        ['before_Z','after_Z','before_Y','after_Y']].mean()
    dest = out / 'Fig3_unfiltered_derived_source.csv'
    grouped.to_csv(dest, index=False)
    return dest


def locate_fig3(out):
    if FIG3_CSV_OVERRIDE is not None:
        p = find_input([], FIG3_CSV_OVERRIDE)
        mode = 'corrected_symmetric' if 'CORRECTED' in p.name.upper() else 'unfiltered_or_unspecified'
        return p, mode
    # Prefer the full one-run corrected dataset with symmetric exclusion
    # in both BEFORE and AFTER metrics. No hidden post-hoc file editing.
    for name in ('E2_Fig4_per_frequency_CORRECTED.csv',):
        found = _files_by_exact_name(name)
        if found:
            return found[0], 'corrected_symmetric'
    if REQUIRE_AUDITED_CORRECTED_FIG3:
        raise FileNotFoundError('Audited symmetric corrected Fig. 3 CSV not found. '
            'Please run Chapter4_validation_E2_one_run_corrected.py first. '
            'Expected E2_Fig4_per_frequency_CORRECTED.csv.')
    print('NOTICE: No audited symmetric corrected CSV found. '
          'Fig. 3 will show ALL held-out paths (unfiltered), not a silently altered graph.')
    for name in ('E2_Fig4_per_frequency_UNFILTERED.csv',
                 'E2_calibration_ablation_by_frequency.csv'):
        found = _files_by_exact_name(name)
        if found:
            return found[0], 'unfiltered'
    paired = find_input(['E2_baseline_calibration_ablation_paired.csv'])
    return _create_unfiltered_summary_from_paired(paired, out), 'unfiltered'


def fig3(source, mode, out):
    df = read_fig3(source)
    x = np.arange(5)
    frequencies = [f'{f:.1f}' if f not in (100.0,1000.0) else str(int(f))
                   for f in APPROX_FREQS_KHZ]
    colors = ('#2878B5', '#EF8123')
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, 3.55), sharex=True)
    for ax, a, b, label, suffix in (
        (ax0, 'before_Z', 'after_Z', r'Relative $L_2$ error of $Z$', '(a)'),
        (ax1, 'before_Y', 'after_Y', r'Relative $L_2$ error of $Y$', '(b)')):
        ax.plot(x, 100.0*df[a].to_numpy(dtype=float), '-o', ms=5.7,
                color=colors[0], label='Original unloaded baseline')
        ax.plot(x, 100.0*df[b].to_numpy(dtype=float), '-s', ms=5.7,
                color=colors[1], label='Recalibrated unloaded baseline')
        ax.set_xticks(x, frequencies)
        ax.set_xlabel(r'Frequency / kHz')
        ax.set_ylabel(label + ' / %')
        ax.set_title(suffix, loc='left', pad=6, fontsize=11.5)
        ax.grid(True, which='major')
        ax.tick_params(direction='in')
        ax.margins(x=0.07)
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=10.0)
    if mode == 'corrected_symmetric':
        fig.text(0.5, 0.002, 'Post-hoc sensitivity: S1, 46.4 kHz, Fold 1, path 1–2 excluded in both stages.',
                 ha='center', va='bottom', fontsize=8.2)
    else:
        fig.text(0.5, 0.002, 'Unfiltered evaluation: all originally held-out paths included.',
                 ha='center', va='bottom', fontsize=8.2)
    fig.subplots_adjust(left=.11, right=.99, bottom=.24, top=.80, wspace=.39)
    safe_plot(fig, out, 'Fig3_frequency_validation')
    df.to_csv(out / 'Fig3_frequency_source.csv', index=False)
    return {'mode': mode, 'means': {k:float(df[k].mean()) for k in ['before_Z','after_Z','before_Y','after_Y']}}


def derive_sensorwise_from_paired(paired_path, out):
    d = pd.read_csv(paired_path, encoding='utf-8-sig')
    check_columns(d, ['condition', 'source_file'], paired_path)
    if len(d) != 630:
        raise ValueError(f'Unexpected validation length {len(d)} in {paired_path}')
    d = d[d['condition'].str.fullmatch(r'sensor(?:[1-9]|1[0-9])')].copy()
    if 'after_Z' in d.columns:
        # E2 calibration-ablation table uses these AFTER values (not old baselines).
        fields = {'after_Z':'complex_Z_L2_mean', 'after_magnitude_MAPE':'magnitude_MAPE_mean',
                  'after_phase_MAE_deg':'phase_MAE_deg_mean'}
    elif 'relative_Z_L2_E2_weak_S2_prior' in d.columns:
        # E0/E2 locality comparison; select only E2 result columns.
        fields = {'relative_Z_L2_E2_weak_S2_prior':'complex_Z_L2_mean',
                  'magnitude_MAPE_E2_weak_S2_prior':'magnitude_MAPE_mean',
                  'phase_MAE_deg_E2_weak_S2_prior':'phase_MAE_deg_mean'}
    else:
        raise ValueError('Cannot detect E2 sensor-resolved metric columns in ' + str(paired_path))
    g = d.groupby('condition', as_index=False).agg(
        physical_scans=('source_file','nunique'),
        n_computational_rows=('source_file','size'),
        **{target:(source,'mean') for source, target in fields.items()})
    g['sensor'] = g.condition.str.replace('sensor','S',regex=False)
    g['order'] = g.sensor.str[1:].astype(int)
    g = g.sort_values('order').drop(columns=['order','condition'])
    if len(g) != 19 or not set(g.sensor) == set(f'S{i}' for i in range(1,20)):
        raise ValueError('Expected results for all 19 single-sensor loads')
    return g


def read_fig4(out):
    if FIG4_CSV_OVERRIDE is not None:
        source = find_input([], FIG4_CSV_OVERRIDE)
    else:
        source = None
        # E2 sensor summary only; do NOT load old V2_7_per_sensor.csv.
        for name in ('E2_sensorwise_summary.csv',):
            found = _files_by_exact_name(name)
            if found:
                source = found[0]
                break
        if source is None:
            source = find_input(['E2_baseline_calibration_ablation_paired.csv',
                                 'E0_E2_five_frequency_paired.csv'])
    df = pd.read_csv(source,encoding='utf-8-sig')
    if 'condition' in df.columns:
        df = derive_sensorwise_from_paired(source, out)
        df.to_csv(out / 'Fig4_derived_E2_sensorwise.csv', index=False)
    else:
        check_columns(df, ['sensor'], source)
        zcol = next((c for c in ('complex_Z_L2_mean', 'E2_relative_Z_L2_mean')
                     if c in df.columns), None)
        if zcol is None:
            raise ValueError('E2 sensor summary missing complex-Z L2 column: '+str(source))
        df = df.rename(columns={zcol:'complex_Z_L2_mean'})
    if len(df) != 19:
        raise ValueError(f'Fig. 4 requires 19 sensors; found {len(df)}')
    df['sensor_idx'] = df['sensor'].astype(str).str.extract(r'S(\d+)').astype(int)
    df = df.sort_values('sensor_idx')
    check_complex_values(df.complex_Z_L2_mean)
    return df, source


def fig4(df, out):
    vals = 100.0*df['complex_Z_L2_mean'].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 3.4))
    xx = np.arange(1, 20)
    ax.bar(xx, vals, width=0.74, color='#2878B5', alpha=0.9)
    ax.set_xticks(xx, [f'S{i}' for i in xx], rotation=40, ha='right')
    ax.set_xlabel('Loaded sensor')
    ax.set_ylabel(r'Mean relative $L_2$ error of $Z$ / %')
    ax.set_ylim(0, max(vals)*1.15)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    ax.tick_params(direction='in')
    fig.subplots_adjust(left=.135, right=.985, bottom=.25, top=.95)
    safe_plot(fig, out, 'Fig4_sensorwise_holdout')
    df.to_csv(out / 'Fig4_sensorwise_source.csv', index=False)
    return {'min_percent':float(vals.min()),'max_percent':float(vals.max())}


def main():
    figure_style()
    stamp = datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    out = OUTPUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=False)

    p2 = find_input(['E2_Fig2_data.csv','F2_E2_main_init30_source_data.csv'], FIG2_CSV_OVERRIDE)
    print('Fig. 2 source:',p2,flush=True)
    s2 = fig2(p2, out)

    p3, mode3 = locate_fig3(out)
    print('Fig. 3 source:',p3, '| mode:',mode3,flush=True)
    s3 = fig3(p3,mode3,out)

    d4,p4 = read_fig4(out)
    print('Fig. 4 source:',p4,flush=True)
    s4 = fig4(d4,out)

    sources = {'Fig2':p2,'Fig3':p3,'Fig4':p4}
    audit = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        input_root=str(BASE), output_dir=str(out),
        identification='Joint forward-model identification with weak logarithmic regularization of unloaded S2 (lambda=1e-4); plotted model startup fraction 0.30 for Fig2',
        sources={k:{'path':str(v),'sha256':sha256(v)} for k,v in sources.items()},
        figure2=s2,figure3=s3,figure4=s4,
        units={'Z':'|Z| / kΩ','phase':'φ / °','frequency':'Frequency / kHz','relative_error':'... / %'},
        scientific_notes=[
            'Fig2 uses all 91 unloaded paths for intrinsic-baseline calibration and is not an independent intrinsic-impedance ground truth.',
            'Fig3 post-hoc exclusion is applied to both pre- and post-calibration stages only when audited corrected source exists.',
            'Fig4 uses unfiltered per-sensor E2 results from the selected E2 identification, not the historical V2.7 model.',
            '630 evaluations are not 630 independent hardware scans.',
            'Original CSVs are preserved; this script writes a new timestamped output directory.'
        ])
    (out/'figure_build_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nDONE:',out,flush=True)
    print('Generated Fig. 2, Fig. 3 and Fig. 4 (PNG, PDF, SVG).',flush=True)
    if mode3 != 'corrected_symmetric':
        print('REMINDER: Fig. 3 is unfiltered. To make the corrected version, '
              'first run the audited Chapter 4 one-run validation script.',flush=True)
    return out

if __name__ == '__main__':
    main()


#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


FIG3_CSV = r"D:\IEEE_TIM_related\Simulation_Experiment\E2_calibration_ablation_by_frequency.csv"
OUT_DIR = r"D:\IEEE_TIM_related\Simulation_Experiment\Fig3_only_updated"


def main():
    df = pd.read_csv(FIG3_CSV)

    # -------- identify columns --------
    if "frequency_khz" in df.columns:
        freq = df["frequency_khz"]
    elif "freq_khz" in df.columns:
        freq = df["freq_khz"]
    elif "frequency" in df.columns:
        freq = df["frequency"]
        if freq.max() > 5000:
            freq = freq / 1000.0
    elif "freq_hz" in df.columns:
        freq = df["freq_hz"] / 1000.0
    else:
        raise ValueError("No frequency column found.")

    possible_old = [
        "z_l2_old_percent",
        "old_impedance_l2_percent",
        "old_relative_l2_impedance_percent",
        "old_complex_z_l2_percent",
        "heldout_complex_z_l2_old_percent",
    ]
    possible_new = [
        "z_l2_recalibrated_percent",
        "recalibrated_impedance_l2_percent",
        "recalibrated_relative_l2_impedance_percent",
        "new_complex_z_l2_percent",
        "heldout_complex_z_l2_recalibrated_percent",
    ]

    old_col = next((c for c in possible_old if c in df.columns), None)
    new_col = next((c for c in possible_new if c in df.columns), None)

    if old_col is None or new_col is None:
        raise ValueError(f"Cannot identify required impedance L2 columns. Columns are: {df.columns.tolist()}")

    plot_df = pd.DataFrame({
        "freq_khz": freq,
        "old": df[old_col],
        "new": df[new_col]
    }).sort_values("freq_khz")

    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(plot_df["freq_khz"], plot_df["old"], marker="o", linewidth=2.5, markersize=8,
            label="Old unloaded baseline")
    ax.plot(plot_df["freq_khz"], plot_df["new"], marker="o", linewidth=2.5, markersize=8,
            label="Recalibrated unloaded baseline")

    ax.set_xlabel("Frequency / kHz")
    ax.set_ylabel("Relative L2 error of impedance / %")
    ax.set_title("Frequency-dependent validation of impedance prediction")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_dir / "Fig3_impedance_relative_L2_only.png", dpi=450, bbox_inches="tight")
    fig.savefig(out_dir / "Fig3_impedance_relative_L2_only.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "Fig3_impedance_relative_L2_only.svg", bbox_inches="tight")
    plt.close(fig)

    print("Done.")
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()