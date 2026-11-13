# -*- coding: utf-8 -*-
"""V1: audit and extract 91-node-pair x 10-frequency AD2 measurements.

Run with Spyder F5. No CLI options. READ-ONLY source folder, no fitting.

Source: D:\\IEEE_TIM_related\\Data\\MultiFrequency_Measurements\\2026_11_05
Three outputs in a new timestamped directory:
    V1_measurements.npz             [complex shape (N,10,91)]
    V1_manifest.csv                 [one row per original sweep]
    V1_extraction_audit.json        [file hashes, issues, statistics]

If input errors exist, a strict audit report is saved but NO NPZ/manifest is
published, preventing incomplete data from entering V2/V3 silently.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = '1.1.0-hyphen-double-support'
SOURCE_DIR = Path(r'D:\IEEE_TIM_related\Data\MultiFrequency_Measurements\2026_11_05')
OUTPUT_ROOT = Path(r'D:\IEEE_TIM_related\Simulation_Experiment\V1_multifrequency_audit')
EXPECTED_FILE_COUNT = 22
N_FREQ = 10
N_PAIR = 91
EXPECTED_FREQUENCIES_HZ = np.array([1e3 * (1e3 ** (i / 9)) for i in range(10)], dtype=np.float64)
FREQUENCY_TOL_REL = 2e-8  # target values are formatted to 10 significant digits
ACTUAL_FREQUENCY_TOL_REL = 1e-3  # record actual frequency deviations >0.1% as errors
PHASE_TOL_DEG = 0.05  # round-off tolerance for deg vs rad

CLOCKWISE_NODES = list(range(1, 8)) + list(range(14, 7, -1))
PAIRS = tuple((CLOCKWISE_NODES[i], CLOCKWISE_NODES[j]) for i in range(13)
              for j in range(i + 1, 14))
assert len(PAIRS) == N_PAIR
CANONICAL_PAIR_KEYS = tuple('-'.join(map(str, sorted(pair))) for pair in PAIRS)
EXPECTED_COLUMNS = (
    'Time', 'Sweep', 'Condition', 'Pair_Index', 'Pair',
    'Frequency_Index', 'Frequency (Hz)', 'Actual Frequency (Hz)',
    'Impedance (Ohm)', 'Phase (deg)', 'Phase (rad)', 'Status',
)


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_condition(label):
    """Return known pressed S1..S19 mask and canonical condition ID."""
    label = str(label).strip().lower()
    if label in ('noload', 'noloads', 'no_load', 'no_loads'):
        return 'noloads', np.zeros(19, dtype=np.uint8)
    m = re.fullmatch(r'sensor(1[0-9]|[1-9])(?:[-_](1[0-9]|[1-9]))?', label)
    if m is None:
        raise ValueError('Unrecognized load condition: ' + label)
    sensors = [int(m.group(1))]
    if m.group(2) is not None:
        sensors.append(int(m.group(2)))
    if len(sensors) != len(set(sensors)):
        raise ValueError('Duplicated sensor in label: ' + label)
    sensors.sort()
    mask = np.zeros(19, dtype=np.uint8)
    mask[np.array(sensors) - 1] = 1
    return 'sensor' + '_'.join(map(str, sensors)), mask


def load_one(path):
    """Extract a single sweep without averaging away repetitions or failures."""
    file_report = {'file': path.name, 'sha256': file_sha256(path), 'issues': []}
    issues = file_report['issues']
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f, delimiter='\t')
        if rd.fieldnames is None:
            raise ValueError('Missing header')
        if tuple(rd.fieldnames) != EXPECTED_COLUMNS:
            raise ValueError('Unexpected TSV header: ' + repr(rd.fieldnames))
        rows = list(rd)
    file_report['row_count'] = len(rows)
    if len(rows) != N_PAIR * N_FREQ:
        issues.append(f'Expected exactly 910 records, got {len(rows)}')

    Z = np.full((N_FREQ, N_PAIR), np.nan + 1j * np.nan, dtype=np.complex128)
    planned = np.full((N_FREQ, N_PAIR), np.nan, dtype=float)
    actual = np.full((N_FREQ, N_PAIR), np.nan, dtype=float)
    status_matrix = np.full((N_FREQ, N_PAIR), 'MISSING', dtype='<U96')
    visited = set()
    labels, sweep_ids = set(), set()
    first_times, last_times = [], []
    for lineno, row in enumerate(rows, 2):
        try:
            label = (row['Condition'] or '').strip().lower()
            labels.add(label)
            sweep_ids.add((row['Sweep'] or '').strip())
            pi = int(row['Pair_Index']) - 1
            fi = int(row['Frequency_Index']) - 1
            if not (0 <= pi < N_PAIR and 0 <= fi < N_FREQ):
                raise ValueError(f'Pair/frequency index out of range: {pi+1},{fi+1}')
            key = (fi, pi)
            if key in visited:
                raise ValueError(f'duplicate pair/frequency index: {key}')
            visited.add(key)
            if row['Pair'].strip() != CANONICAL_PAIR_KEYS[pi]:
                raise ValueError(f'pair order mismatch at index {pi+1}: {row["Pair"]!r} != {CANONICAL_PAIR_KEYS[pi]!r}')
            freq = float(row['Frequency (Hz)'])
            planned[fi, pi] = freq
            if not np.isclose(freq, EXPECTED_FREQUENCIES_HZ[fi], rtol=FREQUENCY_TOL_REL, atol=1e-7):
                raise ValueError(f'unexpected target frequency at index {fi+1}: {freq}')
            state = (row['Status'] or '').strip()
            status_matrix[fi, pi] = state
            if state != 'OK':
                issues.append(f'line {lineno}: status={state!r}')
                continue
            actual_frequency = float(row['Actual Frequency (Hz)'])
            impedance = float(row['Impedance (Ohm)'])
            degrees = float(row['Phase (deg)'])
            radians = float(row['Phase (rad)'])
            if not np.isfinite([actual_frequency, impedance, degrees, radians]).all():
                raise ValueError('nonfinite numeric value')
            if impedance <= 0 or actual_frequency <= 0:
                raise ValueError('nonpositive impedance/frequency')
            if abs(actual_frequency / freq - 1) > ACTUAL_FREQUENCY_TOL_REL:
                raise ValueError('actual frequency differs from programmed by >0.1%')
            if abs(np.degrees(radians) - degrees) > PHASE_TOL_DEG:
                raise ValueError('phase degrees/radians inconsistent')
            actual[fi, pi] = actual_frequency
            Z[fi, pi] = impedance * np.exp(1j * radians)
            if row['Time']:
                first_times.append(row['Time'])
                last_times.append(row['Time'])
        except (ValueError, TypeError, KeyError) as exc:
            issues.append(f'line {lineno}: {exc}')
    if len(visited) != N_PAIR * N_FREQ:
        issues.append(f'Expected 910 distinct indices, got {len(visited)}')
    if len(labels) != 1:
        issues.append(f'Expected one consistent Condition, got {sorted(labels)!r}')
    if len(sweep_ids) != 1:
        issues.append(f'Expected one sweep ID, got {sorted(sweep_ids)!r}')

    # Filenames are grounded by the acquisition script, not guessed from timestamps.
    basename = path.stem.lower()
    filename_m = re.fullmatch(
        r'\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}_\d+_(.+)_sweep(\d+)_91pairs_10freq', basename)
    if filename_m is None:
        issues.append('Filename does not match acquisition naming pattern')
        filename_condition = ''
    else:
        filename_condition = filename_m.group(1)
        try:
            if sweep_ids and int(filename_m.group(2)) != int(next(iter(sweep_ids))):
                issues.append('Sweep ID differs between filename and file contents')
        except ValueError:
            issues.append('Sweep ID is not integer')
    if filename_condition and labels != {filename_condition}:
        issues.append('Condition mismatch: filename vs TSV contents')
    if len(labels) == 1:
        label = next(iter(labels))
        try:
            canonical, mask = parse_condition(label)
        except ValueError as exc:
            issues.append(str(exc))
            canonical, mask = '', np.zeros(19, dtype=np.uint8)
    else:
        canonical, mask = '', np.zeros(19, dtype=np.uint8)
    valid = np.isfinite(Z.real) & np.isfinite(Z.imag)
    file_report.update(condition_raw=next(iter(labels)) if len(labels) == 1 else '',
                       condition_canonical=canonical, sweep_id=next(iter(sweep_ids)) if len(sweep_ids) == 1 else '',
                       pressed_sensors=[int(k+1) for k in np.flatnonzero(mask)],
                       n_valid=int(valid.sum()), n_invalid=int(valid.size-valid.sum()),
                       recorded_time_first=first_times[0] if first_times else '',
                       recorded_time_last=last_times[-1] if last_times else '')
    return Z, actual, mask, file_report


def run(source_dir=SOURCE_DIR, output_root=OUTPUT_ROOT):
    if not source_dir.is_dir():
        raise FileNotFoundError('Source folder missing: ' + str(source_dir))
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(exist_ok=False)
    paths = sorted(source_dir.glob('*_91pairs_10freq.txt'))
    unexpected = sorted(p.name for p in source_dir.iterdir() if p.is_file() and p not in paths)
    report = dict(version=VERSION, source_dir=str(source_dir), out_dir=str(out),
                  frequency_reference_hz=EXPECTED_FREQUENCIES_HZ.tolist(),
                  shape_convention='[sweep_index, frequency_index, canonical_pair_index]',
                  canonical_pairs=list(CANONICAL_PAIR_KEYS), expected_files=EXPECTED_FILE_COUNT,
                  found_matching_files=len(paths), unexpected_files=unexpected,
                  loaded_sensors_are_known_support_only=True,
                  input_files_modified=False,
                  warnings=['This is extraction/audit only, not a model fit or scientific validation.',
                            'Condition name denotes intended loading, not independently verified applied load.',
                            'Single repeated condition scans are retained separately; NOT silently averaged.',
                            'V1.1 fix: both sensor2-16 and sensor2_16 map to the known pressed support [2,16].',
                            'Different frequency grid from prior LCR 1-2-5 grid.'],
                  issues=[], files=[])
    if len(paths) != EXPECTED_FILE_COUNT:
        report['issues'].append(f'Expected {EXPECTED_FILE_COUNT} files, found {len(paths)}')
    if unexpected:
        report['issues'].append('Unrecognized extra files found: ' + repr(unexpected))
    zlist, aflist, masks, manifest = [], [], [], []
    for index, path in enumerate(paths):
        try:
            Z, Af, mask, info = load_one(path)
        except Exception as e:
            info = {'file':path.name, 'sha256':file_sha256(path), 'issues':['File parse error: '+repr(e)]}
            Z = np.full((N_FREQ,N_PAIR),np.nan+1j*np.nan)
            Af = np.full((N_FREQ,N_PAIR),np.nan)
            mask = np.zeros(19,dtype=np.uint8)
        info['index'] = index
        report['files'].append(info)
        if info['issues']:
            report['issues'].append(f'{path.name}: '+ '; '.join(info['issues'][:5]))
        zlist.append(Z); aflist.append(Af); masks.append(mask)
        manifest.append(dict(sweep_index=index,file_name=path.name, sha256=info['sha256'],
                             condition=info.get('condition_canonical',''),
                             original_condition=info.get('condition_raw',''),
                             pressed_sensors=','.join(map(str,info.get('pressed_sensors',[]))),
                             n_pressed=len(info.get('pressed_sensors',[])),
                             n_valid=info.get('n_valid',0), n_invalid=info.get('n_invalid',N_PAIR*N_FREQ),
                             time_first=info.get('recorded_time_first',''),
                             time_last=info.get('recorded_time_last','')))
    report['conditions_found'] = dict(Counter(m['condition'] for m in manifest))
    report['total_valid_points'] = sum(m['n_valid'] for m in manifest)
    report['total_expected_points'] = len(paths)*N_FREQ*N_PAIR
    report['accepted'] = bool(len(paths)==EXPECTED_FILE_COUNT and not report['issues'])
    audit_path=out/'V1_extraction_audit.json'
    if report['accepted']:
        zarr=np.stack(zlist)
        afarr=np.stack(aflist)
        marr=np.stack(masks)
        if not (np.isfinite(zarr.real).all() and np.isfinite(zarr.imag).all()):
            raise ValueError('Internal validation failed: non-finite Z after accepted audit')
        npz_path=out/'V1_measurements.npz'
        np.savez_compressed(npz_path,Z_measured=zarr, Z_measured_real=zarr.real,
                            Z_measured_imag=zarr.imag,frequencies_hz=EXPECTED_FREQUENCIES_HZ,
                            actual_frequencies_hz=afarr,
                            pair_ids=np.array(CANONICAL_PAIR_KEYS),
                            known_pressed_mask=marr,
                            condition_labels=np.array([m['condition'] for m in manifest]),
                            filenames=np.array([m['file_name'] for m in manifest]))
        manifest_path=out/'V1_manifest.csv'
        pd.DataFrame(manifest).to_csv(manifest_path,index=False,encoding='utf-8-sig')
        report['outputs']={p.name:file_sha256(p) for p in (npz_path,manifest_path)}
        report['output_shape']=list(zarr.shape)
    else:
        report['warnings'].append('FAIL-CLOSED: no numerical dataset or manifest exported because source audit failed.')
    audit_path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('V1 input files:',len(paths),'Expected:',EXPECTED_FILE_COUNT)
    print('Accepted:',report['accepted'])
    print('Conditions:',report['conditions_found'])
    print('Results:',out)
    if report['issues']:
        print('FIRST ISSUES:')
        for s in report['issues'][:15]:print('-',s)
    return report


if __name__ == '__main__':
    run()
