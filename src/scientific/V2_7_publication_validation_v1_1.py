# -*- coding: utf-8 -*-
"""V2.7: publication-preparation final comparison, Spyder F5, no command-line args.

Inputs: frozen V2.6 CSV/JSON (latest complete run), 22 original AD2 scans,
        original I.5 checkpoint and original V2.4/V2.6 scripts.
Only computes the missing A-control loaded fits; C comes from frozen V2.6 CSV.
Repairs S2 grouping by deriving it from CONDITION, never trusting old has_S2.
Provides matched A/C tests, 19-sensor summaries, S16 repeatability, fold diagnostics.
Output exactly 3 new files; originals unchanged; fail closed on audit mismatch.
"""
from __future__ import annotations
import hashlib, json, importlib.util, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

BASE = Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
RAW_DIR = Path(r'D:\IEEE_TIM_related\Data\MultiFrequency_Measurements\2026_11_05')
V26_ROOT = BASE / 'V2_6_22sweep_fixed_parasitic_validation'
V26_SCRIPT = BASE / 'V2_6_validate_22_sweeps_fixed_parasitics.py'
OUT_ROOT = BASE / 'V2_7_publication_validation'
EXTRACTOR = BASE / 'V1_extract_multifrequency_91pairs_v1_1.py'
V24_SCRIPT = BASE / 'V2_4_fixed_parasitic_intrinsic_drift.py'
PRIMARY_MIN_HZ = 40000.0  # five points, 46.4 kHz to 1 MHz
METRICS = ('relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg','load_deltaY_relative_L2')
RESTARTS = (0.05,0.30,0.70)


def import_py(path, name):
    if not path.is_file(): raise FileNotFoundError(str(path))
    sp = importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(sp);sys.modules[name]=mod;sp.loader.exec_module(mod)
    return mod


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def safe(obj):
    if isinstance(obj,dict):return {str(k):safe(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [safe(x) for x in obj]
    if isinstance(obj,np.ndarray):return safe(obj.tolist())
    if isinstance(obj,np.generic):return safe(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj


def stats(s):
    v=pd.to_numeric(s,errors='coerce').to_numpy(dtype=float)
    v=v[np.isfinite(v)]
    return {'n_valid':int(len(v)),'mean':float(v.mean()) if len(v) else None,
            'median':float(np.median(v)) if len(v) else None,
            'p90':float(np.percentile(v,90)) if len(v) else None}


def get_latest():
    matches=[]
    for root in V26_ROOT.glob('run_*'):
        files=[root/n for n in ('V2_6_summary.json','V2_6_loaded_holdout.csv','V2_6_intrinsic_estimates.csv')]
        if all(p.is_file() for p in files):
            report=json.loads(files[0].read_text(encoding='utf-8-sig'))
            if report.get('status')=='COMPLETE':matches.append((root,report,files))
    if not matches:raise FileNotFoundError('No complete V2.6 run with 3 outputs in '+str(V26_ROOT))
    return sorted(matches,key=lambda t:t[0].name)[-1]


def required_condition_s2(cond):
    return cond in ('sensor2','sensor2_16')


def main():
    old_dir,original,files=get_latest()
    v26=import_py(V26_SCRIPT,'v27_validated_v26')
    v24=import_py(V24_SCRIPT,'v27_validated_v24')
    parser=import_py(EXTRACTOR,'v27_validated_parser')
    v22=v24.import_v22();previous=v22.get_previous_module();i5=previous.import_i5()
    # Verify algorithms and inputs are the EXACT versions used in the frozen V2.6.
    expected=original['source_script_hashes']
    for f in (EXTRACTOR,V24_SCRIPT,v24.V22_SCRIPT):
        if sha(f)!=expected.get(str(f)):
            raise RuntimeError('Dependency changed since V2.6: '+str(f))
    if sha(previous.I5_SCRIPT)!=original['source_i5_script_sha256']:
        raise RuntimeError('I.5 model script changed since V2.6')
    rec_files=original['input_audit']['files']
    current=sorted(RAW_DIR.glob('*_91pairs_10freq.txt'))
    if len(current)!=22 or len([p for p in RAW_DIR.iterdir() if p.is_file()])!=22:
        raise RuntimeError('22-file source audit mismatch')
    expected_hash={r['file']:r['sha256'] for r in rec_files}
    if {p.name:sha(p) for p in current}!=expected_hash:
        raise RuntimeError('Original AD2 source file hash mismatch')
    Z=[]; mask=[]; condition=[]; names=[]
    for p in current:
        zz,ff,mm,info=parser.load_one(p)
        if info.get('issues'):raise RuntimeError('Extractor audit problem: '+p.name+str(info['issues']))
        Z.append(zz);mask.append(mm);condition.append(info['condition_canonical']);names.append(p.name)
    Z=np.asarray(Z,dtype=np.complex128);mask=np.asarray(mask,dtype=bool)
    names=np.asarray(names);condition=np.asarray(condition)
    freqs=np.asarray(parser.EXPECTED_FREQUENCIES_HZ,dtype=float)
    if Z.shape!=(22,10,91) or mask.shape!=(22,19):raise ValueError('Shape mismatch')
    if not np.array_equal(np.asarray(parser.CANONICAL_PAIR_KEYS).astype(str),previous.PAIR_IDS.astype(str)):
        raise ValueError('Node pair ordering mismatch')
    if not np.array_equal(np.array(['-'.join(map(str,sorted(p))) for p in i5.PAIRS]),previous.PAIR_IDS.astype(str)):
        raise ValueError('I.5 pair ordering mismatch')
    no=int(np.flatnonzero(~mask.any(axis=1))[0])
    loaded=np.flatnonzero(mask.any(axis=1))
    indices=np.flatnonzero(freqs>=PRIMARY_MIN_HZ)
    if len(loaded)!=21 or len(indices)!=5:raise ValueError('Unexpected loaded scans or frequencies')
    epoch,i5run,checkpoint,A,C,models=previous.load_i5_parameters(i5)
    if sha(checkpoint)!=original['source_i5_checkpoint_sha256']:
        raise RuntimeError('I.5 checkpoint changed')
    if not np.allclose([item[0] for item in models],RESTARTS):
        raise ValueError('I.5 restarts changed')
    rawc=pd.read_csv(files[1],encoding='utf-8-sig')
    c=rawc[rawc.protocol=='C_fixed_Yp_intrinsic_drift_LOADED_HELDOUT'].copy()
    c=c[c.frequency_hz>=PRIMARY_MIN_HZ].copy()
    if len(c)!=630 or c[['relative_Z_L2','relative_Y_L2']].isna().any().any():
        raise ValueError('Frozen C primary loaded outcomes incomplete')
    c['has_S2_verified']=c.condition.map(required_condition_s2).astype(bool)
    if set(c.condition)!=set(condition[loaded]):raise ValueError('C condition mismatch')
    expected_c_keys=set(zip(c.restart_fraction.round(6),c.frequency_hz.round(4),c.source_file,c.fold.astype(int)))
    if len(expected_c_keys)!=630:raise ValueError('C comparison keys duplicate')
    # Compute missing A for the SAME original loaded scans, SAME pair holdout.
    allrows=[]
    Y=1/Z
    for frac,base100,para100 in models:
        for fi in indices:
            f=float(freqs[fi]);yp=v24.frozen_yp_at_frequency(para100,f)
            oldbase=v24.ref_baseline_at_frequency(base100,f)
            for ix in loaded:
                pressed=np.flatnonzero(mask[ix]);label=str(condition[ix])
                for fold in (0,1):
                    train=np.flatnonzero(np.arange(91)%2==fold)
                    test=np.flatnonzero(np.arange(91)%2!=fold)
                    key=(round(float(frac),6),round(f,4),str(names[ix]),fold)
                    if key not in expected_c_keys:raise ValueError('A/C key mismatch '+repr(key))
                    fitted,diag=v24.fit_pressed(v22,i5,A,C,Z[ix,fi],oldbase,yp,pressed,train)
                    row={'protocol':'A_frozen_old_baseline_LOADED_HELDOUT',
                         'restart_fraction':float(frac),'frequency_hz':f,
                         'condition':label,'source_file':str(names[ix]),'fold':fold,
                         'has_S2_verified':required_condition_s2(label),
                         'n_pressed':int(len(pressed)),'train_pairs':len(train),
                         'test_pairs':len(test),'fit_success':bool(fitted is not None),
                         **diag}
                    if fitted is None:
                        row.update({m:np.nan for m in METRICS})
                    else:
                        zp=v22.predict_z(i5,A,C,fitted,yp)
                        row.update(v26.metrics(Z[ix,fi,test],zp[test]))
                        dyobs=Y[ix,fi,test]-Y[no,fi,test]
                        dy=v22.intrinsic_y(i5,A,C,fitted)[test]-v22.intrinsic_y(i5,A,C,oldbase)[test]
                        row['load_deltaY_relative_L2']=float(np.linalg.norm(dy-dyobs)/max(np.linalg.norm(dyobs),1e-30))
                    allrows.append(row)
            print('A control:',round(f,1),'Hz, restart',frac,flush=True)
    a=pd.DataFrame(allrows)
    if len(a)!=630:raise ValueError('Missing A outcomes')
    # C rows from original, rename and align. Avoid relying on old has_S2 column.
    c['protocol']='C_new_intrinsic_baseline_LOADED_HELDOUT'
    all_df=pd.concat([a,c],ignore_index=True,sort=False)
    all_df['has_S2_verified']=all_df.condition.map(required_condition_s2).astype(bool)
    primary=all_df[all_df.fit_success.fillna(False).astype(bool)]
    # Macro aggregation: record means are paired across folds/seeds, not independent trials.
    macro=[]
    for protocol,group in all_df.groupby('protocol'):
        for region,g in [('all',group),('without_S2',group.loc[~group.has_S2_verified]),
                         ('S2_involved',group.loc[group.has_S2_verified])]:
            r={'protocol':protocol,'region':region,'n_rows':int(len(g)),
               'n_independent_loaded_scans':int(g.source_file.nunique()),
               'n_failed':int((~g.fit_success.fillna(False).astype(bool)).sum())}
            for m in METRICS:r[m]=stats(g[m])
            macro.append(r)
    # Construct matched paired comparison, only rows where both predictions finite.
    # V2.6 CSV used %.12g floating point formatting. Match validated grid
    # indices, not exact floating point values (e.g. 46415.8883361 differs
    # from the recomputed 46415.8883361278).
    def assign_frequency_index(frame, label):
        values=pd.to_numeric(frame['frequency_hz'],errors='raise').to_numpy(dtype=float)
        grid=np.asarray(freqs,dtype=float)
        nearest=np.abs(values[:,None]-grid[None,:]).argmin(axis=1)
        if not np.isclose(values,grid[nearest],rtol=1e-9,atol=1e-5).all():
            raise ValueError(label+' has frequency values off the original 10-point grid')
        if not np.isin(nearest,indices).all():
            raise ValueError(label+' has unexpected boundary frequencies')
        frame=frame.copy()
        frame['frequency_index']=nearest.astype(int)
        return frame
    a=assign_frequency_index(a,'A')
    c=assign_frequency_index(c,'C')
    keys=['restart_fraction','frequency_index','condition','source_file','fold']
    pair=a[keys+['frequency_hz']+list(METRICS)].merge(
        c[keys+['frequency_hz']+list(METRICS)],on=keys,how='outer',
        validate='one_to_one',indicator=True,suffixes=('_A','_C'))
    if len(pair)!=630 or not pair['_merge'].eq('both').all():
        failed=pair.loc[pair['_merge']!='both',keys+['_merge']].head(10)
        raise ValueError('Unmatched A/C comparison: '+repr(pair['_merge'].value_counts().to_dict())+'; '+failed.to_string(index=False))
    if not np.isclose(pair['frequency_hz_A'],pair['frequency_hz_C'],rtol=1e-9,atol=1e-5).all():
        raise ValueError('A/C frequencies disagree beyond CSV rounding')
    pair['frequency_hz']=pair['frequency_hz_A']
    pair.drop(columns=['frequency_hz_A','frequency_hz_C'],inplace=True)
    pair['relative_Z_L2_improvement_A_minus_C']=pair['relative_Z_L2_A']-pair['relative_Z_L2_C']
    valid=pair.dropna(subset=['relative_Z_L2_A','relative_Z_L2_C'])
    paired={'n':int(len(valid)),
            'mean_error_reduction_pp':float(valid.relative_Z_L2_improvement_A_minus_C.mean()*100),
            'fraction_C_better':float((valid.relative_Z_L2_C<valid.relative_Z_L2_A).mean()),
            'mean_A':float(valid.relative_Z_L2_A.mean()),
            'mean_C':float(valid.relative_Z_L2_C.mean())}
    # Whole-array per-sensor accuracy: single-sensor scan, no double and no duplicate as new unique sensor.
    per=[]
    es=pd.read_csv(files[2],encoding='utf-8-sig')
    es=es[(es.type=='pressed')&(es.frequency_hz>=PRIMARY_MIN_HZ)].copy()
    for sensor_idx in range(1,20):
        label='sensor'+str(sensor_idx)
        g=c[c.condition==label];single=es[(es.condition==label)&(es.sensor=='S'+str(sensor_idx))]
        zvals=pd.to_numeric(single['fold_complex_Z_disagreement'],errors='coerce').dropna()
        if len(zvals):
            unique=single.drop_duplicates(['restart_fraction','frequency_hz','source_file','sensor'])
            zvals=pd.to_numeric(unique.fold_complex_Z_disagreement,errors='coerce').dropna()
        per.append({'sensor':'S'+str(sensor_idx),'n_loaded_scans':int(g.source_file.nunique()),
            'n_eval_rows':int(len(g)),'complex_Z_holdout_error_mean':stats(g.relative_Z_L2)['mean'],
            'magnitude_MAPE_mean':stats(g.magnitude_MAPE)['mean'],
            'phase_MAE_deg_mean':stats(g.phase_MAE_deg)['mean'],
            'fold_complex_Z_disagreement_median':float(zvals.median()) if len(zvals) else None,
            'fold_complex_Z_disagreement_gt50pct_fraction':float((zvals>0.5).mean()) if len(zvals) else None,
            'n_fold_comparisons':int(len(zvals))})
    sensor_table=pd.DataFrame(per)
    # S16 repeat from frozen V2.6 (no recomputation); each freq has one actual physical repeat pair.
    repeat=[]
    original_raw=original['s16_repeated_raw_measurement']
    original_intrinsic=original['s16_repeated_intrinsic_reconstruction']
    transfer=rawc[rawc.protocol=='C_sensor16_CROSS_SWEEP_TRANSFER']
    for f in freqs[indices]:
        rr=[r for r in original_raw if np.isclose(r['frequency_hz'],f)]
        ri=[r for r in original_intrinsic if np.isclose(r['frequency_hz'],f)]
        tr=transfer[np.isclose(transfer.frequency_hz,f)]
        if len(rr)!=3 or len(ri)!=6 or len(tr)!=12:
            raise ValueError('Expected S16 duplicate metrics 3 restart * 2 folds at f='+str(f))
        repeat.append({'frequency_hz':float(f),
           'n_actual_hardware_repeat_pairs':1,
           'raw_91_Z_symmetric_relative_L2':float(np.mean([r['raw_Z_relative_L2_symmetric'] for r in rr])),
           'raw_deltaY_symmetric_relative_L2':float(np.mean([r['raw_deltaY_relative_L2_symmetric'] for r in rr])),
           'recovered_S16_complex_Z_disagreement_mean':float(np.mean([r['Z_complex_disagreement'] for r in ri])),
           'cross_sweep_transfer_relative_Z_L2_mean':stats(tr.relative_Z_L2)['mean'],
           'n_cross_sweep_eval_rows':int(len(tr))})
    report={'experiment':'V2.7_final_submission_preparation_A_C_and_repeats','version':'1.1-frequency-index-merge',
       'source_V26_run':str(old_dir),'source_V26_files_sha256':{p.name:sha(p) for p in files},
       'source_V26_script_sha256':sha(V26_SCRIPT),'source_checkpoint_sha256':sha(checkpoint),
       'source_raw_files_verified_sha256':expected_hash,'primary_frequencies_hz':[float(freqs[k]) for k in indices],
       'method':'Original V2.6 C re-used unchanged; A fitted on same 22 scans using I.5 old baseline scaled as RC, frozen parasitics; identical train/holdout pairs and known press labels.',
       's2_filter_fix':'S2 involvement derived from condition equal to sensor2 or sensor2_16; NOT the old pandas mixed-dtype bitwise negation.',
       'primary_macro_metrics':macro,'paired_A_C_result':paired,
       'by_frequency':[{'protocol':p,'frequency_hz':float(f),'relative_Z_L2':stats(g.relative_Z_L2),
                        'magnitude_MAPE':stats(g.magnitude_MAPE),'phase_MAE_deg':stats(g.phase_MAE_deg)}
                       for (p,f),g in all_df.groupby(['protocol','frequency_hz'])],
       'per_single_sensor_C':per,'S16_repeatability':repeat,
       'limitations':['No direct intrinsic impedance ground truth; relative Z errors describe held-out pair prediction only.',
        'No-load C calibration used ALL 91 paths; only load-specific sensor parameters are held out on alternating pair subsets.',
        'A has uncalibrated old baseline, C uses the newly acquired no-load scan: comparison measures value of calibration, not identical calibration budget.',
        'S16 is ONE physical repeat pair; three checkpoint restarts and two folds are not independent physical repeats.',
        'Load identity is known from file name; activation is not inferred. Actual force not quantified in these files.',
        'Only 22 scans from one measurement session, no time-series generalization claim.',
        'Repeatability of inferred intrinsic S16 is NOT direct measurement repeatability.']}
    json_txt=json.dumps(safe(report),ensure_ascii=False,indent=2,allow_nan=False)
    out=OUT_ROOT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    # Only 3 output files, original source untouched.
    all_df.to_csv(out/'V2_7_matched_A_C_holdout.csv',index=False,float_format='%.12g',encoding='utf-8-sig')
    sensor_table.to_csv(out/'V2_7_per_sensor.csv',index=False,float_format='%.12g',encoding='utf-8-sig')
    (out/'V2_7_summary.json').write_text(json_txt,encoding='utf-8')
    print('V2.7 finished; outputs:',out)
    print('Matched A/C:',paired)
    print('S16 repeat summary:',repeat)
    return report

if __name__=='__main__':
    main()
