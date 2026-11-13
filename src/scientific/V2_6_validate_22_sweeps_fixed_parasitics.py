# -*- coding: utf-8 -*-
"""V2.6 — AUDITED 22-sweep external-batch validation (Spyder F5, no CLI).

Original measurements: D:/IEEE_TIM_related/Data/MultiFrequency_Measurements/2026_11_05
Expected: 1 no-load, S1..S19 once, S16 twice in total, S2+S16 once.
Exactly 22 x 910 recorded complex impedances (10 frequencies x 91 pairs).

Physics: physical I.5 14-node forward; frozen 91-path Gp,Cp; at each
frequency fit 19 intrinsic baseline G,B ONLY from the separate no-load sweep.
For each loaded sweep fit ONLY its known pressed sensor G,B on even or odd
pairs; evaluate complementary held-out loaded pair measurements.
Three I.5 converged restarts (not independent hardware repetitions).

Five PRIMARY frequencies: 46.4k to 1M. Two BOUNDARY diagnostic frequencies:
10k and 21.5k; 1k-4.64k excluded because of 100kOhm AD2 reference mismatch.
The reference's performance remains a *hypothesis*, not established here.

Extra genuine repeat: sensor16 loaded twice. Report raw complex Z repeatability,
load-effect repeatability, and cross-sweep transfer of sensor16 fitted on one
scan and evaluated against the other scan WITHOUT target-scan fitting.

Three outputs in a fresh run directory:
  V2_6_summary.json            extraction audit + full summary + limitations
  V2_6_loaded_holdout.csv      91D prediction test + cross-sweep transfer
  V2_6_intrinsic_estimates.csv baseline and pressed sensor estimates
No original input files, checkpoints, or prior results are modified.

CAUTION: all 91 no-load pairs were used to calibrate the baseline; loaded
holdout pairs are held out only from PRESS parameter fitting. Known pressed
support is supplied. No direct sensor-Z ground truth. S2 is retained and
reported separately, never silently discarded.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys, traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
SOURCE=Path(r'D:\IEEE_TIM_related\Data\MultiFrequency_Measurements\2026_11_05')
OUT_ROOT=BASE/'V2_6_22sweep_fixed_parasitic_validation'
EXTRACTOR=BASE/'V1_extract_multifrequency_91pairs_v1_1.py'
V24=BASE/'V2_4_fixed_parasitic_intrinsic_drift.py'
VERSION='1.0.0'
F_REF=100000.0
F_MIN=10000.0
PRIMARY_F_MIN=40000.0
RESTARTS=(0.05,0.30,0.70)
# Match previously declared V2.4 constraints; no tuning using test labels.
MAX_BOUND_COMPONENT_LOG=np.log(10.0)  # 19-sensor no-load G and B: .1x..10x
LOADED_MIN_FACTOR=0.1
LOADED_MAX_FACTOR=100.0
S16_ID=15  # 0-based


def import_file(path,name):
    if not path.is_file():raise FileNotFoundError('Required unchanged script missing: '+str(path))
    spec=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for piece in iter(lambda:stream.read(1<<20),b''):h.update(piece)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def metrics(z,zp):
    z=np.asarray(z,dtype=complex);zp=np.asarray(zp,dtype=complex)
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(zp)) and np.all(abs(z)>1e-30) and np.all(abs(zp)>1e-30)):
        return {k:np.nan for k in ('relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg')}
    return dict(relative_Z_L2=float(np.linalg.norm(zp-z)/np.linalg.norm(z)),
       relative_Y_L2=float(np.linalg.norm(1/zp-1/z)/np.linalg.norm(1/z)),
       magnitude_MAPE=float(np.mean(np.abs(abs(zp)-abs(z))/abs(z))),
       phase_MAE_deg=float(np.mean(abs(np.angle(zp/z,deg=True)))))


def group_summary(df,cols,groupby):
    out=[]
    for key,g in df.groupby(groupby,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        item={k:(v.item() if isinstance(v,np.generic) else v) for k,v in zip(groupby,key)}
        item['rows']=int(len(g))
        for col in cols:
            v=pd.to_numeric(g[col],errors='coerce').to_numpy(dtype=float)
            v=v[np.isfinite(v)]
            item[col]={'mean':float(v.mean()) if len(v) else None,
                       'median':float(np.median(v)) if len(v) else None,
                       'p90':float(np.percentile(v,90)) if len(v) else None,
                       'n_valid':int(len(v))}
        out.append(item)
    return out


def run():
    if not SOURCE.is_dir():raise FileNotFoundError('Source folder not found: '+str(SOURCE))
    extract=import_file(EXTRACTOR,'v26_extraction_parser')
    v24=import_file(V24,'v26_frozen_parasitic_model')
    v22=v24.import_v22()
    previous=v22.get_previous_module()
    i5=previous.import_i5()

    # Strict file audit. No reliance on screenshot as source of numerical data.
    paths=sorted(SOURCE.glob('*_91pairs_10freq.txt'))
    all_files=[p for p in SOURCE.iterdir() if p.is_file()]
    unexpected=sorted(p.name for p in all_files if p not in paths)
    audit=dict(source_dir=str(SOURCE),n_expected=22,n_matching=len(paths),
               unexpected_files=unexpected,issues=[],files=[])
    if len(paths)!=22:audit['issues'].append('Exactly 22 scans required, found '+str(len(paths)))
    # Other files in the raw source folder are not silently consumed.
    if unexpected:audit['issues'].append('Unrecognized file(s): '+repr(unexpected))
    zz=[]; masks=[]; labels=[]; names=[]
    for p in paths:
        try:
            z,actual_freq,mask,info=extract.load_one(p)
        except Exception as exc:
            info={'file':p.name,'sha256':hash_file(p),'issues':[repr(exc)]}
            z=None;mask=None
        audit['files'].append(info)
        if info.get('issues'):
            audit['issues'].append(p.name+': '+'; '.join(map(str,info['issues'][:5])))
        if z is None:continue
        zz.append(z)
        masks.append(np.asarray(mask,dtype=bool))
        labels.append(info['condition_canonical']);names.append(p.name)
    counts=Counter(labels)
    audit['conditions']=dict(sorted(counts.items()))
    expected={'noloads':1,'sensor2_16':1,**{f'sensor{i}':(2 if i==16 else 1) for i in range(1,20)}}
    if counts!=Counter(expected):
        audit['issues'].append('Condition multiplicity mismatch; expected '+repr(expected))
    if audit['issues']:
        # On audit failure save audit ONLY, do not fit or publish partial metrics.
        out=OUT_ROOT/('failed_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
        out.mkdir(parents=True,exist_ok=False)
        (out/'V2_6_summary.json').write_text(json.dumps(clean({'status':'FAILED_AUDIT',
              'audit':audit}),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        raise RuntimeError('Raw data audit failed; full details at '+str(out))
    Z=np.stack(zz);masks=np.stack(masks);names=np.asarray(names);labels=np.asarray(labels)
    freqs=np.asarray(extract.EXPECTED_FREQUENCIES_HZ,dtype=float)
    if Z.shape!=(22,10,91) or masks.shape!=(22,19):raise ValueError('Unexpected validated raw shapes')
    if not np.array_equal(np.asarray(extract.CANONICAL_PAIR_KEYS).astype(str),previous.PAIR_IDS.astype(str)):
        raise ValueError('91 pair mapping mismatch between extractor and validated I.5 pipeline')
    canonical_i5=np.array(['-'.join(map(str,sorted(p))) for p in i5.PAIRS])
    if not np.array_equal(canonical_i5,previous.PAIR_IDS.astype(str)):
        raise ValueError('I.5 pair ordering mismatch')
    f_indices=np.flatnonzero(freqs>=F_MIN*(1-1e-9))
    if len(f_indices)!=7:raise ValueError('Expected seven frequencies >=10kHz')
    loaded=np.flatnonzero(masks.any(axis=1))
    no_idx=int(np.flatnonzero(~masks.any(axis=1))[0])
    s16_indices=np.flatnonzero(labels=='sensor16')
    if len(s16_indices)!=2:raise ValueError('S16 repeat missing')
    epoch,i5_run,ckpt,A,C,models=previous.load_i5_parameters(i5)
    if len(models)!=3 or not np.allclose([x[0] for x in models],RESTARTS):
        raise ValueError('Unexpected I.5 initializations')
    Y=1/Z
    rows=[];estimates=[];base_diagnostics=[];repeat=[]
    allmetric=('relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg')
    for restart,base100,yp100 in models:
        print('I.5 restart:',restart,flush=True)
        if (np.any(base100.real<=0) or np.any(base100.imag<=0) or np.any(yp100.real<0) or np.any(yp100.imag<0)):
            raise ValueError('Nonpassive I.5 parameter set')
        for fi in f_indices:
            f=float(freqs[fi]);freq_group='primary' if f>=PRIMARY_F_MIN else 'boundary_diagnostic'
            yp=v24.frozen_yp_at_frequency(yp100,f)
            prior=v24.ref_baseline_at_frequency(base100,f)
            baseline,bd=v24.calibrate_baseline(v22,i5,A,C,Y[no_idx,fi],prior,yp)
            base_diagnostics.append(dict(restart_fraction=float(restart),frequency_hz=f,
                                         frequency_group=freq_group,**bd))
            if not bd['success']:
                print('Warning: baseline optimization failed',restart,f,flush=True)
            # Keep a full complex baseline for all 19 sensors.
            for sensor in range(19):
                b=baseline[sensor]; z0=1/b
                estimates.append(dict(type='baseline',restart_fraction=float(restart),
                    frequency_hz=f,frequency_group=freq_group,condition='noloads',
                    source_file=names[no_idx],fold=-1,sensor=f'S{sensor+1}',
                    G_s=float(b.real),B_s=float(b.imag),Z_real_ohm=float(z0.real),
                    Z_imag_ohm=float(z0.imag),Z_abs_ohm=float(abs(z0)),
                    Z_phase_deg=float(np.angle(z0,deg=True)),
                    at_bound=np.nan,fit_success=bool(bd['success'])))
            # Include original I.5 control for no-load calibration checks.
            for proto,bs in [('A_old_baseline_RC_control',prior),('C_new_intrinsic_baseline',baseline)]:
                zpred=v22.predict_z(i5,A,C,bs,yp)
                rows.append(dict(protocol=proto+'_NOLOAD_CALIBRATION_NOT_TEST',
                     restart_fraction=float(restart),frequency_hz=f,frequency_group=freq_group,
                     condition='noloads',source_file=names[no_idx],fold=-1,
                     train_pairs=91,test_pairs=0,fit_success=bool(bd['success']),
                     **metrics(Z[no_idx,fi],zpred)))
            # All 21 loaded scans (19 distinct singles, extra S16, S2/S16 double).
            fitted={}  # (sweep_id, fold) -> calibrated fitted sensor vector
            for ix in loaded:
                pressed=np.flatnonzero(masks[ix])
                for fold in (0,1):
                    train=np.flatnonzero(np.arange(91)%2==fold)
                    test=np.flatnonzero(np.arange(91)%2!=fold)
                    rec=dict(protocol='C_fixed_Yp_intrinsic_drift_LOADED_HELDOUT',
                             restart_fraction=float(restart),frequency_hz=f,
                             frequency_group=freq_group,condition=str(labels[ix]),
                             source_file=str(names[ix]),sweep_index=int(ix),fold=fold,
                             n_pressed=int(len(pressed)),has_S2=bool(1 in pressed),
                             train_pairs=int(len(train)),test_pairs=int(len(test)),
                             baseline_success=bool(bd['success']),
                             baseline_bound_components=int(bd['n_components_at_bound']))
                    if bd['success']:
                        fit,fitdiag=v24.fit_pressed(v22,i5,A,C,Z[ix,fi],baseline,yp,pressed,train)
                        rec.update(fitdiag)
                    else:
                        fit=None;rec.update(fit_success=False,fit_status='baseline_fit_failed')
                    if fit is not None:
                        zp=v22.predict_z(i5,A,C,fit,yp)
                        rec.update(metrics(Z[ix,fi,test],zp[test]))
                        dyo=Y[ix,fi,test]-Y[no_idx,fi,test]
                        dyp=v22.intrinsic_y(i5,A,C,fit)[test]-v22.intrinsic_y(i5,A,C,baseline)[test]
                        rec['load_deltaY_relative_L2']=float(np.linalg.norm(dyp-dyo)/max(np.linalg.norm(dyo),1e-30))
                        fitted[(int(ix),fold)]=fit
                        for sensor in pressed:
                            yy=fit[sensor];zs=1/yy
                            estimates.append(dict(type='pressed',restart_fraction=float(restart),
                                frequency_hz=f,frequency_group=freq_group,
                                condition=str(labels[ix]),source_file=str(names[ix]),
                                sweep_index=int(ix),fold=fold,sensor=f'S{sensor+1}',
                                G_s=float(yy.real),B_s=float(yy.imag),
                                Z_real_ohm=float(zs.real),Z_imag_ohm=float(zs.imag),
                                Z_abs_ohm=float(abs(zs)),Z_phase_deg=float(np.angle(zs,deg=True)),
                                at_bound=int(fitdiag['n_at_bound']),fit_success=True))
                    else:
                        rec.update({k:np.nan for k in allmetric})
                        rec['load_deltaY_relative_L2']=np.nan
                    rows.append(rec)
            # Real S16 re-measurement; independent LOADED scan, same no-load reference.
            u,v=map(int,s16_indices)
            repeat.append(dict(restart_fraction=float(restart),frequency_hz=f,
                 frequency_group=freq_group,
                 repeated_sweep_1=str(names[u]),repeated_sweep_2=str(names[v]),
                 raw_Z_relative_L2_symmetric=float(np.linalg.norm(Z[u,fi]-Z[v,fi])/
                     max(0.5*(np.linalg.norm(Z[u,fi])+np.linalg.norm(Z[v,fi])),1e-30)),
                 raw_deltaY_relative_L2_symmetric=float(np.linalg.norm((Y[u,fi]-Y[no_idx,fi])-(Y[v,fi]-Y[no_idx,fi]))/
                     max(0.5*(np.linalg.norm(Y[u,fi]-Y[no_idx,fi])+np.linalg.norm(Y[v,fi]-Y[no_idx,fi])),1e-30))))
            for origin,target in ((u,v),(v,u)):
                for fold in (0,1):
                    train=np.flatnonzero(np.arange(91)%2==fold)
                    test=np.flatnonzero(np.arange(91)%2!=fold)
                    fit=fitted.get((origin,fold))
                    row=dict(protocol='C_sensor16_CROSS_SWEEP_TRANSFER',
                        restart_fraction=float(restart),frequency_hz=f,frequency_group=freq_group,
                        condition='sensor16',source_file=str(names[origin]),
                        target_file=str(names[target]),fold=fold,n_pressed=1,has_S2=False,
                        train_pairs=int(len(train)),test_pairs=int(len(test)),
                        fit_success=fit is not None)
                    if fit is not None:
                        pred=v22.predict_z(i5,A,C,fit,yp)
                        row.update(metrics(Z[target,fi,test],pred[test]))
                        dyo=Y[target,fi,test]-Y[no_idx,fi,test]
                        dyp=v22.intrinsic_y(i5,A,C,fit)[test]-v22.intrinsic_y(i5,A,C,baseline)[test]
                        row['load_deltaY_relative_L2']=float(np.linalg.norm(dyp-dyo)/max(np.linalg.norm(dyo),1e-30))
                    else:
                        row.update({k:np.nan for k in allmetric});row['load_deltaY_relative_L2']=np.nan
                    rows.append(row)
            print('  f=',round(f,1),'Hz; loaded records=',len(loaded)*2,'; S16 repeat compared',flush=True)
    df=pd.DataFrame(rows);est=pd.DataFrame(estimates)
    hold=df[df.protocol=='C_fixed_Yp_intrinsic_drift_LOADED_HELDOUT'].copy()
    transfer=df[df.protocol=='C_sensor16_CROSS_SWEEP_TRANSFER'].copy()
    if len(hold)!=len(models)*len(f_indices)*len(loaded)*2:
        raise RuntimeError('Internal test row count error')
    # Fold full COMPLEX Z consistency for each sensor/scan/freq/restart.
    gkeys=['restart_fraction','frequency_hz','sweep_index','sensor']
    folded=[]
    for key,group in est[est.type=='pressed'].groupby(gkeys):
        g=group.sort_values('fold')
        if len(g)!=2:continue
        x=complex(g.iloc[0].Z_real_ohm,g.iloc[0].Z_imag_ohm)
        y=complex(g.iloc[1].Z_real_ohm,g.iloc[1].Z_imag_ohm)
        folded.append(dict(restart_fraction=key[0],frequency_hz=key[1],
            sweep_index=key[2],sensor=key[3],
            fold_complex_Z_disagreement=float(abs(x-y)/max((abs(x)+abs(y))/2,1e-30)),
            fold_magnitude_Z_disagreement=float(abs(abs(x)-abs(y))/max((abs(x)+abs(y))/2,1e-30)),
            fold_phase_deg=float(abs(np.angle(x/y,deg=True)))))
    fmap={(r['restart_fraction'],r['frequency_hz'],r['sweep_index'],r['sensor']):r for r in folded}
    for column in ('fold_complex_Z_disagreement','fold_magnitude_Z_disagreement','fold_phase_deg'):
        est[column]=[fmap.get((row.restart_fraction,row.frequency_hz,
                              getattr(row,'sweep_index',-1),row.sensor),{}).get(column,np.nan)
                     for row in est.itertuples(index=False)]
    # S16 repeat variability of recovered sensor Z across two real sweeps.
    s16_sensor=est[(est.type=='pressed')&(est.condition=='sensor16')&(est.sensor=='S16')]
    s16_repeat_intrinsic=[]
    for key,g in s16_sensor.groupby(['restart_fraction','frequency_hz','fold']):
        if len(g)!=2:continue
        a,b=g.sort_values('source_file').itertuples(index=False)
        x=complex(a.Z_real_ohm,a.Z_imag_ohm);y=complex(b.Z_real_ohm,b.Z_imag_ohm)
        s16_repeat_intrinsic.append(dict(restart_fraction=float(key[0]),
            frequency_hz=float(key[1]),fold=int(key[2]),
            Z_complex_disagreement=float(abs(x-y)/max(0.5*(abs(x)+abs(y)),1e-30)),
            Z_magnitude_disagreement=float(abs(abs(x)-abs(y))/max(0.5*(abs(x)+abs(y)),1e-30)),
            Z_phase_disagreement_deg=float(abs(np.angle(x/y,deg=True)))))
    # Several aggregations; restart/fold evaluations are NOT independent samples.
    metrics_cols=list(allmetric)+['load_deltaY_relative_L2']
    summary=dict(experiment='V2.6_independent_22_AD2_sweeps_fixed_I5_parasitics',
      version=VERSION,status='COMPLETE' if hold.relative_Z_L2.notna().all() else 'PARTIAL_FIT_FAILURE',
      input_audit=audit,source_script_hashes={str(p):hash_file(p) for p in (EXTRACTOR,V24,v24.V22_SCRIPT)},
      i5_epoch=int(epoch),source_i5_checkpoint=str(ckpt),source_i5_checkpoint_sha256=hash_file(ckpt),
      source_i5_script=str(previous.I5_SCRIPT), source_i5_script_sha256=hash_file(previous.I5_SCRIPT),
      frequencies_hz=freqs[f_indices].tolist(),primary_frequency_min_hz=PRIMARY_F_MIN,
      strict_conditions_verified=True,loaded_scans=21,unique_loaded_conditions=20,
      loaded_rows=int(len(hold)),failed_loaded_rows=int(hold.relative_Z_L2.isna().sum()),
      S16_transfer_rows=int(len(transfer)),
      model={'physical_network':'I.5 original 14 nodes, 19 intrinsic sensor edges, 91 measured pairs',
       'parasitics':'I.5 frozen Gp,Cp, frequency RC law Gp+j*2*pi*f*Cp',
       'intrinsic_baseline':'Independent fit at each frequency from ONE separate all-91 no-load scan',
       'known_pressed_sensors':'Only those sensor G/B adjusted using 45/46 loaded train paths',
       'baseline_log_bounds':['0.1x','10x'],'loaded_component_bounds':[LOADED_MIN_FACTOR,LOADED_MAX_FACTOR],
       'five_primary_frequencies':'46.4k,100k,215k,464k,1M',
       'boundary_frequencies':'10k,21.5k',
       'folds':'parity of original 91 pair measurement ordering',
       'seeds':'3 independently initialized I.5 checkpoint results, not hardware repeats'},
      loaded_metrics_by_frequency=group_summary(hold,metrics_cols,['frequency_group','frequency_hz']),
      loaded_metrics_by_sensor_condition=group_summary(hold,metrics_cols,['condition']),
      loaded_metrics_primary=group_summary(hold[hold.frequency_group=='primary'],metrics_cols,['frequency_group']),
      loaded_metrics_primary_no_S2=group_summary(hold[(hold.frequency_group=='primary')&(~hold.has_S2)],metrics_cols,['frequency_group']),
      loaded_metrics_primary_S2_included=group_summary(hold[(hold.frequency_group=='primary')&(hold.has_S2)],metrics_cols,['frequency_group']),
      s16_repeated_raw_measurement=repeat,
      s16_repeated_intrinsic_reconstruction=s16_repeat_intrinsic,
      s16_cross_sweep_transfer=group_summary(transfer,metrics_cols,['frequency_group','frequency_hz']),
      fold_complex_stability_by_sensor=group_summary(pd.DataFrame(folded),
                   ['fold_complex_Z_disagreement','fold_magnitude_Z_disagreement','fold_phase_deg'],['sensor']),
      baseline_diagnostics=base_diagnostics,
      limitations=['Screenshot names identify files, but raw data are audited from disk; no actual raw numerical files supplied in conversation.',
       'Only one no-load sweep; baseline fit is calibration, not independently held-out validation.',
       'S16 has TWO hardware repeats. The 19 other single scans and the double scan have one scan each.',
       'Same baseline from no-load is shared between S16 repeats: between-run baseline drift cannot be isolated.',
       'Loaded train/test pairs from the same 14-node network are correlated.',
       'Known pressed sensor labels, not predicted; filename indicates intended load, not confirmed force.',
       'No direct intrinsic sensor Z reference: good 91D predictions are not proof of intrinsic truth.',
       'The 3 I.5 restarts and 2 folds do not create 6 independent hardware replicates.',
       'The fixed I.5 parasitic RC frequency law is a modeling hypothesis, not independently verified.',
       'Low-frequency measurement effects from 100kOhm reference are possible, not proven.',
       'No-load and loaded scans were taken sequentially; drift during the session may confound load.',
       'S2 physically special edge-port location; retained and reported separately.'])
    # Validate serialization in memory BEFORE writing numerical results.
    json_data=json.dumps(clean(summary),indent=2,ensure_ascii=False,allow_nan=False)
    out=OUT_ROOT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    df.to_csv(out/'V2_6_loaded_holdout.csv',index=False,float_format='%.12g',encoding='utf-8-sig')
    est.to_csv(out/'V2_6_intrinsic_estimates.csv',index=False,float_format='%.12g',encoding='utf-8-sig')
    (out/'V2_6_summary.json').write_text(json_data,encoding='utf-8')
    print('\nV2.6 COMPLETE. Files:',len(paths),' Loaded:',len(loaded),
          ' records:',len(hold),' failed:',int(hold.relative_Z_L2.isna().sum()))
    print('Output:',out)
    print('Primary mean Z relative error: ',
      float(hold[hold.frequency_group=='primary'].relative_Z_L2.mean()))
    return summary

if __name__=='__main__':
    run()
