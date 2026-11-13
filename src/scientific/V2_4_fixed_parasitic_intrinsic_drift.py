# -*- coding: utf-8 -*-
"""V2.4 - frozen I.5 parasitics, sensor-only baseline drift, f >= 10 kHz.

Spyder F5. No edits to inputs or old results. Exactly 3 files in new run folder:
  V2_4_summary.json, V2_4_per_case_frequency.csv, V2_4_intrinsic_spectra.csv

I.5 checkpoint contains 100 kHz parasitic Yp=Gp+j*omega*Cp. Gp and Cp
stay FROZEN across measurement batches AND frequencies; Yp(f) is reconstructed
from those fixed RC components. Each frequency has its own 19-sensor baseline
fit against a SEPARATE 91-pair no-load scan; loaded sensor(s) then fitted ONLY
on 45/46 training pairs and scored on complementary unseen loaded pairs.
Physical topology fixed. No frequency law is presumed for sensor intrinsic Y.

The no-load scan uses all 91 paths for calibration: loaded held-out paths are
independent of LOADED parameter fits, not a fully path-blind end-to-end test.
Known pressed sensor identities are provided. No independent sensor-Z truth.
"""
from __future__ import annotations
import hashlib, json, importlib.util, sys, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BASE=Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
V22_SCRIPT=BASE/'V2_2_two_offset_hypotheses.py'
OUTPUT_ROOT=BASE/'V2_4_fixed_parasitic_intrinsic_drift'
VERSION='1.0'
F_REF=100000.0
MIN_FREQUENCY=10000.0  # exactly 10 kHz is INcluded
RESTARTS=(0.05,0.30,0.70)
BASELINE_LOG_BOUND=float(np.log(10.0))
BASELINE_PRIOR=0.02   # same as V2.2
BASELINE_MAX_NFEV=400
# Fitting pressed-sensor parameters is regularized only by explicit multiplicative
# limits, applied equally to control and drift model. These are diagnostics,
# not physical ground truth. 100x is a declared upper bound, not tuned to holdout.
LOADED_MIN_FACTOR=0.1
LOADED_MAX_FACTOR=100.0
LOADED_STARTS=(1.0,5.0,20.0)
LOADED_MAX_NFEV=800
SOLVER_TOL=1e-8
# Large disparity between fold solutions is reported, never silently resolved.
FOLD_Z_DISAGREEMENT_THRESHOLD=0.5


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def strict_json(x):
    if isinstance(x,dict): return {str(k):strict_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [strict_json(v) for v in x]
    if isinstance(x,np.ndarray): return strict_json(x.tolist())
    if isinstance(x,np.generic): return strict_json(x.item())
    if isinstance(x,float) and not np.isfinite(x): return None
    return x


def import_v22():
    if not V22_SCRIPT.is_file():
        raise FileNotFoundError('Missing original V2.2 script: '+str(V22_SCRIPT))
    spec=importlib.util.spec_from_file_location('v24_v22_read_only',V22_SCRIPT)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m


def frozen_yp_at_frequency(yp100,f):
    """I.5 Gp, Cp frozen. Frequency dependence of Yp is NOT held constant."""
    yp100=np.asarray(yp100,dtype=np.complex128)
    return yp100.real+1j*yp100.imag*(float(f)/F_REF)


def ref_baseline_at_frequency(base100,f):
    """Unvalidated RC scaling only for optimization prior/control."""
    y=np.asarray(base100,dtype=np.complex128)
    return y.real+1j*y.imag*float(f)/F_REF


def calibrate_baseline(v22,i5,A,C,obs_y,base_prior,yp):
    """38 log parameters, only independent no-load measurement, fixed Yp."""
    g=np.asarray(base_prior.real,dtype=float)
    b=np.asarray(base_prior.imag,dtype=float)
    if np.any(g<=0) or np.any(b<=0):
        raise ValueError('Non-positive reference baseline G/B')
    scale=np.maximum(abs(obs_y),max(float(np.median(abs(obs_y)))*1e-3,1e-20))
    def decode(x):return g*np.exp(x[0::2])+1j*b*np.exp(x[1::2])
    def resid(x):
        y=v22.intrinsic_y(i5,A,C,decode(x))+yp
        err=(y-obs_y)/scale
        return np.r_[err.real/np.sqrt(91),err.imag/np.sqrt(91),
                     BASELINE_PRIOR*x/np.sqrt(38)]
    res=least_squares(resid,np.zeros(38),
         bounds=(-BASELINE_LOG_BOUND,BASELINE_LOG_BOUND),
         max_nfev=BASELINE_MAX_NFEV,ftol=SOLVER_TOL,
         xtol=SOLVER_TOL,gtol=SOLVER_TOL)
    base=decode(res.x)
    ypred=v22.intrinsic_y(i5,A,C,base)+yp
    metrics=v22.eval_metrics(1/obs_y,1/ypred)
    return base,{'success':bool(res.success),
       'optimizer_status':int(res.status),'message':str(res.message),
       'nfev':int(res.nfev),
       'n_components_at_bound':int(np.sum(abs(res.x)>=BASELINE_LOG_BOUND-1e-4)),
       'max_abs_log_change':float(np.max(abs(res.x))),
       'relative_Y_residual':float(np.linalg.norm(ypred-obs_y)/np.linalg.norm(obs_y)),
       'noload_metrics':metrics}


def fit_pressed(v22,i5,A,C,z_obs,base,yp,pressed,train):
    """Only training pair impedance/admittance. Converged starts only."""
    pressed=np.asarray(pressed,dtype=int)
    g=np.asarray(base[pressed].real,dtype=float)
    b=np.asarray(base[pressed].imag,dtype=float)
    if not len(pressed) or np.any(g<=0) or np.any(b<=0):
        raise ValueError('Invalid pressed sensor G/B')
    yobs=1/np.asarray(z_obs)[train]
    scale=max(float(np.sqrt(np.mean(np.abs(yobs)**2))),1e-20)
    def decode(x):
        ys=base.copy()
        ys[pressed]=g*np.exp(x[0::2])+1j*b*np.exp(x[1::2])
        return ys
    def residual(x):
        yp_model=v22.intrinsic_y(i5,A,C,decode(x))+yp
        d=(yp_model[train]-yobs)/scale
        return np.r_[d.real,d.imag]
    lower=np.full(2*len(pressed),np.log(LOADED_MIN_FACTOR))
    upper=np.full(2*len(pressed),np.log(LOADED_MAX_FACTOR))
    attempts=[]
    for mult in LOADED_STARTS:
        x0=np.full(2*len(pressed),np.log(mult))
        result=least_squares(residual,x0,method='trf',bounds=(lower,upper),
               max_nfev=LOADED_MAX_NFEV,xtol=SOLVER_TOL,
               ftol=SOLVER_TOL,gtol=SOLVER_TOL)
        attempts.append(result)
    good=[r for r in attempts if r.success and np.all(np.isfinite(r.fun))]
    diag={'fit_success':bool(good),
          'fit_nfev_all':';'.join(str(int(r.nfev)) for r in attempts),
          'fit_status_all':';'.join(str(int(r.status)) for r in attempts),
          'best_any_train_cost':float(min(r.fun@r.fun for r in attempts))}
    if not good:
        diag.update({'fit_status':'nonconverged','n_at_bound':None})
        return None,diag
    best=min(good,key=lambda r:float(r.fun@r.fun))
    diag.update({'fit_status':'converged','fit_nfev':int(best.nfev),
      'fit_train_cost':float(best.fun@best.fun),
      'n_at_bound':int(np.sum((abs(best.x-lower)<1e-4)|(abs(best.x-upper)<1e-4)))} )
    return decode(best.x),diag


def avg_by_group(df,metrics,group_keys):
    out=[]
    for keys,g in df.groupby(group_keys,dropna=False):
        if not isinstance(keys,tuple):keys=(keys,)
        d={k:(v.item() if isinstance(v,np.generic) else v)
           for k,v in zip(group_keys,keys)}
        d['n_rows']=len(g)
        for metric in metrics:
            values=pd.to_numeric(g[metric],errors='coerce')
            finite=values[np.isfinite(values)]
            d[metric+'_mean']=float(finite.mean()) if len(finite) else None
            d[metric+'_median']=float(finite.median()) if len(finite) else None
            d[metric+'_valid_n']=int(len(finite))
        out.append(d)
    return out


def run():
    v22=import_v22()
    previous=v22.get_previous_module()
    source_dir,npz,auditpath,audit=v22.find_verified_npz()
    with np.load(npz,allow_pickle=False) as d:
        Z=np.asarray(d['Z_measured'],dtype=np.complex128)
        freqs=np.asarray(d['frequencies_hz'],dtype=float)
        pairs=np.asarray(d['pair_ids']).astype(str)
        masks=np.asarray(d['known_pressed_mask'],dtype=bool)
        names=np.asarray(d['condition_labels']).astype(str)
        files=np.asarray(d['filenames']).astype(str)
    if Z.shape!=(5,10,91) or freqs.shape!=(10,) or masks.shape!=(5,19):
        raise ValueError('Unexpected dimensions in trusted V1V2 NPZ')
    if not np.array_equal(pairs,previous.PAIR_IDS.astype(str)):
        raise ValueError('Pair mapping mismatch relative to V1V2')
    i5=previous.import_i5()
    original_pairs=np.array(['-'.join(map(str,sorted(p))) for p in i5.PAIRS])
    if not np.array_equal(pairs,original_pairs):
        raise ValueError('I.5 canonical pair IDs / ordering mismatch')
    if not np.isfinite(Z).all() or np.any(abs(Z)<1e-30):
        raise ValueError('Missing or invalid impedance')
    no_idx=np.flatnonzero(~masks.any(axis=1))
    loaded=np.flatnonzero(masks.any(axis=1))
    if len(no_idx)!=1 or len(loaded)!=4:
        raise ValueError('Expected exactly one unloaded + four loaded scans')
    no_idx=int(no_idx[0]); f0_idx=int(np.argmin(abs(freqs-F_REF)))
    if not np.isclose(freqs[f0_idx],F_REF,rtol=1e-6):
        raise ValueError('No 100 kHz reference frequency')
    f_idx=np.flatnonzero(freqs>=MIN_FREQUENCY*(1-1e-9))
    if len(f_idx)!=7:
        raise ValueError('Expected 7 frequency points >= 10 kHz; got '+str(freqs[f_idx]))
    epoch,i5run,ckpt,A,C,models=previous.load_i5_parameters(i5)
    if len(models)!=3 or not np.allclose([x[0] for x in models],RESTARTS):
        raise ValueError('Wrong I.5 restarts')
    Y=1/Z
    rows=[];spectrum=[];fits=[]
    metrics=['relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg']
    for frac,base100,para100 in models:
        if (np.any(base100.real<=0) or np.any(base100.imag<=0) or
            np.any(para100.real<0) or np.any(para100.imag<0)):
            raise ValueError('Nonpassive I.5 checkpoint parameters')
        for fi in f_idx:
            f=float(freqs[fi]);yp=frozen_yp_at_frequency(para100,f)
            base_prior=ref_baseline_at_frequency(base100,f)
            baseline,bd=calibrate_baseline(v22,i5,A,C,Y[no_idx,fi],base_prior,yp)
            fits.append({'restart_fraction':float(frac),'frequency_hz':f,
                         'baseline_calibration':bd})
            # No-load calibration accuracy, not independent holdout
            old_noload=v22.eval_metrics(Z[no_idx,fi],v22.predict_z(i5,A,C,base_prior,yp))
            for scenario,base,nomet in (
               ('A_frozen_100k_RC_control',base_prior,old_noload),
               ('C_fixed_Yp_intrinsic_drift',baseline,bd['noload_metrics'])):
                rows.append({'protocol':scenario+'_NOLOAD_CALIBRATION_NOT_TEST',
                   'restart_fraction':float(frac),'frequency_index':int(fi),
                   'frequency_hz':f,'condition':str(names[no_idx]),
                   'source_file':str(files[no_idx]),'fold':-1,
                   'train_pairs':91,'test_pairs':0,'fit_success':bd['success'] if scenario.startswith('C_') else True,
                   'fit_status':'baseline_calibration_not_test','n_at_bound':bd['n_components_at_bound'],
                   **nomet})
            for s in range(19):
                for scenario,bs in [('A_reference_RC',base_prior),('C_fitted_intrinsic',baseline)]:
                    y=bs[s];z=1/y
                    spectrum.append({'record_type':'sensor_baseline',
                      'scenario':scenario,'restart_fraction':float(frac),
                      'frequency_hz':f,'sensor':f'S{s+1}',
                      'condition':'noloads','fold':-1,'fit_success':True,
                      'G_S':float(y.real),'B_S':float(y.imag),
                      'Z_abs_ohm':float(abs(z)),'Z_phase_deg':float(np.angle(z,deg=True)),
                      'Z_real_ohm':float(z.real),'Z_imag_ohm':float(z.imag),
                      'baseline_log_G_change':float(np.log(baseline[s].real/base_prior[s].real)) if scenario.startswith('C_') else 0.,
                      'baseline_log_B_change':float(np.log(baseline[s].imag/base_prior[s].imag)) if scenario.startswith('C_') else 0.})
            for idx in loaded:
                pressed=np.flatnonzero(masks[idx])
                for fold in (0,1):
                    train=np.flatnonzero(np.arange(91)%2==fold)
                    test=np.flatnonzero(np.arange(91)%2!=fold)
                    fitted_by_scenario={}
                    for scenario,base in [('A_frozen_100k_RC_control',base_prior),
                                          ('C_fixed_Yp_intrinsic_drift',baseline)]:
                        rec={'protocol':scenario+'_LOADED_HELDOUT',
                             'restart_fraction':float(frac),'frequency_index':int(fi),
                             'frequency_hz':f,'condition':str(names[idx]),
                             'source_file':str(files[idx]),'fold':fold,
                             'train_pairs':int(len(train)),'test_pairs':int(len(test)),
                             'baseline_fit_success':bool(bd['success']) if scenario.startswith('C_') else True,
                             'baseline_n_at_bound':int(bd['n_components_at_bound']) if scenario.startswith('C_') else 0}
                        if scenario.startswith('C_') and not bd['success']:
                            rec.update({'fit_success':False,'fit_status':'baseline_nonconvergence',
                               'n_at_bound':None,**{m:np.nan for m in metrics}})
                            rows.append(rec)
                            continue
                        fitted,fd=fit_pressed(v22,i5,A,C,Z[idx,fi],base,yp,pressed,train)
                        rec.update(fd)
                        if fitted is not None:
                            zp=v22.predict_z(i5,A,C,fitted,yp)
                            rec.update(v22.eval_metrics(Z[idx,fi,test],zp[test]))
                            # Relative norm of load-induced changes in Y, cancellation
                            # of an unchanging nuisance path Yp in each scenario.
                            dyobs=Y[idx,fi,test]-Y[no_idx,fi,test]
                            dypred=v22.intrinsic_y(i5,A,C,fitted)[test]-v22.intrinsic_y(i5,A,C,base)[test]
                            rec['load_deltaY_relative_L2']=float(np.linalg.norm(dypred-dyobs)/max(np.linalg.norm(dyobs),1e-30))
                            for sensor in pressed:
                                yi=fitted[sensor];zi=1/yi
                                spectrum.append({'record_type':'loaded_sensor','scenario':scenario,
                                 'restart_fraction':float(frac),'frequency_hz':f,
                                 'sensor':f'S{sensor+1}','condition':str(names[idx]),'fold':fold,
                                 'fit_success':True,'G_S':float(yi.real),'B_S':float(yi.imag),
                                 'Z_abs_ohm':float(abs(zi)),'Z_phase_deg':float(np.angle(zi,deg=True)),
                                 'Z_real_ohm':float(zi.real),'Z_imag_ohm':float(zi.imag),
                                 'n_at_bound':fd['n_at_bound']})
                        else:
                            rec.update({m:np.nan for m in metrics})
                            rec['load_deltaY_relative_L2']=np.nan
                        rows.append(rec)
    table=pd.DataFrame(rows)
    spec=pd.DataFrame(spectrum)
    test=table[table['protocol'].str.endswith('_LOADED_HELDOUT')].copy()
    if test.empty: raise RuntimeError('No load test records')
    # Flag divergent fitted sensor spectra between the two complementary pair splits.
    unstable=[]
    loaded_spec=spec[spec.record_type=='loaded_sensor']
    for key,g in loaded_spec.groupby(['scenario','restart_fraction','frequency_hz','condition','sensor']):
        if len(g)!=2:continue
        values=g.sort_values('fold')['Z_abs_ohm'].to_numpy(dtype=float)
        disagreement=float(abs(values[0]-values[1])/max(np.mean(values),1e-30))
        unstable.append({'scenario':key[0],'restart_fraction':key[1],
            'frequency_hz':key[2],'condition':key[3], 'sensor':key[4],
            'fold_magnitude_disagreement':disagreement,
            'flag_above_50pct':bool(disagreement>FOLD_Z_DISAGREEMENT_THRESHOLD)})
    # Store fold consistency as columns of sensor spectra, rather than fourth file.
    disagreements={(r['scenario'],r['restart_fraction'],r['frequency_hz'],
        r['condition'],r['sensor']):r['fold_magnitude_disagreement'] for r in unstable}
    spec['fold_Z_abs_disagreement']= [disagreements.get((r.scenario,r.restart_fraction,
        r.frequency_hz,r.condition,r.sensor),np.nan) for r in spec.itertuples()]
    loaded_metrics=metrics+['load_deltaY_relative_L2']
    report={'experiment':'V2.4_frozen_parasitics_intrinsic_drift_10k_to_1MHz',
      'version':VERSION,
      'status':'COMPLETE' if test['relative_Z_L2'].notna().all() else 'PARTIAL_FIT_FAILURE',
      'source_v1v2_npz':str(npz),'source_npz_sha256':sha256(npz),
      'source_v1v2_audit':str(auditpath),'source_audit_sha256':sha256(auditpath),
      'source_i5_checkpoint':str(ckpt),'source_checkpoint_sha256':sha256(ckpt),
      'source_v22_script_sha256':sha256(V22_SCRIPT),
      'i5_epoch':int(epoch), 'restarts':list(RESTARTS),
      'frequency_min_inclusive_hz':MIN_FREQUENCY,
      'frequencies_hz':freqs[f_idx].tolist(),
      'conditions':names.tolist(),
      'pair_alignment':'all 91 physical undirected pair IDs checked in original order',
      'assumptions':{'Yp_frequency_law':'frozen I5 Gp and Cp: Yp(f)=Gp+j 2 pi f Cp',
           'sensor_frequency_law':'NONE: every frequency independently calibrates baseline from new no-load',
           'old_sensor_baseline_prior':'I5 G+j B*(f/100k), ONLY an RC-scaled initialization and control',
           'baseline_log_bounds_multiplier':[0.1,10.0],
           'baseline_log_prior_weight':BASELINE_PRIOR,
           'loaded_fitted_components_bounds_multiplier':[LOADED_MIN_FACTOR,LOADED_MAX_FACTOR],
           'loaded_starts':list(LOADED_STARTS),
           'regularization_not_tuned_on_heldout':True},
      'evaluation':{'separate_noload_all_91_used_for_baseline_calibration':True,
           'loaded_sensor_admittance_fit_on_45_or_46_pairs':True,
           'opposite_45_or_46_loaded_pairs_held_out':True,
           'known_press_support':True,
           'number_of_new_independent_loaded_scans':4,
           'n_loaded_evaluation_rows':int(len(test)),
           'n_failed_loaded_rows':int(test['relative_Z_L2'].isna().sum())},
      'loaded_by_scenario':avg_by_group(test,loaded_metrics,['protocol']),
      'loaded_by_frequency':avg_by_group(test,loaded_metrics,['protocol','frequency_hz']),
      'loaded_by_condition':avg_by_group(test,loaded_metrics,['protocol','condition']),
      'noload_calibration_metrics':avg_by_group(table[table['fold']==-1],metrics,['protocol','frequency_hz']),
      'baseline_calibration_diagnostics':fits,
      'unstable_loaded_estimates':unstable,
      'n_fold_disagreements_over_50pct':int(sum(x['flag_above_50pct'] for x in unstable)),
      'limitations':['No-load 91 pairs calibrate each frequency: not zero-shot frequency prediction.',
           'Frozen I5 parasitic Gp/Cp across frequency is an UNVALIDATED RC assumption.',
           '10 kHz is included as boundary diagnostic; possible 100kOhm AD2 reference mismatch.',
           'Old I5 baselines are RC-extrapolated only as a prior; sensor f dispersion not presumed RC.',
           'Known pressed identities, not predicted; heldout physical pairs are correlated.',
           'No reference truth for intrinsic impedances; a good fit does not establish correct recovery.',
           'Bounds may distort inferred sensor Z; bound hits and fold disagreement must be examined.',
           'Different frequency calibration does NOT demonstrate cross-frequency generalization.']}
    out=OUTPUT_ROOT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    # Validate all serializations BEFORE writing results; no half-written outputs.
    json_text=json.dumps(strict_json(report),indent=2,ensure_ascii=False,allow_nan=False)
    table.to_csv(out/'V2_4_per_case_frequency.csv',index=False,float_format='%.12g')
    spec.to_csv(out/'V2_4_intrinsic_spectra.csv',index=False,float_format='%.12g')
    (out/'V2_4_summary.json').write_text(json_text,encoding='utf-8')
    print('\nV2.4 DONE:',out)
    print('Number of loaded evaluations:',len(test),'failed:',report['evaluation']['n_failed_loaded_rows'])
    for row in report['loaded_by_scenario']:
        print(row['protocol'], 'mean rel Z=',row['relative_Z_L2_mean'],
              'median rel Z=',row['relative_Z_L2_median'])
    print('Output exactly 3 files. Original inputs read-only.')


if __name__=='__main__':
    run()
