# -*- coding: utf-8 -*-
"""V2.2: two exclusive explanations of 100 kHz no-load transfer shift.

Spyder F5; reads prior audited V1+V2 five-scan NPZ + original I.5.
Three scenarios, identical deterministic held-out-pair evaluation on 4 loads:
 A) original frozen I.5 (control),
 B) all no-load 91-path admittance shift attributed to PARASITICS,
 C) parasitics FROZEN; change 19 baseline sensor admittances to fit no-load.

Important: B is an *effective* 91-path correction, NOT independently measured
parasitics. C is an overdetermined 19-sensor constrained inverse fit, with an
explicit (nonzero possible) residual; neither decomposition is identified by
one 91-path no-load observation alone. Calibration uses ONLY new no-load.

Outputs exactly 3 files in a fresh folder: JSON summary, CSV per-case/fold,
CSV per-path & per-sensor diagnostics. Original files are never modified.
"""
from __future__ import annotations
import importlib.util
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import torch

BASE = Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
V1V2_ROOT = BASE / 'V1_V2_new5_independent_validation'
V1V2_SCRIPT = BASE / 'V1_V2_new5_100k_validation.py'
OUTPUT_ROOT = BASE / 'V2_2_two_offset_hypotheses'
VERSION = '1.0'
F_REF = 100_000.0
RESTARTS = (0.05, 0.30, 0.70)
# Fixed, predeclared physical/regularization constraints (no tuning on loads).
BASE_LOG_BOUND = float(np.log(10.0))  # at most 10x or 0.1x for base G and B
BASE_PRIOR_WEIGHT = 0.02  # fixed log-space regularization scaled per data residual
START_MULTIPLIERS = (1.0, 5.0, 20.0)  # same as original V2


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def get_previous_module():
    if not V1V2_SCRIPT.is_file():
        raise FileNotFoundError(f'Missing prior V1V2 script: {V1V2_SCRIPT}')
    spec = importlib.util.spec_from_file_location('v22_previous_validated', V1V2_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def find_verified_npz():
    candidates=[]
    for folder in V1V2_ROOT.glob('run_*'):
        js=folder/'V1V2_audit_and_summary.json'
        npz=folder/'V1V2_measurements.npz'
        if not js.is_file() or not npz.is_file():
            continue
        audit=json.loads(js.read_text(encoding='utf-8'))
        if audit.get('status')!='V1_PASSED_V2_COMPLETED' or not audit.get('accepted'):
            continue
        if audit.get('npz_sha256')!=sha256(npz):
            continue
        candidates.append((folder,npz,js,audit))
    if not candidates:
        raise FileNotFoundError('No complete hash-verified V1+V2 five-scan NPZ found')
    candidates.sort(key=lambda x:x[0].name)
    return candidates[-1]


def load_new_data(mod, npz, audit):
    with np.load(npz,allow_pickle=False) as data:
        Z=np.asarray(data['Z_measured'],dtype=np.complex128)
        freqs=np.asarray(data['frequencies_hz'],dtype=float)
        pairids=np.asarray(data['pair_ids']).astype(str)
        masks=np.asarray(data['known_pressed_mask'],dtype=bool)
        conds=np.asarray(data['condition_labels']).astype(str)
        filenames=np.asarray(data['filenames']).astype(str)
    if Z.shape!=(5,10,91) or masks.shape!=(5,19):
        raise ValueError('Unexpected five-case NPZ shape')
    if not np.array_equal(pairids,mod.PAIR_IDS.astype(str)):
        raise ValueError('V1V2 NPZ pair ordering differs from prior validated canonical I.5 order')
    fi=int(np.argmin(abs(freqs-F_REF)))
    if not np.isclose(freqs[fi],F_REF,rtol=1e-6):
        raise ValueError('100kHz slice missing')
    if not np.all(np.isfinite(Z)) or np.any(abs(Z)==0):
        raise ValueError('Invalid measured impedance')
    noload=np.flatnonzero(~masks.any(axis=1))
    if len(noload)!=1 or np.count_nonzero(masks.any(axis=1))!=4:
        raise ValueError('Expected EXACTLY one no-load, four loaded cases')
    return Z[:,fi,:],masks,conds,filenames,int(noload[0]),pairids,float(freqs[fi])


def intrinsic_y(mod,A,C,base):
    """Physical 19-sensor -> 91-path intrinsic admittance, no parasitics."""
    with torch.no_grad():
        yy=torch.as_tensor(np.asarray(base)[None,:],dtype=torch.complex128)
        _,intr=mod.forward_y_sensor_to_intrinsic(A,C,yy)
        result=intr[0].detach().cpu().numpy()
    if result.shape!=(91,) or not np.isfinite(result).all():
        raise ValueError('Physical model invalid')
    return result


def predict_z(mod,A,C,ys,yp):
    yy=intrinsic_y(mod,A,C,ys)+yp
    if np.any(abs(yy)<1e-30):
        raise ValueError('Zero predicted Y')
    return 1/yy


def fit_baseline_drift(mod,A,C,obs_y_noload,old_base,old_para):
    """H2: 38 positive G/B baseline unknowns, 91 complex no-load constraints.

    Uniform relative-admittance residual + small pre-fixed log drift prior.
    No load-case information used, and no per-path free correction allowed.
    """
    g=np.real(old_base).copy()
    b=np.imag(old_base).copy()
    if np.any(g<=0) or np.any(b<=0):
        raise ValueError('I.5 baseline violates positive conductance/susceptance')
    scale=np.maximum(abs(obs_y_noload),np.median(abs(obs_y_noload))*1e-3)
    def expand(x):
        return g*np.exp(x[::2])+1j*b*np.exp(x[1::2])
    def res(x):
        pred=intrinsic_y(mod,A,C,expand(x))+old_para
        error=(pred-obs_y_noload)/scale
        return np.r_[error.real/np.sqrt(91),error.imag/np.sqrt(91),
                     BASE_PRIOR_WEIGHT*x/np.sqrt(38)]
    x0=np.zeros(38)
    result=least_squares(res,x0,bounds=(-BASE_LOG_BOUND,BASE_LOG_BOUND),
                         max_nfev=350,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    baseline=expand(result.x)
    achieved=intrinsic_y(mod,A,C,baseline)+old_para
    remaining=obs_y_noload-achieved
    return baseline, {
        'optimizer_success':bool(result.success),'optimizer_status':int(result.status),
        'optimizer_message':str(result.message),'nfev':int(result.nfev),
        'n_at_bound':int(np.sum(abs(result.x)>BASE_LOG_BOUND-1e-4)),
        'max_absolute_log_drift':float(np.max(abs(result.x))),
        'noload_remaining_Y_rel_L2':float(np.linalg.norm(remaining)/np.linalg.norm(obs_y_noload)),
        'mean_abs_relative_Y_error':float(np.mean(abs(remaining)/abs(obs_y_noload))),
        'baseline_delta_Y_19_real':(baseline-old_base).real.tolist(),
        'baseline_delta_Y_19_imag':(baseline-old_base).imag.tolist(),
        'log_drift_38':result.x.tolist(),
    }


def fit_loaded_sensors(mod,A,C,measured_z,base,para,pressed,train):
    """Fit ONLY known pressed G/B on train node pairs; identical in A/B/C."""
    pressed=np.asarray(pressed,dtype=int)
    g0=np.asarray(base[pressed].real)
    b0=np.asarray(base[pressed].imag)
    if np.any(g0<=0) or np.any(b0<=0):
        raise ValueError('Nonpositive sensor base G/B')
    target=1/measured_z[train]
    scale=max(float(np.sqrt(np.mean(abs(target)**2))),1e-20)
    def expand(x):
        y=base.copy()
        y[pressed]=g0*np.exp(x[0::2])+1j*b0*np.exp(x[1::2])
        return y
    def residual(x):
        ypred=intrinsic_y(mod,A,C,expand(x))+para
        d=(ypred[train]-target)/scale
        return np.r_[d.real,d.imag]
    fitted=[]
    for multiple in START_MULTIPLIERS:
        x0=np.full(2*len(pressed),np.log(multiple))
        result=least_squares(residual,x0,method='trf',bounds=(-12.,12.),
                             max_nfev=250,xtol=1e-11,ftol=1e-11,gtol=1e-11)
        fitted.append(result)
    best=min(fitted,key=lambda r:float(r.fun@r.fun))
    return expand(best.x),{'success':bool(best.success),
                           'nfev':int(best.nfev),
                           'train_cost':float(best.fun@best.fun)}


def eval_metrics(observed,predicted):
    zo=np.asarray(observed);zp=np.asarray(predicted)
    yo=1/zo;yp=1/zp
    return {
        'relative_Z_L2':float(np.linalg.norm(zp-zo)/np.linalg.norm(zo)),
        'relative_Y_L2':float(np.linalg.norm(yp-yo)/np.linalg.norm(yo)),
        'magnitude_MAPE':float(np.mean(abs(abs(zp)-abs(zo))/abs(zo))),
        'phase_MAE_deg':float(np.mean(abs(np.angle(zp/zo,deg=True)))),
        'pred_mag_below_obs_fraction':float(np.mean(abs(zp)<abs(zo)))
    }


def run():
    mod=get_previous_module()
    oldfolder,npz,js,audit=find_verified_npz()
    obs,masks,conds,files,noidx,pairs,frequency=load_new_data(mod,npz,audit)
    # Strong source alignment gate: loaded I.5 original pair IDs and checkpoint signatures.
    i5=mod.import_i5()
    original_ids=np.array(['-'.join(map(str,sorted(p))) for p in i5.PAIRS])
    if not np.array_equal(pairs,original_ids):
        raise ValueError('I.5 pair ID / order mismatch: must fix mapping before V2.2')
    epoch,i5run,checkpoint,A,C,models=mod.load_i5_parameters(i5)
    if len(models)!=3 or not np.allclose([m[0] for m in models],RESTARTS):
        raise ValueError('Unexpected I.5 model restarts')
    y_noload=1/obs[noidx]
    loaded_indices=np.flatnonzero(masks.any(axis=1))
    allpairs=np.arange(91)
    rows=[];drows=[];summ=[]
    for frac,original_base,original_para in models:
        y_pred_noload=intrinsic_y(i5,A,C,original_base)+original_para
        dy=y_noload-y_pred_noload
        # H1: exact 91-dimensional effective parasitic shift, including any
        # nonphysical (negative G/C) values; report, NEVER silently clip.
        shifted_para=original_para+dy
        # H2: fixed parasitics; a physically constrained, regularized 19-sensor
        # baseline drift fit using ONLY no-load measurements.
        changed_base,drift=fit_baseline_drift(i5,A,C,y_noload,original_base,original_para)
        scenarios={
           'A_original':(original_base,original_para),
           'B_parasitic_only':(original_base,shifted_para),
           'C_intrinsic_drift_only':(changed_base,original_para),
        }
        nonphysical=int(np.count_nonzero((shifted_para.real<0)|(shifted_para.imag<0)))
        initial=eval_metrics(obs[noidx],1/y_pred_noload)
        summ.append({
            'restart_fraction':float(frac),
            'before_noload_metrics':initial,
            'H1_no_load_Y_residual_relative_L2':float(np.linalg.norm((intrinsic_y(i5,A,C,original_base)+shifted_para)-y_noload)/np.linalg.norm(y_noload)),
            'H1_delta_Y_91_relative_to_measured_Y':float(np.linalg.norm(dy)/np.linalg.norm(y_noload)),
            'H1_negative_effective_parasitic_paths':nonphysical,
            'H1_negative_G_paths':int(np.sum(shifted_para.real<0)),
            'H1_negative_C_paths':int(np.sum(shifted_para.imag<0)),
            'H2_baseline_fit':drift,
            'H2_no_load_metrics':eval_metrics(obs[noidx],predict_z(i5,A,C,changed_base,original_para)),
            'H1_no_load_metrics':eval_metrics(obs[noidx],predict_z(i5,A,C,original_base,shifted_para)),
        })
        for k in range(91):
            drows.append({
                'kind':'pair','restart_fraction':float(frac),
                'pair_index_1based':k+1,'pair':pairs[k],
                'measured_noload_Y_real_S':float(y_noload[k].real),
                'measured_noload_Y_imag_S':float(y_noload[k].imag),
                'I5_predicted_noload_Y_real_S':float(y_pred_noload[k].real),
                'I5_predicted_noload_Y_imag_S':float(y_pred_noload[k].imag),
                'offset_Y_real_S':float(dy[k].real),'offset_Y_imag_S':float(dy[k].imag),
                'H1_I5_Gp_S':float(original_para[k].real),
                'H1_I5_Cp_F':float(original_para[k].imag/(2*np.pi*frequency)),
                'H1_effective_Gp_S':float(shifted_para[k].real),
                'H1_effective_Cp_F':float(shifted_para[k].imag/(2*np.pi*frequency)),
                'H2_fixed_Gp_S':float(original_para[k].real),
                'H2_fixed_Cp_F':float(original_para[k].imag/(2*np.pi*frequency)),
                'H2_unexplained_Y_real_S':float((y_noload-(intrinsic_y(i5,A,C,changed_base)+original_para))[k].real),
                'H2_unexplained_Y_imag_S':float((y_noload-(intrinsic_y(i5,A,C,changed_base)+original_para))[k].imag)
            })
        for s in range(19):
            drows.append({
                'kind':'sensor','restart_fraction':float(frac),
                'sensor_index_1based':s+1,
                'I5_baseline_Y_real_S':float(original_base[s].real),
                'I5_baseline_Y_imag_S':float(original_base[s].imag),
                'H2_baseline_Y_real_S':float(changed_base[s].real),
                'H2_baseline_Y_imag_S':float(changed_base[s].imag),
                'H2_baseline_Z_magnitude_ohm':float(abs(1/changed_base[s])),
                'H2_base_G_ratio':float(changed_base[s].real/original_base[s].real),
                'H2_base_B_ratio':float(changed_base[s].imag/original_base[s].imag),
            })
        for scenario,(base,para) in scenarios.items():
            for r in loaded_indices:
                pressed=np.flatnonzero(masks[r])
                for fold in (0,1):
                    train=allpairs[allpairs%2==fold]
                    test=allpairs[allpairs%2!=fold]
                    estimate,fit=fit_loaded_sensors(i5,A,C,obs[r],base,para,pressed,train)
                    prediction=predict_z(i5,A,C,estimate,para)
                    record={
                        'scenario':scenario,'restart_fraction':float(frac),
                        'condition':conds[r],'source_file':files[r],
                        'sweep_index':int(r),'fold':int(fold),
                        'train_pair_count':len(train),'test_pair_count':len(test),
                        'fit_success':fit['success'],'fit_nfev':fit['nfev'],
                        'fit_train_squared_cost':fit['train_cost'],
                    }
                    record.update(eval_metrics(obs[r,test],prediction[test]))
                    for s in pressed:
                        z=1/estimate[s]
                        record[f'S{s+1}_fit_Z_real_ohm']=float(z.real)
                        record[f'S{s+1}_fit_Z_imag_ohm']=float(z.imag)
                    rows.append(record)
    df=pd.DataFrame(rows)
    if not df['fit_success'].all():
        raise ValueError('At least one load fit unsuccessful. No incomplete results exported.')
    summary_table=(df.groupby('scenario')[['relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg']]
                   .agg(['mean','median']))
    base_means=df.groupby('scenario')[['relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg']].mean()
    # Each line corresponds to the SAME new loaded case/fold/I5 restart for A/B/C.
    per_cond=df.groupby(['scenario','condition'])[['relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg']].mean().reset_index()
    report={
        'experiment':'V2.2_two_exclusive_common_offset_hypotheses','version':VERSION,
        'scientific_question':'Does attributing the new no-load shift to parasitics OR sensor baseline drift improve held-out loaded-pair predictions?',
        'source_v1v2_dir':str(oldfolder),'source_v1v2_npz_sha256':sha256(npz),
        'source_v1v2_json_sha256':sha256(js),'source_v1v2_script_sha256':sha256(V1V2_SCRIPT),
        'source_i5_dir':str(i5run),'source_i5_checkpoint_sha256':sha256(checkpoint),
        'frequency_hz':frequency,'n_new_scans':5,'n_noload':1,'n_load_cases':4,
        'labels':conds.tolist(),'noload_file':files[noidx],
        'i5_checkpoint_epoch':int(epoch),'i5_restarts':list(RESTARTS),
        'pair_alignment':'exact 91 pair ID and ordered equality checked against I.5',
        'hypotheses':{
          'A_original':'Original I.5 baseline and parasitics frozen; fit only known pressed sensors on train pairs.',
          'B_parasitic_only':'Per-path effective deltaY = Y_new_noload - Y_I5_pred_noload; add it to I.5 parasitics, leave sensor baseline frozen. No-load matches exactly by construction; deltaY not proven parasitic.',
          'C_intrinsic_drift_only':'91 parasitic paths frozen, 19 baseline complex admittances fitted from new no-load under positivity, bounded log shift and declared regularization. Residual not forced to zero.'
        },
        'H2_constraints':{'max_G_or_B_factor':10.0,'min_G_or_B_factor':0.1,
                          'baseline_log_prior_weight':BASE_PRIOR_WEIGHT,
                          'baseline_optimizer_max_nfev':350},
        'evaluation':{'protocol':'Even/odd 91 node pair holdout, reverse folds. Calibration uses ONLY separate new no-load; loaded pressed sensor G/B fitted on train pairs, evaluated on unseen held-out pairs. Known pressed support supplied.',
                      'n_rows_per_scenario':int(len(df)//3),
                      'n_rows_total':int(len(df))},
        'noload_per_I5_restart':summ,
        'heldout_group_mean':base_means.reset_index().to_dict('records'),
        'heldout_by_condition_mean':per_cond.to_dict('records'),
        'limitations':[
           'B and C are assumption-dependent decompositions and not independent measurements of true parasitic vs intrinsic components.',
           'H1 exact no-load fit is guaranteed by 91 free per-path offsets; it is not a validation result.',
           'H2 constraints/regularization affect the fitted baseline; inspect residuals and bound hits.',
           'No-load is calibration data for B/C, NOT an independent holdout.',
           'New loaded evaluation is conditional on known pressed sensor identities and fitting their admittances on training pairs.',
           'Heldout node pairs are correlated by the same physical network.',
           'Only new 5 sweeps at 100 kHz analyzed; historical 804 no-load scans are not independent calibration for this batch.',
           'Output parasitics use Yp=Gp+j omega Cp at 100 kHz only, not frequency-generalized.'
        ]
    }
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    out=OUTPUT_ROOT/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
    out.mkdir(exist_ok=False)
    p_rows=out/'V2_2_heldout_per_case.csv'
    p_paths=out/'V2_2_parasitic_and_baseline_diagnostics.csv'
    p_summary=out/'V2_2_summary.json'
    df.to_csv(p_rows,index=False,encoding='utf-8-sig')
    pd.DataFrame(drows).to_csv(p_paths,index=False,encoding='utf-8-sig')
    report['outputs_sha256']={p.name:sha256(p) for p in (p_rows,p_paths)}
    report['out_dir']=str(out)
    p_summary.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nV2.2 COMPLETE:',out)
    print('Mean holdout relative Z and Y error:')
    print(base_means[['relative_Z_L2','relative_Y_L2']].to_string())
    for row in summ:
        print('Restart:',row['restart_fraction'],
              'H1 negative effective parasitic paths:',row['H1_negative_effective_parasitic_paths'],
              'H2 no-load residual:',row['H2_baseline_fit']['noload_remaining_Y_rel_L2'])
    return report

if __name__=='__main__':
    run()
