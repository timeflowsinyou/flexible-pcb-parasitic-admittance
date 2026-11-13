# -*- coding: utf-8 -*-
"""I.5 S2 prior sensitivity: E1 initialization, E2 weak, E3 medium prior.

SPYDER F5, no command-line arguments.

Re-runs the SAME original I.5 644-sample, 91-pair joint inverse model.
- E1: only the initial unloaded S2 G and B are changed to the median of
      the ORIGINAL I.5 checkpoint's other 18 unloaded sensor G and B.
- E2: identical initialization and a weak logarithmic S2 prior (1e-4).
- E3: identical initialization and a medium logarithmic S2 prior (1e-3).

The initial target and prior are frozen from the ORIGINAL checkpoint, by
restart fraction (0.05, 0.30, 0.70), to prevent circular self-regularization.
The 91 parasitic admittances ARE RE-ESTIMATED in each new I.5 run.
No original data, checkpoint, historical results, or prior estimates are edited.

The script saves restart-safe scenario checkpoints, raw unregularized loss,
S2 baseline, all 91 parasitic RC coefficients, 19 unloaded branch admittances,
and cross-experiment comparison. It optionally evaluates the resulting 9 models
on the independent 22-sweep set at 100 kHz, using the original V2.4 fitting
rules (new unloaded calibration followed by 45/46 pair held-out evaluation).

CAUTION: E1/E2/E3 are sensitivity analyses. Weak and medium strengths are
prespecified exploratory values, not estimated from held-out validation.
Using reference medians derived from the original I.5 checkpoint does not
constitute independent sensor ground truth.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
SOURCE_SCRIPT = BASE / 'experiment_I5_checkpointed_convergence.py'
EXPECTED_I5_SCRIPT_SHA256 = '2a05426478cc347e7c58cd8308b064d72ae630223a240bb334613ec197af1c08'
SOURCE_DIR = BASE / 'real91_extract' / 'run_20261103T145251_758871Z'
ORIGINAL_CHECKPOINT = (BASE / 'experiment_I5_checkpointed_convergence' /
                       'run_20261104T090803_878019Z' / 'checkpoint.pt')
EXPECTED_ORIGINAL_SHA256 = '1840612c1acb81c1e4bb97df3cfb4315a76deb587a991627ee46563d91cbe366'
V26_SCRIPT = BASE / 'V2_6_validate_22_sweeps_fixed_parasitics.py'
OUT_ROOT = BASE / 'experiment_I5_S2_E1_E2_E3'
VERSION = '1.0.0'
FRACTIONS = (0.05, 0.30, 0.70)
# Specify a fixed epoch budget to compare against the archived I.5 checkpoint
# (epoch 5200). This is NOT an assertion of optimization convergence.
MAX_EPOCHS = 5200
CHECK_EVERY = 200
# Lambda multiplies the sum of the two S2 log deviations squared.
SCENARIOS = [('E1_initialization_only', 0.0),
             ('E2_weak_S2_prior', 1e-4),
             ('E3_medium_S2_prior', 1e-3)]
# Run independent 100k held-out evaluation using RAW 22 sweeps if present.
RUN_EXTERNAL_100KHZ_VALIDATION = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def import_py(path,name):
    if not path.is_file():raise FileNotFoundError(f'Missing script: {path}')
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod


def atomic_torch_save(item,path):
    temp=path.with_suffix('.temporary')
    torch.save(item,temp)
    os.replace(temp,path)


def atomic_json_save(obj,path):
    tmp=path.with_suffix('.temporary')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)


def floatnum(x):return float(np.asarray(x))


def find_source():
    if sha256(SOURCE_SCRIPT) != EXPECTED_I5_SCRIPT_SHA256:
        raise ValueError('Original I.5 source script differs from the verified version')
    i5=import_py(SOURCE_SCRIPT,'s2_prior_original_i5')
    required=[SOURCE_DIR/'real_91pair_measurements.npz',
              SOURCE_DIR/'sample_manifest.csv',
              SOURCE_DIR/'extraction_audit.json',ORIGINAL_CHECKPOINT]
    for f in required:
        if not f.is_file():raise FileNotFoundError('Required original I.5 input missing: '+str(f))
    if sha256(ORIGINAL_CHECKPOINT)!=EXPECTED_ORIGINAL_SHA256:
        raise ValueError('Original checkpoint SHA256 differs from the V2.6 verified I.5 checkpoint')
    z,y,manifest,mask=i5.load_inputs(required[0],required[1])
    if y.shape!=(644,91) or mask.shape!=(644,19) or mask.sum()!=1801:
        raise ValueError('I.5 644-sample audit failed')
    A,C=i5.incidence_matrices(DEVICE)
    # The checkpoint is created by the user's original I.5 program. Do not
    # deserialize checkpoints obtained from unknown/untrusted sources.
    old=torch.load(ORIGINAL_CHECKPOINT,map_location='cpu',weights_only=False)
    if int(old['epoch'])!=5200 or len(old['models'])!=3:
        raise ValueError('Unexpected original 5200-epoch state')
    priors={}
    original=[]
    for fraction,state in zip(FRACTIONS,old['models']):
        # base parameters store log(G/1e-6) and log(B/1e-6)
        bg=state['base_g_log'].cpu().numpy().astype(float)
        bb=state['base_b_log'].cpu().numpy().astype(float)
        if bg.shape!=(19,) or bb.shape!=(19,):raise ValueError('Wrong original branch dimensions')
        g=np.exp(bg)*1e-6;b=np.exp(bb)*1e-6
        idx=[j for j in range(19) if j!=1]
        gref=float(np.median(g[idx]));bref=float(np.median(b[idx]))
        priors[fraction]={'G_ref_S':gref,'B_ref_S':bref,
                          'original_S2_G_S':float(g[1]),'original_S2_B_S':float(b[1])}
        yp=(np.exp(state['gpara_log'].cpu().numpy())+
            1j*np.exp(state['bpara_log'].cpu().numpy()))*1e-6
        original.append({'scenario':'E0_original_I5','fraction':fraction,
                         'S2_Z_abs_ohm':float(abs(1/(g[1]+1j*b[1]))),
                         'S2_G_S':float(g[1]),'S2_B_S':float(b[1]),
                         'reference_G_S':gref,'reference_B_S':bref,
                         'original_Yp':yp})
    signature={'script_version':VERSION,
               'source_script_sha256':sha256(SOURCE_SCRIPT),
               'original_checkpoint_sha256':sha256(ORIGINAL_CHECKPOINT),
               'raw_sha256':{p.name:sha256(p) for p in required[:3]},
               'n_samples':len(y),'n_pairs':91,'n_active':int(mask.sum()),
               'max_epochs':MAX_EPOCHS,'check_every':CHECK_EVERY,
               'learning_rate':i5.LR,'seed':i5.SENSOR_INIT_SEED,
               'scenarios':SCENARIOS,'fractions':FRACTIONS}
    return i5,z,y,manifest,mask,A,C,priors,original,signature


def run_one(i5, y, mask, A, C, priors, signature, scenario, lam, frac, target, denom):
    folder=OUT_ROOT/scenario/f'init_{int(round(frac*100)):02d}pct'
    folder.mkdir(parents=True,exist_ok=True)
    resume=folder/'checkpoint.pt'
    p=priors[frac]
    model,opt=i5.make_model_state(y,mask,frac,DEVICE)
    # Only S2 initial baseline is changed. All other model parameters match
    # the original initialization. The prior remains the ORIGINAL frozen median.
    with torch.no_grad():
        model.base_g_log[1]=np.log(p['G_ref_S']/1e-6)
        model.base_b_log[1]=np.log(p['B_ref_S']/1e-6)
    epoch0=0;history=[]
    if resume.exists():
        # Only checkpoint files previously generated in this owned output folder.
        state=torch.load(resume,map_location='cpu',weights_only=False)
        expected_sig={**signature,'scenario':scenario,'fraction':frac,'lambda':lam}
        if state['signature']!=expected_sig:
            raise ValueError('RESUME SIGNATURE MISMATCH: '+str(resume))
        epoch0=int(state['epoch']);history=state['history']
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        for v in opt.state.values():
            for key,val in v.items():
                if isinstance(val,torch.Tensor):v[key]=val.to(DEVICE)
        print(f'Resumed {scenario} f={frac} epoch={epoch0}',flush=True)
    prior_g=torch.tensor(np.log(p['G_ref_S']/1e-6),dtype=torch.float64,device=DEVICE)
    prior_b=torch.tensor(np.log(p['B_ref_S']/1e-6),dtype=torch.float64,device=DEVICE)
    def objective():
        yp,ys,zi,yi,yhat=model(A,C)
        fit=torch.mean(torch.abs(yhat-target)**2)/denom
        prior=(model.base_g_log[1]-prior_g)**2+(model.base_b_log[1]-prior_b)**2
        return fit+lam*prior,fit,prior
    for epoch in range(epoch0+1,MAX_EPOCHS+1):
        opt.zero_grad(set_to_none=True)
        total,fit,prior=objective()
        if not torch.isfinite(total):raise FloatingPointError(f'{scenario}, {frac}, epoch={epoch} loss nonfinite')
        total.backward();opt.step()
        if epoch%CHECK_EVERY!=0 and epoch!=MAX_EPOCHS:continue
        with torch.no_grad():
            total,fit,prior=objective()
            current={'epoch':epoch,'total_loss':floatnum(total.cpu()),
                     'unregularized_normalized_squared_Y_loss':floatnum(fit.cpu()),
                     'S2_prior_log_squared':floatnum(prior.cpu()),
                     'S2_Z_abs_ohm':float(abs(1/(np.exp(floatnum(model.base_g_log[1].cpu()))*1e-6+
                                                 1j*np.exp(floatnum(model.base_b_log[1].cpu()))*1e-6)))}
        history.append(current)
        atomic_torch_save({'signature':{**signature,'scenario':scenario,'fraction':frac,'lambda':lam},
                           'epoch':epoch,'model':model.state_dict(),
                           'optimizer':opt.state_dict(),'history':history},resume)
        print(f'{scenario} {frac}: epoch {epoch}/{MAX_EPOCHS}, '
              f'RMS_Y={np.sqrt(current["unregularized_normalized_squared_Y_loss"]):.5f}, '
              f'S2_Z={current["S2_Z_abs_ohm"]:.4g} ohm',flush=True)
    with torch.no_grad():
        yp,ys,zi,yi,yhat=model(A,C)
        yp=yp.cpu().numpy()
        # The first sample may be loaded. Recover the common UNLOADED baseline
        # directly from its two optimized parameter vectors, not ys[0].
        base=(torch.exp(model.base_g_log).cpu().numpy()+
              1j*torch.exp(model.base_b_log).cpu().numpy())*1e-6
        fit=floatnum((torch.mean(torch.abs(yhat-target)**2)/denom).cpu())
        reg=floatnum(((model.base_g_log[1]-prior_g)**2+
                      (model.base_b_log[1]-prior_b)**2).cpu())
        zpred=(1/yhat).cpu().numpy()
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(yp))):
        raise FloatingPointError('Nonfinite optimized parameters')
    pd.DataFrame({'pair_index':np.arange(91),
                  'pair':[f'{u}-{v}' for u,v in i5.PAIRS],
                  'Gp_S':yp.real,
                  'Cp_F':yp.imag/(2*np.pi*1e5)}).to_csv(folder/'parasitic_parameters.csv',index=False)
    pd.DataFrame({'sensor':[f'S{k}' for k in range(1,20)],
                  'G_S':base.real,'B_S':base.imag,
                  'Z_abs_ohm':abs(1/base)}).to_csv(folder/'baseline_sensor_parameters.csv',index=False)
    pd.DataFrame(history).to_csv(folder/'optimization_history.csv',index=False)
    summary={'scenario':scenario,'lambda':lam,'fraction':frac,'n_epochs':MAX_EPOCHS,
             'unregularized_RMS_Y':np.sqrt(fit),
             'unregularized_squared_Y_loss':fit,'S2_prior_log_squared':reg,
             'S2_G_S':float(base[1].real),'S2_B_S':float(base[1].imag),
             'S2_Z_abs_ohm':float(abs(1/base[1])),
             'S2_to_reference_G_ratio':float(base[1].real/p['G_ref_S']),
             'S2_to_reference_B_ratio':float(base[1].imag/p['B_ref_S']),
             'median_other18_G_S':p['G_ref_S'],
             'median_other18_B_S':p['B_ref_S'],
             'mean_sample_relative_Z_L2':float(np.mean(np.linalg.norm(zpred-1/y,axis=1)/np.linalg.norm(1/y,axis=1))),
             'checkpoint_path':str(resume),
             'fit_not_proof_of_convergence':True}
    atomic_json_save(summary,folder/'fit_summary.json')
    return summary,base,yp


def external_100khz(i5,A,C,results,signature):
    """Independent no-load calibration plus held-out loaded paths, V2.4 rules.

    Uses the exact physical 22 raw sweeps, without reusing any loaded pair as
    an input to its own held-out prediction. The 91 no-load pairs are used for
    baseline calibration, as in the original V2.6/V2.7 validation.
    """
    if not V26_SCRIPT.exists():
        print('SKIP independent validation: V2.6 script not found',flush=True);return
    v26=import_py(V26_SCRIPT,'e123_v26')
    if not v26.SOURCE.is_dir():
        print('SKIP independent validation: 22 raw scans folder not available',flush=True);return
    ext=v26.import_file(v26.EXTRACTOR,'e123_extract')
    v24=v26.import_file(v26.V24,'e123_v24')
    v22=v24.import_v22()
    raw_files=sorted(v26.SOURCE.glob('*_91pairs_10freq.txt'))
    if len(raw_files)!=22:raise ValueError('Expected exactly 22 multi-frequency raw sweeps')
    records=[]
    for path in raw_files:
        zz,ff,mask,info=ext.load_one(path)
        if info.get('issues'):raise ValueError('Unverified raw sweep: '+str(path))
        records.append((path,zz,mask,info))
    nos=[q for q in records if not np.asarray(q[2]).any()]
    loads=[q for q in records if np.asarray(q[2]).any()]
    if len(nos)!=1 or len(loads)!=21:raise ValueError('Unloaded/loaded scan composition invalid')
    freqs=np.asarray(ext.EXPECTED_FREQUENCIES_HZ,dtype=float)
    fi=int(np.argmin(abs(freqs-100000.0)))
    if not np.isclose(freqs[fi],100000.0,rtol=1e-6):raise ValueError('100 kHz point missing')
    Yno=1/np.asarray(nos[0][1][fi],dtype=complex)
    summary_rows=[]
    for (scenario,frac),(base100,para100) in results.items():
        base,diag=v24.calibrate_baseline(v22,i5,A.cpu(),C.cpu(),Yno,base100,para100)
        if not diag['success']:raise RuntimeError('No-load calibration failed '+scenario)
        for path,zz,mask,info in loads:
            active=np.flatnonzero(mask)
            for fold in (0,1):
                train=np.flatnonzero(np.arange(91)%2==fold)
                hold=np.flatnonzero(np.arange(91)%2!=fold)
                fit,fd=v24.fit_pressed(v22,i5,A.cpu(),C.cpu(),zz[fi],base,para100,active,train)
                if fit is None or not fd['fit_success']:
                    raise RuntimeError('Loaded fit failed for '+str(path))
                zhat=v22.predict_z(i5,A.cpu(),C.cpu(),fit,para100)
                met=v26.metrics(zz[fi,hold],zhat[hold])
                summary_rows.append({'scenario':scenario,'fraction':frac,'source_file':path.name,
                                     'condition':info['condition_canonical'],
                                     'fold':fold,'train_pairs':len(train),'holdout_pairs':len(hold),
                                     'baseline_relative_Y_residual':diag['relative_Y_residual'],
                                     **met})
    df=pd.DataFrame(summary_rows)
    if len(df)!=9*21*2:raise ValueError('Wrong external held-out evaluation row count')
    df.to_csv(OUT_ROOT/'independent_100k_heldout_detail.csv',index=False)
    summary=df.groupby(['scenario','fraction'],as_index=False).agg(
        n=('relative_Z_L2','size'),
        heldout_Z_relative_L2_mean=('relative_Z_L2','mean'),
        heldout_Y_relative_L2_mean=('relative_Y_L2','mean'),
        magnitude_MAPE_mean=('magnitude_MAPE','mean'),
        phase_MAE_deg_mean=('phase_MAE_deg','mean'))
    summary.to_csv(OUT_ROOT/'independent_100k_heldout_summary.csv',index=False)
    print('External 100 kHz held-out validation saved',flush=True)


def main():
    i5,z,y,manifest,mask,A,C,priors,original,signature=find_source()
    OUT_ROOT.mkdir(parents=True,exist_ok=True)
    # Save the immutable study definition before optimizations.
    atomic_json_save({'signature':signature,'priors_by_fraction':{str(k):v for k,v in priors.items()},
                      'note':'Reference prior from other 18 ORIGINAL I.5 baseline parameters; not sensor ground truth.'},
                     OUT_ROOT/'study_definition.json')
    target=torch.as_tensor(y,dtype=torch.complex128,device=DEVICE)
    denom=torch.mean(torch.abs(target)**2)
    rows=[];params={}
    for scenario,lam in SCENARIOS:
        for frac in FRACTIONS:
            result,base,yp=run_one(i5,y,mask,A,C,priors,signature,scenario,lam,frac,target,denom)
            rows.append(result)
            params[(scenario,frac)]=(base,yp)
            pd.DataFrame(rows).to_csv(OUT_ROOT/'experiment_summary_partial.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT_ROOT/'experiment_summary.csv',index=False)
    # Each experiment has newly estimated path-specific parasitic parameters.
    # Compare their changes against the frozen original I.5 estimation.
    orig={entry['fraction']:entry for entry in original}
    diff=[]
    for result in rows:
        scenario=result['scenario'];frac=result['fraction']
        yy=params[(scenario,frac)][1]
        y0=orig[frac]['original_Yp']
        diff.append({'scenario':scenario,'fraction':frac,
                     'original_S2_Z_abs_ohm':orig[frac]['S2_Z_abs_ohm'],
                     'new_S2_Z_abs_ohm':result['S2_Z_abs_ohm'],
                     'Yp_symmetric_relative_L2':float(2*np.linalg.norm(yy-y0)/
                       max(np.linalg.norm(yy)+np.linalg.norm(y0),1e-30))})
    pd.DataFrame(diff).to_csv(OUT_ROOT/'compare_original_I5.csv',index=False)
    if RUN_EXTERNAL_100KHZ_VALIDATION:
        external_100khz(i5,A,C,params,signature)
    print('FINISHED: '+str(OUT_ROOT),flush=True)


if __name__=='__main__':
    main()
