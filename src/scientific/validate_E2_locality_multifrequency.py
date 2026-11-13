# -*- coding: utf-8 -*-
"""E2 versus original I.5: parasitic change locality + five-frequency validation.

Run with Spyder F5 in D:/IEEE_TIM_related/Simulation_Experiment.
READ ONLY: existing I.5/E2 optimizations and original AD2 data.
No parameter re-identification, no overwrite of any prior result.

Locality groups (mutually exclusive):
  direct_S2: node pair (1,8) [1 pair]
  S2_terminal_incident: other pairs involving node 1 or node 8 [24 pairs]
  remote: remaining pairs [66 pairs]
This is topological incidence, NOT an assertion about physical cable locality.

Five-frequency test duplicates V2.6 calibration and 45/46 complement holdouts.
Results from E0 and E2 are paired by scan, frequency, fold, I.5 restart.
The data used to select E2 include the 100 kHz validation; hence the
five-frequency comparison is exploratory, NOT an entirely untouched test set.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
if not BASE.is_dir(): BASE=Path(__file__).resolve().parent
E2_ROOT=BASE/'experiment_I5_S2_E1_E2_E3'/'E2_weak_S2_prior'
V26=BASE/'V2_6_validate_22_sweeps_fixed_parasitics.py'
V27_ROOT=BASE/'V2_7_publication_validation'/'run_20261105T150005_372663Z'
RAW=Path(r'D:\IEEE_TIM_related\Data\MultiFrequency_Measurements\2026_11_05')
OUT_ROOT=BASE/'E2_locality_and_multifrequency_validation'
FRACTIONS=(0.05,0.30,0.70)
PRIMARY=np.array([46415.8883361278,100000.,215443.4690031884,464158.88336127775,1e6])
VERSION='1.0.0'
ORIGINAL_I5_CHECKPOINT_SHA256='1840612c1acb81c1e4bb97df3cfb4315a76deb587a991627ee46563d91cbe366'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def import_mod(path,name):
    if not path.is_file(): raise FileNotFoundError(str(path))
    sp=importlib.util.spec_from_file_location(name,str(path))
    m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m)
    return m

def fpct(fr):return f'init_{int(round(fr*100)):02d}pct'

def canon(s):
    u,v=map(int,str(s).strip().split('-'))
    if u==v or u not in range(1,15) or v not in range(1,15):
        raise ValueError('Invalid node pair '+str(s))
    return tuple(sorted((u,v)))

def grp(pair):
    if pair==(1,8):return 'direct_S2_1-8'
    if pair[0] in (1,8) or pair[1] in (1,8):return 'S2_terminal_incident_other'
    return 'remote'

def fstats(x):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    if not len(a):return {'n':0,'mean':None,'median':None,'p90':None}
    return {'n':int(len(a)),'mean':float(a.mean()),'median':float(np.median(a)),'p90':float(np.percentile(a,90))}

def figure_plot(fig, folder, stem):
    fig.savefig(folder/(stem+'.png'),dpi=300,bbox_inches='tight')
    fig.savefig(folder/(stem+'.pdf'),bbox_inches='tight')
    plt.close(fig)

def get_models(v26):
    ext=import_mod(v26.EXTRACTOR,'e2_extract')
    v24=import_mod(v26.V24,'e2_v24')
    v22=v24.import_v22()
    prev=v22.get_previous_module()
    i5=prev.import_i5()
    epoch,oldrun,ckpt,A,C,old=prev.load_i5_parameters(i5)
    if epoch != 5200 or len(old)!=3:
        raise ValueError('Unexpected I.5 checkpoint epoch/restart count')
    if sha(Path(ckpt))!=ORIGINAL_I5_CHECKPOINT_SHA256:
        raise ValueError('Original I.5 checkpoint SHA256 differs from audited V2.7 source')
    if not np.allclose([m[0] for m in old],FRACTIONS):
        raise ValueError('I.5 restart order mismatch')
    pairs=[canon(f'{a}-{b}') for a,b in i5.PAIRS]
    if len(pairs)!=91 or len(set(pairs))!=91:
        raise ValueError('Invalid unique I.5 pair mapping')
    if Counter(grp(p) for p in pairs)!=Counter({'direct_S2_1-8':1,'S2_terminal_incident_other':24,'remote':66}):
        raise ValueError('Unexpected topological incidence counts')
    return ext,v24,v22,prev,i5,A,C,old,pairs,ckpt

def e2_parameters(pairs):
    output={};hashes={}
    for fr in FRACTIONS:
        folder=E2_ROOT/fpct(fr)
        p=folder/'parasitic_parameters.csv'
        b=folder/'baseline_sensor_parameters.csv'
        fit=folder/'fit_summary.json'
        for path in [p,b,fit]:
            if not path.is_file():raise FileNotFoundError(str(path))
            hashes[str(path)]=sha(path)
        prm=pd.read_csv(p,encoding='utf-8-sig')
        sens=pd.read_csv(b,encoding='utf-8-sig')
        meta=json.loads(fit.read_text(encoding='utf-8'))
        if str(meta['scenario'])!='E2_weak_S2_prior' or not np.isclose(float(meta['lambda']),1e-4) or not np.isclose(float(meta['fraction']),fr):
            raise ValueError('Wrong E2 model metadata')
        if len(prm)!=91 or not np.array_equal(prm.pair_index.to_numpy(),np.arange(91)):
            raise ValueError('Incorrect parasitic CSV pair indices')
        if [canon(s) for s in prm.pair]!=pairs:
            raise ValueError('E2 91-pair ordering DOES NOT match I.5 -- stop, do not reorder')
        if sens.sensor.astype(str).tolist()!=[f'S{i}' for i in range(1,20)]:
            raise ValueError('E2 19-sensor ordering mismatch')
        base=sens.G_S.to_numpy(float)+1j*sens.B_S.to_numpy(float)
        yp=prm.Gp_S.to_numpy(float)+1j*2*np.pi*1e5*prm.Cp_F.to_numpy(float)
        if (not np.isfinite(base).all() or not np.isfinite(yp).all()
            or np.any(base.real<=0) or np.any(base.imag<=0)
            or np.any(yp.real<0) or np.any(yp.imag<0)):
            raise ValueError('Non-passive or invalid E2 coefficients')
        output[fr]=(base,yp)
    return output, hashes

def locality(old,e2,pairs,out):
    records=[];summary=[]
    for fr,base0,yp0 in old:
        base1,yp1=e2[fr]
        for ki,pair in enumerate(pairs):
            # Per-pair: relative to mean amplitude (bounded between 0 and 2)
            dy=yp1[ki]-yp0[ki]
            d=np.abs(dy)
            denom=max(0.5*(abs(yp1[ki])+abs(yp0[ki])),1e-20)
            records.append(dict(restart_fraction=fr,pair_index_1based=ki+1,
                pair=f'{pair[0]}-{pair[1]}',group=grp(pair),
                Gp_original_S=float(yp0[ki].real),Gp_E2_S=float(yp1[ki].real),
                Cp_original_F=float(yp0[ki].imag/(2*np.pi*1e5)),
                Cp_E2_F=float(yp1[ki].imag/(2*np.pi*1e5)),
                Yp_original_abs_S=float(abs(yp0[ki])),Yp_E2_abs_S=float(abs(yp1[ki])),
                abs_delta_Yp_S=float(d),symmetric_relative_Yp_change=float(d/denom),
                delta_Gp_S=float(dy.real),delta_Cp_F=float(dy.imag/(2*np.pi*1e5))))
        for name in ('direct_S2_1-8','S2_terminal_incident_other','remote'):
            ix=[k for k,p in enumerate(pairs) if grp(p)==name]
            diff=yp1[ix]-yp0[ix]
            summary.append(dict(restart_fraction=fr,group=name,n_pairs=len(ix),
                norm_delta_Yp_S=float(np.linalg.norm(diff)),
                norm_old_Yp_S=float(np.linalg.norm(yp0[ix])),
                symmetric_L2_change=float(2*np.linalg.norm(diff)/max(np.linalg.norm(yp1[ix])+np.linalg.norm(yp0[ix]),1e-20)),
                mean_abs_delta_Yp_S=float(np.mean(np.abs(diff)))))
    df=pd.DataFrame(records)
    df.to_csv(out/'E2_per_pair_parasitic_changes.csv',index=False)
    ss=pd.DataFrame(summary)
    # Extra fair comparisons: group change energy fraction, group baseline energy fraction
    for fr,group in ss.groupby('restart_fraction'):
        mask=ss.restart_fraction==fr
        ss.loc[mask,'change_energy_fraction']=group.norm_delta_Yp_S.to_numpy()**2/sum(group.norm_delta_Yp_S.to_numpy()**2)
        ss.loc[mask,'original_energy_fraction']=group.norm_old_Yp_S.to_numpy()**2/sum(group.norm_old_Yp_S.to_numpy()**2)
    ss.to_csv(out/'E2_locality_summary.csv',index=False)
    # One practical indicator: which paths carry largest changes, without hiding others
    ranked=df.groupby(['pair_index_1based','pair','group'],as_index=False).agg(
        mean_absolute_Yp_change_S=('abs_delta_Yp_S','mean'),
        mean_symmetric_relative_change=('symmetric_relative_Yp_change','mean'))
    ranked=ranked.sort_values('mean_absolute_Yp_change_S',ascending=False)
    ranked.to_csv(out/'E2_locality_ranked_all91.csv',index=False)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    x=np.arange(1,92)
    for fr,g in df.groupby('restart_fraction'):
        g=g.sort_values('pair_index_1based')
        axes[0].plot(x,g.abs_delta_Yp_S*1e6,label=f'I.5 init. {fr:.2f}',linewidth=1)
        axes[1].plot(x,g.symmetric_relative_Yp_change,label=f'I.5 init. {fr:.2f}',linewidth=1)
    axes[0].set_ylabel('|ΔYp| (µS)')
    axes[1].set_ylabel('Per-pair symmetric relative change')
    axes[1].set_xlabel('Original 91-path acquisition index')
    for ax in axes:
        ax.axvspan(0.5,13.5,alpha=.08,color='grey')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    figure_plot(fig,out,'Fig_E2_91_path_parasitic_changes')
    return df,ss,ranked

def load_raw(ext,summary,pairs):
    ref=summary['source_raw_files_verified_sha256']
    files=sorted(RAW.glob('*_91pairs_10freq.txt'))
    if len(files)!=22 or set(p.name for p in files)!=set(ref):
        raise ValueError('Independent dataset must match EXACTLY the 22 V2.7 audited sweep filenames')
    Z=[];masks=[];labels=[];names=[]
    for p in files:
        if sha(p)!=ref[p.name]:raise ValueError('SHA256 mismatch: '+p.name)
        zz,ff,mask,info=ext.load_one(p)
        if info.get('issues'):raise ValueError('Extract audit: '+repr(info['issues']))
        Z.append(zz);masks.append(mask);labels.append(info['condition_canonical']);names.append(p.name)
    Z=np.stack(Z);masks=np.asarray(masks,dtype=bool);labels=np.asarray(labels);names=np.asarray(names)
    if Z.shape!=(22,10,91) or masks.shape!=(22,19) or not np.isfinite(Z).all():
        raise ValueError('Independent dataset shape/values mismatch')
    if Counter(labels)!=Counter({'noloads':1,'sensor2_16':1,**{f'sensor{i}':(2 if i==16 else 1) for i in range(1,20)}}):
        raise ValueError('Loaded-case composition differs from V2.7')
    freqs=np.asarray(ext.EXPECTED_FREQUENCIES_HZ,dtype=float)
    sel=[]
    for f in PRIMARY:
        k=int(np.argmin(abs(freqs-f)))
        if not np.isclose(freqs[k],f,rtol=2e-7):raise ValueError('Primary freq mapping mismatch')
        sel.append(k)
    noload=int(np.flatnonzero(~masks.any(axis=1))[0]);loaded=np.flatnonzero(masks.any(axis=1))
    return Z,masks,labels,names,noload,loaded,sel,freqs

def validate(v26,v24,v22,i5,A,C,old,e2,raw,summary,out):
    Z,masks,labels,names,noload,loaded,sel,freqs=raw
    rows=[];cal_rows=[]
    for fr,base0,yp0 in old:
        for scenario,(base100,yp100) in [('E0_original_I5',(base0,yp0)),('E2_weak_S2_prior',e2[fr])]:
            for fi in sel:
                f=float(freqs[fi]);yp=v24.frozen_yp_at_frequency(yp100,f)
                prior=v24.ref_baseline_at_frequency(base100,f)
                baseline,bd=v24.calibrate_baseline(v22,i5,A,C,1/Z[noload,fi],prior,yp)
                if not bd['success']:raise RuntimeError(f'Baseline fit failed {scenario} {fr} {f}')
                cal_rows.append(dict(scenario=scenario,restart_fraction=fr,frequency_hz=f,
                    baseline_relative_Y_residual=bd['relative_Y_residual'],
                    baseline_n_components_at_bound=bd['n_components_at_bound'],
                    S2_unloaded_calibrated_Z_abs_ohm=abs(1/baseline[1])))
                for ix in loaded:
                    pressed=np.flatnonzero(masks[ix])
                    for fold in (0,1):
                        train=np.flatnonzero(np.arange(91)%2==fold)
                        test=np.flatnonzero(np.arange(91)%2!=fold)
                        yfit,fd=v24.fit_pressed(v22,i5,A,C,Z[ix,fi],baseline,yp,pressed,train)
                        if yfit is None or not fd['fit_success']:
                            raise RuntimeError(f'Loaded fit failed {scenario} {fr} {names[ix]} f={f} fold={fold}')
                        zp=v22.predict_z(i5,A,C,yfit,yp)
                        mt=v26.metrics(Z[ix,fi,test],zp[test])
                        rows.append(dict(scenario=scenario,restart_fraction=fr,
                            frequency_hz=f,condition=labels[ix],source_file=names[ix],
                            fold=fold,n_train=len(train),n_holdout=len(test),
                            S2_involved=bool(masks[ix,1]),
                            train_fit_cost=float(fd['fit_train_cost']),
                            n_active_at_bound=int(fd['n_at_bound']),**mt))
                print(f'COMPLETE: {scenario} init {fr:.2f}, {f:.1f} Hz',flush=True)
    detail=pd.DataFrame(rows);expected=2*3*5*21*2
    if len(detail)!=expected or detail.duplicated(['scenario','restart_fraction','frequency_hz','source_file','fold']).any():
        raise ValueError('Expected 1260 unique E0/E2 heldout rows')
    detail.to_csv(out/'E0_E2_five_frequency_holdout_detail.csv',index=False)
    pd.DataFrame(cal_rows).to_csv(out/'E0_E2_unloaded_baseline_diagnostics.csv',index=False)
    pivot=detail.pivot(index=['restart_fraction','frequency_hz','source_file','condition','fold','S2_involved'],columns='scenario',
        values=['relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg'])
    if pivot.isna().any().any() or len(pivot)!=630:raise ValueError('E0/E2 matched pairing incomplete')
    pivot.columns=[f'{a}_{b}' for a,b in pivot.columns]
    paired=pivot.reset_index()
    for metric in ('relative_Z_L2','relative_Y_L2','magnitude_MAPE','phase_MAE_deg'):
        paired[f'{metric}_E0_minus_E2']=paired[f'{metric}_E0_original_I5']-paired[f'{metric}_E2_weak_S2_prior']
    paired.to_csv(out/'E0_E2_five_frequency_paired.csv',index=False)
    ag=detail.groupby(['scenario','frequency_hz'],as_index=False).agg(
        n=('relative_Z_L2','size'),
        Z_mean=('relative_Z_L2','mean'),Y_mean=('relative_Y_L2','mean'),
        magnitude_MAPE_mean=('magnitude_MAPE','mean'),phase_MAE_deg_mean=('phase_MAE_deg','mean'),
        n_active_bounds=('n_active_at_bound','sum'))
    ag.to_csv(out/'E0_E2_by_frequency.csv',index=False)
    groups=detail.assign(sensor_group=np.where(detail.S2_involved,'S2_involved','without_S2'))
    groups.groupby(['scenario','sensor_group'],as_index=False).agg(n=('relative_Z_L2','size'),
        Z_mean=('relative_Z_L2','mean'),Y_mean=('relative_Y_L2','mean')).to_csv(out/'E0_E2_S2_subgroup.csv',index=False)
    paired.groupby('frequency_hz',as_index=False).agg(
        n=('relative_Z_L2_E0_minus_E2','size'),
        mean_Z_advantage_E2=('relative_Z_L2_E0_minus_E2','mean'),
        fraction_E2_better_Z=('relative_Z_L2_E0_minus_E2',lambda s:float((s>0).mean())),
        mean_Y_advantage_E2=('relative_Y_L2_E0_minus_E2','mean'),
        fraction_E2_better_Y=('relative_Y_L2_E0_minus_E2',lambda s:float((s>0).mean()))).to_csv(out/'E0_E2_paired_by_frequency.csv',index=False)
    # Fail closed if E0 E2 result baseline does not reproduce exact historical result for 100 kHz.
    ref=summary['by_frequency']
    expected0=next(v for v in ref if v['protocol'].startswith('C_') and np.isclose(v['frequency_hz'],1e5,rtol=1e-7))
    actual=ag[(ag.scenario=='E0_original_I5')&np.isclose(ag.frequency_hz,1e5,rtol=1e-7)]
    if len(actual)!=1 or not np.isclose(actual.Z_mean.iloc[0],expected0['relative_Z_L2']['mean'],rtol=1e-5,atol=1e-7):
        raise ValueError('E0 100k reproduction does NOT match V2.7 audited baseline')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    for m,ax in [('Z_mean',axes[0]),('Y_mean',axes[1])]:
        for sc,g in ag.groupby('scenario'):
            g=g.sort_values('frequency_hz')
            ax.plot(np.arange(5),100*g[m].to_numpy(),marker='o',label=sc)
        ax.set_xticks(range(5),['46.4','100','215.4','464.2','1000'])
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Mean held-out complex '+('Z' if m=='Z_mean' else 'Y')+' relative L2 (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();figure_plot(fig,out,'Fig_E0_E2_five_frequency_holdout')
    return detail,paired,ag

def main():
    needed=[V26,V27_ROOT/'V2_7_summary.json',RAW]
    for p in needed:
        if not p.exists():raise FileNotFoundError(str(p))
    summary=json.loads((V27_ROOT/'V2_7_summary.json').read_text(encoding='utf-8-sig'))
    v26=import_mod(V26,'e2_v26')
    ext,v24,v22,prev,i5,A,C,old,pairs,ckpt=get_models(v26)
    e2,hashes=e2_parameters(pairs)
    out=OUT_ROOT/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=False)
    locality_df,locality_sum,ranked=locality(old,e2,pairs,out)
    print('LOCALITY complete: ',out,flush=True)
    raw=load_raw(ext,summary,pairs)
    detail,paired,byfreq=validate(v26,v24,v22,i5,A,C,old,e2,raw,summary,out)
    # Audit: the same 3 restart cohorts; no hidden tuning in validation.
    audit={'version':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'e2_hyperparameter_lambda':1e-4,
        'source_hashes':{**hashes,str(ckpt):sha(ckpt),str(V26):sha(V26),
            str(V27_ROOT/'V2_7_summary.json'):sha(V27_ROOT/'V2_7_summary.json')},
        'raw_file_count':22,'n_locality_rows':len(locality_df),'n_heldout_rows':len(detail),
        'n_matched_rows':len(paired),
        'notes':['Locality groups use incidence to endpoints of S2 physical edge (1,8), not actual cable proximity.',
            'E2 is a 100 kHz validation-informed candidate; other frequencies are follow-up tests, not fully preregistered held-out confirmation.',
            'Both E0 and E2 are separately calibrated on ALL 91 no-load pairs at each frequency.',
            'Only loaded-state 45/46 pairs are held out from state fitting.',
            'Three restarts and complementary folds do NOT represent independent hardware repeats.',
            'Absolute intrinsic-impedance ground truth is unavailable.']}
    (out/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'README.txt').write_text('E2 local parasitic changes and five-frequency E0/E2 matched validation.\n'
       'Read audit.json for methodological limitations. Raw data and original checkpoints are unchanged.\n',encoding='utf-8')
    print('\nFINISHED:',out,flush=True)
    print('\nPer-frequency heldout error:\n',byfreq.to_string(index=False),flush=True)
    print('\nLocality summary:\n',locality_sum.groupby('group')[['symmetric_L2_change','change_energy_fraction']].mean().to_string(),flush=True)

if __name__=='__main__':main()
