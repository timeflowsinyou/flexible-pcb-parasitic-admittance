# -*- coding: utf-8 -*-
"""Experiment I.5 — restart-safe convergence diagnosis (calibration free).

Spyder F5, no arguments needed. Only first 644 rows, with fixed known
press support and 91 shared complex parasitic admittances. No prior Gp/Cp.

IMPORTANT: I.4 did NOT save its model/optimizer state; the first execution
must restart from the same initial seeds. This script saves a COMPLETE torch
checkpoint every 200 epochs, enabling exact optimizer-state continuation.

Three principal outputs: identifiability_summary.json,
estimated_parasitics_stability.csv, example_impedance_cases.png.
One additional .pt checkpoint is essential for genuine resumption.
"""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

VERSION='1.0.0-true-checkpoint'
BASE=Path(r'D:\IEEE_TIM_related\Simulation_Experiment')
INPUT_DIR=BASE/'real91_extract'/'run_20261103T145251_758871Z'
NPZ_PATH=INPUT_DIR/'real_91pair_measurements.npz'
MANIFEST_PATH=INPUT_DIR/'sample_manifest.csv'
AUDIT_PATH=INPUT_DIR/'extraction_audit.json'
OUTPUT_ROOT=BASE/'experiment_I5_checkpointed_convergence'
N_USE=644
N_EXPECTED_SOURCE=804
FREQUENCY_HZ=100000.0
INITIAL_FRACTIONS=(0.05,0.30,0.70)
SENSOR_INIT_SEED=24
LR=0.025
# After 2400 epochs in I.4, losses and parameters had NOT reached plateau.
MAX_EPOCHS=8000
MIN_EPOCHS=2600
CHECK_EVERY=200
LOSS_PLATEAU_THRESHOLD=0.001
PARAM_PLATEAU_THRESHOLD=0.01
CROSS_RESTART_GAP_THRESHOLD=0.05  # symmetric L2 gap < 5% for every pair
# Require two consecutive 200-epoch windows satisfying ALL tests.
REQUIRED_CONSECUTIVE_WINDOWS=2
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
TORCH_REAL=torch.float64
TORCH_COMPLEX=torch.complex128

PHYSICAL_EDGES=((1,2),(1,8),(2,3),(2,9),(3,4),(3,10),(4,5),(4,11),
                (5,6),(5,12),(6,7),(6,13),(7,14),(8,9),(9,10),
                (10,11),(11,12),(12,13),(13,14))
CANON_NODES=tuple(list(range(1,8))+list(range(14,7,-1)))
PAIRS=tuple((u,v) for i,u in enumerate(CANON_NODES[:-1]) for v in CANON_NODES[i+1:])
assert len(PHYSICAL_EDGES)==19 and len(PAIRS)==91

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def parse_support(label):
    s = str(label).strip().strip('"').strip("'")
    if s in ('0','0.0',''):
        return []
    values = [int(v) for v in s.split('-')]
    if any(v < 1 or v > 19 for v in values) or len(values) != len(set(values)):
        raise ValueError(f'Invalid or duplicate sensor in load label {label!r}')
    return values


def load_inputs(npz_path=NPZ_PATH,manifest_path=MANIFEST_PATH):
    # The original raw measurement extraction is the source of truth.
    with np.load(npz_path,allow_pickle=False) as d:
        if d['Z_measured'].shape != (N_EXPECTED_SOURCE,91):
            raise ValueError(f"Unexpected Z shape: {d['Z_measured'].shape}")
        z = np.asarray(d['Z_measured'][:N_USE],dtype=np.complex128)
        npz_labels = d['labels'][:N_USE].astype(str)
        pairs = d['pair_ids'].astype(str).tolist()
        freq = float(d['frequency_hz'])
        file_index = d['file_sample_index'][:N_USE].astype(int)
    if pairs != [f'{u}-{v}' for u,v in PAIRS]:
        raise ValueError('Pair order inconsistent with original 91-pair data.')
    if not np.isclose(freq,FREQUENCY_HZ):
        raise ValueError(f'Frequency mismatch {freq}')
    m = pd.read_csv(manifest_path,dtype=str,keep_default_na=False)
    m.columns = m.columns.str.strip().str.lstrip('\ufeff')
    cols={'global_row','source_file','file_sample_index_zero_based','load_label'}
    if not cols.issubset(m.columns):
        raise ValueError(f'Manifest missing columns: {cols-set(m.columns)}')
    if len(m)!=N_EXPECTED_SOURCE:
        raise ValueError(f'Unexpected manifest rows: {len(m)}')
    m = m.iloc[:N_USE].copy().reset_index(drop=True)
    if not np.array_equal(m['global_row'].to_numpy(dtype=int),np.arange(N_USE)):
        raise ValueError('Original row ordering mismatch in manifest')
    if not np.array_equal(m['file_sample_index_zero_based'].to_numpy(dtype=int),file_index):
        raise ValueError('File row ordering mismatch in manifest')
    if not np.array_equal(m['load_label'].to_numpy(str),npz_labels):
        raise ValueError('Load labels disagree between NPZ and manifest')
    if not np.isfinite(z).all() or np.any(np.abs(z)<1e-14):
        raise ValueError('Invalid complex measured impedances')
    y=1/z
    if np.any(y.real<=0) or np.any(y.imag<=0):
        raise ValueError('The passive RC network model requires G_meas>0, B_meas>0')
    mask=np.zeros((N_USE,19),dtype=bool)
    for n,label in enumerate(m['load_label']):
        for s in parse_support(label): mask[n,s-1]=True
    return z, y, m, mask


def incidence_matrices(device):
    # Original E model: node 14 is grounded; measurement-pair order is canonical.
    A=np.zeros((13,19),dtype=np.float64)
    C=np.zeros((13,91),dtype=np.float64)
    for k,(u,v) in enumerate(PHYSICAL_EDGES):
        if u<14:A[u-1,k]+=1
        if v<14:A[v-1,k]-=1
    for k,(u,v) in enumerate(PAIRS):
        if u<14:C[u-1,k]+=1
        if v<14:C[v-1,k]-=1
    if np.linalg.matrix_rank(A)!=13:raise ValueError('Physical network not connected')
    return (torch.tensor(A,dtype=TORCH_COMPLEX,device=device),
            torch.tensor(C,dtype=TORCH_COMPLEX,device=device))


def forward_y_sensor_to_intrinsic(A,C,y_sensor):
    # L[n,:,:]=A diag(y[n,:]) A^T. Solve 91 unit-current excitations.
    L=torch.einsum('ik,nk,jk->nij',A,y_sensor,A)
    V=torch.linalg.solve(L,C.expand(y_sensor.shape[0],-1,-1))
    z_intrinsic=torch.einsum('ik,nik->nk',C,V)
    return z_intrinsic,1/z_intrinsic


class KnownSupportJointModel(torch.nn.Module):
    def __init__(self,y_measured_np,mask_np,fraction,device):
        super().__init__()
        rng=np.random.default_rng(SENSOR_INIT_SEED)
        n,k=mask_np.shape
        i,j=np.nonzero(mask_np)
        self.register_buffer('active_rows',torch.tensor(i,dtype=torch.long,device=device))
        self.register_buffer('active_cols',torch.tensor(j,dtype=torch.long,device=device))
        self.n=n
        self.k=k
        # Common sensor starts for all fractions, to isolate parasitic initialization.
        base_g=np.full((k,),3e-6)*rng.lognormal(0.,0.15,k)
        base_b=np.full((k,),1e-5)*rng.lognormal(0.,0.15,k)
        med_g=np.median(y_measured_np.real,axis=0)
        med_b=np.median(y_measured_np.imag,axis=0)
        self.gpara_log=torch.nn.Parameter(torch.tensor(np.log(np.maximum(fraction*med_g,1e-10)/1e-6),dtype=TORCH_REAL,device=device))
        self.bpara_log=torch.nn.Parameter(torch.tensor(np.log(np.maximum(fraction*med_b,1e-10)/1e-6),dtype=TORCH_REAL,device=device))
        self.base_g_log=torch.nn.Parameter(torch.tensor(np.log(base_g/1e-6),dtype=TORCH_REAL,device=device))
        self.base_b_log=torch.nn.Parameter(torch.tensor(np.log(base_b/1e-6),dtype=TORCH_REAL,device=device))
        # Only known pressed positions receive trainable change parameters.
        self.delta_g=torch.nn.Parameter(torch.zeros(len(i),dtype=TORCH_REAL,device=device))
        self.delta_b=torch.nn.Parameter(torch.zeros(len(i),dtype=TORCH_REAL,device=device))

    def forward(self,A,C):
        gg=self.base_g_log.unsqueeze(0).expand(self.n,-1).clone()
        bb=self.base_b_log.unsqueeze(0).expand(self.n,-1).clone()
        gg[self.active_rows,self.active_cols] += self.delta_g
        bb[self.active_rows,self.active_cols] += self.delta_b
        sensor=torch.complex(torch.exp(gg)*1e-6,torch.exp(bb)*1e-6)
        ypara=torch.complex(torch.exp(self.gpara_log)*1e-6,torch.exp(self.bpara_log)*1e-6)
        zintr,yintr=forward_y_sensor_to_intrinsic(A,C,sensor)
        yfit=yintr+ypara.unsqueeze(0)
        return ypara,sensor,zintr,yintr,yfit




def make_model_state(ym, mask, fraction, device):
    # Same optimizer setup, physical mapping, and initial values as I.4.
    torch.manual_seed(SENSOR_INIT_SEED)
    model=KnownSupportJointModel(ym,mask,fraction,device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    return model,opt


def current_evaluation(model,A,C,ytarget,denom):
    with torch.no_grad():
        yp,ys,zi,yi,yfit=model(A,C)
        loss=float((torch.mean(torch.abs(yfit-ytarget)**2)/denom).cpu())
        return loss,yp.detach().cpu().numpy().copy()


def symmetric_gap(x,y):
    return float(2*np.linalg.norm(x-y)/max(np.linalg.norm(x)+np.linalg.norm(y),1e-18))


def build_signature(paths, y, mask):
    return dict(version=VERSION, sha256={p.name:sha256(p) for p in paths},
                n_samples=N_USE, n_pairs=91, n_active=int(mask.sum()),
                frequency_hz=FREQUENCY_HZ, fractions=list(INITIAL_FRACTIONS),
                seed=SENSOR_INIT_SEED, lr=LR,
                physical_edges=[list(x) for x in PHYSICAL_EDGES],
                check_every=CHECK_EVERY)


def find_resume_run(root):
    """Auto-resume newest checkpoint-bearing run. Rename/remove it to start new."""
    if not root.is_dir():return None
    runs=sorted(root.glob('run_*'),reverse=True)
    for r in runs:
        if (r/'checkpoint.pt').is_file() and not (r/'FINALIZED').exists():
            return r
    return None


def save_atomic(state, path):
    tmp=path.with_suffix('.tmp')
    torch.save(state,tmp)
    os.replace(tmp,path)


def render_plot(output, zmeas, models, A,C, mask, manifest, yt):
    with torch.no_grad():
        finals=[]
        for model in models:
            yp,ys,zi,yi,yfit=model(A,C)
            cur_loss=float(torch.mean(torch.abs(yfit-yt)**2).cpu())
            finals.append((cur_loss,yp.cpu().numpy(),zi.cpu().numpy(),yfit.cpu().numpy()))
    best=min(finals,key=lambda x:x[0])
    counts=mask.sum(axis=1)
    rows=[]
    for n in (0,4):
        hits=np.flatnonzero(counts==n)
        if len(hits):rows.append(int(hits[0]))
    if len(rows)<2:rows=[0,1]
    fig,axes=plt.subplots(2,len(rows),figsize=(7.2*len(rows),8.0),squeeze=False)
    xp=np.arange(1,92)
    for col,i in enumerate(rows):
        curves=[('Measured Z',zmeas[i]),('Estimated parasitic Z',1/best[1]),
                ('Estimated intrinsic Z',best[2][i]),('Parallel reconstructed Z',1/best[3][i])]
        for name,z in curves:
            axes[0,col].plot(xp,np.abs(z),label=name,linewidth=1.2)
            axes[1,col].plot(xp,np.degrees(np.angle(z)),label=name,linewidth=1.2)
        axes[0,col].set_title(f'row {i} | pressed={manifest.iloc[i]["load_label"]}')
        axes[0,col].set(ylabel='|Z| (Ohm)',xlabel='Measurement pair index',yscale='log')
        axes[1,col].set(ylabel='Phase (deg)',xlabel='Measurement pair index')
        for ax in axes[:,col]:ax.grid(alpha=0.25)
        if col==0:axes[0,col].legend(fontsize=8)
    fig.suptitle('I.5: measured, parasitic, intrinsic, reconstructed')
    fig.tight_layout()
    fig.savefig(output,dpi=160)
    plt.close(fig)
    return rows


def export_outputs(out, models, A,C,yt,z,mask,m,signature,epoch,records,reason):
    yp_all=[]; fit_rows=[]
    with torch.no_grad():
        for fr,mod in zip(INITIAL_FRACTIONS,models):
            yp,ys,zi,yi,yfit=mod(A,C)
            yp=yp.cpu().numpy(); yp_all.append(yp)
            yhat=yfit.cpu().numpy(); zhat=1/yhat
            loss=float(np.mean(np.abs(yhat-1/z)**2)/np.mean(np.abs(1/z)**2))
            fit_rows.append(dict(initial_fraction=fr,
                normalized_squared_Y_loss=loss,
                relative_RMS_Y_error=float(np.sqrt(loss)),
                mean_sample_relative_L2_Y=float(np.mean(np.linalg.norm(yhat-1/z,axis=1)/np.linalg.norm(1/z,axis=1))),
                mean_sample_relative_L2_Z=float(np.mean(np.linalg.norm(zhat-z,axis=1)/np.linalg.norm(z,axis=1)))))
    vals=np.stack(yp_all)
    tab=pd.DataFrame({'pair_index':np.arange(91),'pair':[f'{u}-{v}' for u,v in PAIRS]})
    for fr,yp in zip(INITIAL_FRACTIONS,yp_all):
        nm=f'init_{int(round(fr*100))}pct'
        tab[nm+'_Gp_S']=yp.real
        tab[nm+'_Cp_F']=yp.imag/(2*np.pi*FREQUENCY_HZ)
    tab['Gp_CV']=np.std(vals.real,axis=0)/np.maximum(np.mean(vals.real,axis=0),1e-20)
    tab['Cp_CV']=np.std(vals.imag,axis=0)/np.maximum(np.mean(vals.imag,axis=0),1e-20)
    csv=out/'estimated_parasitics_stability.csv'
    tab.to_csv(csv,index=False,encoding='utf-8-sig')
    fig=out/'example_impedance_cases.png'
    selected=render_plot(fig,z,models,A,C,mask,m,yt)
    gap=[]
    for i in range(3):
        for j in range(i+1,3):
            gap.append({'fractions':[INITIAL_FRACTIONS[i],INITIAL_FRACTIONS[j]],
                        'symmetric_relative_L2':symmetric_gap(vals[i],vals[j])})
    summary=dict(experiment='I5_known_support_checkpointed_convergence',version=VERSION,
                 source='First 644 calibration-associated measurements only; last 160 excluded',
                 n_samples=644,n_node_pairs=91,n_sensors=19,
                 source_sha256=signature['sha256'],physics=dict(
                     model='14-node/19-sensor nodal forward with known pressed support',
                     Y_measured='Y_intrinsic + Y_parasitic',
                     constraints=['Gp>0','Cp>0','Re/Im sensor admittance >0',
                                  'unpressed sensors equal common baselines']),
                 optimization=dict(epoch=epoch,max_epochs=MAX_EPOCHS,min_epochs=MIN_EPOCHS,
                     check_every=CHECK_EVERY,loss_plateau_threshold=LOSS_PLATEAU_THRESHOLD,
                     param_plateau_threshold=PARAM_PLATEAU_THRESHOLD,
                     cross_restart_gap_threshold=CROSS_RESTART_GAP_THRESHOLD,
                     stopped_because=reason,complete=reason!='interrupted'),
                 fits=fit_rows,trajectory=records,
                 final_pairwise_Ypara_distances=gap,
                 Gp_CV_median=float(tab.Gp_CV.median()),
                 Cp_CV_median=float(tab.Cp_CV.median()),
                 plot_rows=selected,
                 warnings=['I.4 did not store optimizer states; first run must reproduce earlier steps.',
                           'Cross-initialization agreement is not proof of structural identifiability.',
                           'No independent ground-truth intrinsic impedance.',
                           'Input files are read-only; checkpoint data is separate.'])
    summary['output_sha256']={p.name:sha256(p) for p in [csv,fig]}
    (out/'identifiability_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    return summary


def main():
    paths=(NPZ_PATH,MANIFEST_PATH,AUDIT_PATH)
    for path in paths:
        if not path.is_file():raise FileNotFoundError(f'Input file missing: {path}')
    z,y,m,mask=load_inputs(NPZ_PATH,MANIFEST_PATH)
    if mask.sum()!=1801:raise ValueError('Pressed support count unexpectedly differs from verified 1801.')
    signature=build_signature(paths,y,mask)
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    out=find_resume_run(OUTPUT_ROOT)
    if out is None:
        out=OUTPUT_ROOT/datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
        out.mkdir(exist_ok=False)
        epoch0=0; previous=None; records=[]; consecutive=0
        print('New run; cannot load I.4 endpoint because no I.4 checkpoint exists.',flush=True)
    else:
        # Load only locally generated checkpoints in the user-controlled output directory.
        # weights_only=False is required for NumPy diagnostic arrays and optimizer state;
        # NEVER load checkpoint files from untrusted sources.
        saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Checkpoint/input/config mismatch, refusing to resume.')
        epoch0=int(saved['epoch']); previous=saved['previous']; records=saved['records']; consecutive=saved['consecutive']
        print(f'Resuming {out}, starting AFTER epoch {epoch0}',flush=True)
    A,C=incidence_matrices(DEVICE)
    target=torch.tensor(y,dtype=TORCH_COMPLEX,device=DEVICE)
    denom=torch.mean(torch.abs(target)**2)
    models=[]; optimizers=[]
    for fr in INITIAL_FRACTIONS:
        model,opt=make_model_state(y,mask,fr,DEVICE)
        models.append(model); optimizers.append(opt)
    if epoch0:
        for i in range(3):
            models[i].load_state_dict(saved['models'][i])
            optimizers[i].load_state_dict(saved['optimizers'][i])
            # Checkpoint is device-independent; optimizer states should follow params.
            for state in optimizers[i].state.values():
                for key,value in state.items():
                    if isinstance(value,torch.Tensor):state[key]=value.to(DEVICE)
        del saved
    if epoch0>=MAX_EPOCHS:
        print(f'Existing checkpoint already reached epoch {epoch0} >= MAX_EPOCHS={MAX_EPOCHS}. '
              'Increase MAX_EPOCHS in this script to continue.',flush=True)
        return
    reason='max_epochs_reached'
    try:
        for ep in range(epoch0+1,MAX_EPOCHS+1):
            for model,opt in zip(models,optimizers):
                opt.zero_grad(set_to_none=True)
                yp,ys,zi,yi,yhat=model(A,C)
                loss=torch.mean(torch.abs(yhat-target)**2)/denom
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at epoch {ep}')
                loss.backward()
                opt.step()
            if ep%CHECK_EVERY!=0:continue
            data=[current_evaluation(model,A,C,target,denom) for model in models]
            losses=[x[0] for x in data]
            paras=[x[1] for x in data]
            gaps=[symmetric_gap(paras[i],paras[j]) for i,j in ((0,1),(0,2),(1,2))]
            windows=[]
            if previous is not None:
                for i in range(3):
                    l0=previous['losses'][i]
                    relative_loss_change=abs(l0-losses[i])/max(l0,1e-18)
                    para_change=float(np.linalg.norm(paras[i]-previous['paras'][i])/max(np.linalg.norm(previous['paras'][i]),1e-18))
                    windows.append({'init':INITIAL_FRACTIONS[i],
                                    'abs_relative_loss_change':relative_loss_change,
                                    'relative_parameter_change':para_change,
                                    'both_plateau':relative_loss_change<LOSS_PLATEAU_THRESHOLD and para_change<PARAM_PLATEAU_THRESHOLD})
            cross_ok=all(g<CROSS_RESTART_GAP_THRESHOLD for g in gaps)
            plateau_ok=bool(windows and all(w['both_plateau'] for w in windows))
            if ep>=MIN_EPOCHS and plateau_ok and cross_ok:
                consecutive+=1
            else:consecutive=0
            record={'epoch':ep,'normalized_squared_Y_losses':losses,
                    'RMS_Y_errors':[float(np.sqrt(q)) for q in losses],
                    'symmetric_L2_Ypara_gaps':gaps,
                    'window_checks':windows,
                    'cross_restarts_within_5pct':cross_ok,
                    'plateau_all_restarts':plateau_ok,
                    'consecutive_passing_windows':consecutive}
            records.append(record)
            previous={'losses':losses,'paras':paras}
            checkpoint={'signature':signature,'epoch':ep,'models':[q.state_dict() for q in models],
                        'optimizers':[q.state_dict() for q in optimizers],
                        'previous':previous,'records':records,'consecutive':consecutive}
            save_atomic(checkpoint,out/'checkpoint.pt')
            print(f'epoch {ep:5d}: RMS={[round(np.sqrt(l),6) for l in losses]} '
                  f'para_gaps={[round(g,4) for g in gaps]} '
                  f'plateau={plateau_ok} cross={cross_ok} pass={consecutive}',flush=True)
            # Interim report and image so an interruption does not discard diagnostics.
            export_outputs(out,models,A,C,target,z,mask,m,signature,ep,records,'interrupted')
            if consecutive>=REQUIRED_CONSECUTIVE_WINDOWS:
                reason='all_loss_parameter_plateaus_and_cross_restart_agreement'
                break
        summary=export_outputs(out,models,A,C,target,z,mask,m,signature,ep,records,reason)
        if reason=='all_loss_parameter_plateaus_and_cross_restart_agreement':
            (out/'FINALIZED').write_text(reason,encoding='utf-8')
        print('Experiment I.5 finished. Reason:',reason,'output:',out,flush=True)
        if reason=='max_epochs_reached':
            print('WARNING: Reaching MAX_EPOCHS does NOT imply convergence. '
                  'Change MAX_EPOCHS and resume with checkpoint if desired.',flush=True)
    except BaseException:
        print('Experiment stopped before finalization. Restart same script to resume '
              'from last completed 200-epoch checkpoint.',flush=True)
        raise


if __name__=='__main__':
    main()
