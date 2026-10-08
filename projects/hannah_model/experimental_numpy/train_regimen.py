#!/usr/bin/env python3
"""Full-window two-epoch Hannah training with auditable metrics/checkpoints.

This is the NumPy implementation's reproducible baseline. It consumes each
next-character window once per epoch, shuffling window order with the recorded
seed. No Quinn schedule is implied or silently applied.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from autograd import Adam
from model import MiniHannahLM


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('hannah_two_epoch_run'))
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--limit-files',type=int,default=20)
    p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--d-model',type=int,default=256)
    p.add_argument('--d-ff',type=int,default=1024)
    p.add_argument('--layers',type=int,default=4)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--sphere-dim',type=int,default=20)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=20261007)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--checkpoint-every-batches',type=int,default=1000)
    p.add_argument('--resume',type=Path,default=None)
    a=p.parse_args()
    if min(a.epochs,a.seq_len,a.batch_size,a.limit_files,a.d_model,a.d_ff,a.layers,a.heads,a.sphere_dim)<=0:
        p.error('epochs, lengths, dimensions, and file limit must be positive')
    files=sorted(a.data_dir.glob('THRM-*.yaml'))[:a.limit_files]
    if not files: raise SystemExit(f'No THRM-*.yaml files in {a.data_dir}')
    # The recovered trainer appends two newlines after every source file.
    text=''.join(f.read_text(encoding='utf-8')+'\n\n' for f in files)
    chars=sorted(set(text)); c2i={c:i for i,c in enumerate(chars)}
    ids=np.fromiter((c2i[c] for c in text),dtype=np.int64,count=len(text))
    windows=len(ids)-a.seq_len
    if windows<=0: raise SystemExit(f'Corpus length {len(ids)} must exceed seq-len {a.seq_len}')
    batches=(windows+a.batch_size-1)//a.batch_size
    rng=np.random.default_rng(a.seed)
    model=MiniHannahLM(len(chars),d_model=a.d_model,d_ff=a.d_ff,n_layers=a.layers,
        n_heads=a.heads,seq_len=a.seq_len,sphere_dim=a.sphere_dim,
        dropout=a.dropout,seed=a.seed)
    optimizer=Adam(model.parameters(),lr=a.learning_rate)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    config={'program':'hannah_numpy_lab.train_regimen','framework':'NumPy reverse-mode autodiff',
        'optimizer':'Adam','epochs':a.epochs,'file_count':len(files),
        'files':[{'name':f.name,'sha256':sha256(f),'bytes':f.stat().st_size} for f in files],
        'characters_with_separator':len(text),'vocabulary_size':len(chars),
        'windows_per_epoch':windows,'batch_size':a.batch_size,'batches_per_epoch':batches,
        'sequence_length':a.seq_len,'model':{'d_model':a.d_model,'d_ff':a.d_ff,
        'layers':a.layers,'heads':a.heads,'sphere_dim':a.sphere_dim,
        'parameters':sum(p.data.size for p in model.parameters())},
        'learning_rate':a.learning_rate,'dropout':a.dropout,'seed':a.seed,
        'numpy':np.__version__,'python':platform.python_version(),
        'historical_target_match':{'20_files':len(files)==20,'characters_408359':len(text)==408359,
        'vocab_133':len(chars)==133,'batches_25515':batches==25515,
        'warning':'These aggregate checks do not establish corpus identity; compare per-file hashes.'}}
    (a.output_dir/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    metrics=a.output_dir/'metrics.jsonl'
    completed_epochs=set()
    if a.resume and metrics.exists():
        completed_epochs={json.loads(line)['epoch'] for line in metrics.read_text().splitlines()
                          if json.loads(line).get('event')=='epoch'}
    begin_epoch=1; next_batch=0; order=None; epoch_losses=[]; global_step=0; elapsed_before=0.0
    if a.resume:
        with np.load(a.resume,allow_pickle=False) as saved:
            state=json.loads(str(saved['metadata'].item()))
            if state['config_signature']!=config_signature(config):
                raise SystemExit('Resume checkpoint config/corpus does not match this run')
            for i,param in enumerate(model.parameters()): param.data[:]=saved[f'p{i}']
            n=len(model.parameters())
            optimizer.load_state_dict({'t':state['adam_t'],'lr':state['lr'],'beta1':state['beta1'],
                'beta2':state['beta2'],'eps':state['eps'],
                'm':[saved[f'm{i}'] for i in range(n)],'v':[saved[f'v{i}'] for i in range(n)]})
            rng.bit_generator.state=state['shuffle_rng']; model.rng.bit_generator.state=state['model_rng']
            begin_epoch=state['epoch']; next_batch=state['next_batch']; order=saved['order'].copy()
            epoch_losses=saved['epoch_losses'].tolist(); global_step=state['global_step']
            elapsed_before=float(state.get('elapsed_before',0.0))
    mode='a' if a.resume else 'w'
    with metrics.open(mode,encoding='utf-8') as log:
        if not a.resume: log.write(json.dumps({'event':'run_start','unix_time':time.time(),**config})+'\n')
        for epoch in range(begin_epoch,a.epochs+1):
            model.training=True; t0=time.time();
            if order is None:
                order=rng.permutation(windows); epoch_losses=[]; next_batch=0; elapsed_before=0.0
            for bi in range(next_batch+1,batches+1):
                lo=(bi-1)*a.batch_size
                starts=order[lo:lo+a.batch_size]
                x=np.stack([ids[s:s+a.seq_len] for s in starts])
                y=np.stack([ids[s+1:s+a.seq_len+1] for s in starts])
                optimizer.zero_grad(); _,loss=model.forward(x,y); loss.backward(); optimizer.step()
                value=float(loss.data); epoch_losses.append(value); global_step+=1
                active_elapsed=elapsed_before+time.time()-t0
                rec={'event':'batch','epoch':epoch,'batch':bi,'batches_total':batches,
                    'loss':value,'elapsed_s':active_elapsed}
                log.write(json.dumps(rec)+'\n')
                if bi==1 or bi%max(1,batches//100)==0 or bi==batches:
                    log.flush()
                    print(json.dumps(rec),flush=True)
                if a.checkpoint_every_batches and (bi%a.checkpoint_every_batches==0 or bi==batches):
                    save_checkpoint(a.output_dir/'checkpoint_latest.npz',model,optimizer,order,
                        epoch_losses,rng,epoch,bi,global_step,config,active_elapsed)
            model.training=False
            if epoch in completed_epochs:
                np.savez_compressed(a.output_dir/f'checkpoint_epoch_{epoch}.npz',
                    **{f'p{i}':p.data for i,p in enumerate(model.parameters())})
                order=None; next_batch=0; epoch_losses=[]; elapsed_before=0.0
                continue
            summary={'event':'epoch','epoch':epoch,'batches':len(epoch_losses),
                'mean_loss':float(np.mean(epoch_losses)),'first_batch_loss':epoch_losses[0],
                'batch_106_loss':epoch_losses[105] if len(epoch_losses)>=106 else None,
                'last_batch_loss':epoch_losses[-1],'elapsed_s':elapsed_before+time.time()-t0}
            log.write(json.dumps(summary)+'\n'); log.flush()
            np.savez_compressed(a.output_dir/f'checkpoint_epoch_{epoch}.npz',
                **{f'p{i}':p.data for i,p in enumerate(model.parameters())})
            order=None; next_batch=0; epoch_losses=[]; elapsed_before=0.0
            (a.output_dir/f'epoch_{epoch}.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(json.dumps(summary),flush=True)
    # Report historical observations without treating mismatched conditions as a failure.
    records=[json.loads(x) for x in metrics.read_text().splitlines()]
    batches_by_epoch={}
    for rec in records:
        if rec.get('event')=='batch': batches_by_epoch.setdefault(rec['epoch'],{})[rec['batch']]=rec['loss']
    observed={'epoch_1_batch_1':batches_by_epoch.get(1,{}).get(1),
        'epoch_1_batch_106':batches_by_epoch.get(1,{}).get(106),
        'epoch_1_batch_count':next((x['batches'] for x in records if x.get('event')=='epoch' and x['epoch']==1),None),
        'epoch_1_elapsed_s':next((x['elapsed_s'] for x in records if x.get('event')=='epoch' and x['epoch']==1),None),
        'epoch_2_batch_7303':batches_by_epoch.get(2,{}).get(7303),
        'minimum_batch_loss':min((x['loss'] for x in records if x.get('event')=='batch'),default=None)}
    report={'corpus_condition_match':config['historical_target_match'],'observed':observed,
        'observed_epoch_summaries':[x for x in records if x.get('event')=='epoch'],
        'historical_observations':{'batch_1_loss_about':4.86,'batch_106_loss_about':3.06,
            'epoch_1_elapsed_about_s':52975,'epoch_1_batches':25515,
            'later_epoch_2_batch_7303_loss':2.4427,'later_reported_minimum_loss':1.3113},
        'interpretation':'Compare only when corpus hashes, tokenization, model, optimizer, seed/order, and update semantics match. Historical values are reported observations, not guaranteed acceptance thresholds.'}
    (a.output_dir/'historical_comparison.json').write_text(json.dumps(report,indent=2)+'\n')


def config_signature(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def save_checkpoint(path,model,optimizer,order,losses,rng,epoch,next_batch,global_step,config,elapsed_before):
    state=optimizer.state_dict()
    metadata={'config_signature':config_signature(config),'epoch':epoch,'next_batch':next_batch,
        'global_step':global_step,'elapsed_before':elapsed_before,
        'adam_t':state['t'],'lr':state['lr'],'beta1':state['beta1'],
        'beta2':state['beta2'],'eps':state['eps'],'shuffle_rng':rng.bit_generator.state,
        'model_rng':model.rng.bit_generator.state}
    values={f'p{i}':p.data for i,p in enumerate(model.parameters())}
    values.update({f'm{i}':x for i,x in enumerate(state['m'])})
    values.update({f'v{i}':x for i,x in enumerate(state['v'])})
    values.update(order=order,epoch_losses=np.asarray(losses),metadata=np.asarray(json.dumps(metadata)))
    tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**values); tmp.replace(path)

if __name__=='__main__': main()
