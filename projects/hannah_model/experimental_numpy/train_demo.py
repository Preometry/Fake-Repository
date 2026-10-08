#!/usr/bin/env python3
"""Train the standalone NumPy Hannah mini-model on theorem text."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from autograd import Adam
from model import MiniHannahLM


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('numpy_run'))
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--limit-files',type=int,default=20)
    p.add_argument('--steps-per-epoch',type=int,default=100,
                   help='Use 0 to consume every possible window once; default keeps the first run short.')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seq-len',type=int,default=16)
    p.add_argument('--d-model',type=int,default=16)
    p.add_argument('--d-ff',type=int,default=32)
    p.add_argument('--layers',type=int,default=1)
    p.add_argument('--heads',type=int,default=2)
    p.add_argument('--sphere-dim',type=int,default=8)
    p.add_argument('--learning-rate',type=float,default=1e-3)
    p.add_argument('--seed',type=int,default=20261007)
    a=p.parse_args()
    rng=np.random.default_rng(a.seed)
    files=sorted(a.data_dir.glob('THRM-*.yaml'))[:a.limit_files]
    if not files: raise SystemExit(f'No THRM-*.yaml files in {a.data_dir}')
    text=''.join(f.read_text(encoding='utf-8')+'\n' for f in files)
    vocab=sorted(set(text)); c2i={c:i for i,c in enumerate(vocab)}
    ids=np.fromiter((c2i[c] for c in text),dtype=np.int64,count=len(text))
    windows=len(ids)-a.seq_len
    if windows < a.batch_size: raise SystemExit('Corpus is too short for selected sequence/batch sizes')
    steps=(windows+a.batch_size-1)//a.batch_size if a.steps_per_epoch==0 else a.steps_per_epoch
    model=MiniHannahLM(len(vocab),d_model=a.d_model,d_ff=a.d_ff,n_layers=a.layers,
        n_heads=a.heads,seq_len=a.seq_len,sphere_dim=a.sphere_dim,seed=a.seed)
    opt=Adam(model.parameters(),lr=a.learning_rate)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    run={'files':[{'path':str(f),'bytes':f.stat().st_size} for f in files],
         'characters':len(text),'vocab_size':len(vocab),'parameters':sum(p.data.size for p in model.parameters()),
         'config':vars(a)|{'data_dir':str(a.data_dir)},'batches_per_epoch':steps,
         'note':'NumPy mini-model; dimensions intentionally differ from historical 256D, 4-layer model.'}
    (a.output_dir/'run.json').write_text(json.dumps(run,indent=2,default=str)+'\n')
    with (a.output_dir/'metrics.jsonl').open('w') as log:
        for epoch in range(a.epochs):
            model.training=True; started=time.time(); vals=[]
            for step in range(steps):
                starts=rng.integers(0,windows,size=a.batch_size)
                x=np.stack([ids[s:s+a.seq_len] for s in starts])
                y=np.stack([ids[s+1:s+a.seq_len+1] for s in starts])
                opt.zero_grad(); _,loss=model.forward(x,y); loss.backward(); opt.step()
                val=float(loss.data); vals.append(val)
                log.write(json.dumps({'epoch':epoch+1,'step':step+1,'loss':val})+'\n')
            model.training=False
            result={'epoch':epoch+1,'steps':steps,'mean_loss':float(np.mean(vals)),
                    'first_loss':vals[0],'last_loss':vals[-1],'elapsed_s':time.time()-started}
            print(json.dumps(result)); log.write(json.dumps(result)+'\n'); log.flush()
            np.savez_compressed(a.output_dir/f'checkpoint_epoch_{epoch+1}.npz',
                **{f'p{i}':p.data for i,p in enumerate(model.parameters())})


if __name__=='__main__': main()
