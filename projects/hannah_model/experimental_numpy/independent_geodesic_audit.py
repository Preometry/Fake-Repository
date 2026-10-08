#!/usr/bin/env python3
"""Independent geometric check of the model's spherical attention distances."""
import numpy as np
from model import MiniHannahLM


def main():
    model=MiniHannahLM(vocab_size=7,d_model=8,d_ff=16,n_layers=1,n_heads=2,
        seq_len=6,sphere_dim=5,dropout=0,seed=23)
    model.capture_geodesic_vectors=True
    # Strengthen this audit-only instance so its geodesic path has a measurable
    # gradient; this is not the training initialization or a benchmark setup.
    rng=np.random.default_rng(12)
    for name in ('token_embedding','hannah_channel','hannah_position','hannah_winding',
        'remainder_projection.weight','channel_mixer.weight','block0.head0.q','block0.head0.k',
        'block0.head0.v','block0.head0.o','block0.head1.q','block0.head1.k',
        'block0.head1.v','block0.head1.o','block0.attention_output','block0.ffn1','block0.ffn2'):
        model.p[name].data[:]=rng.normal(0,.4,size=model.p[name].data.shape)
    ids=np.array([[0,1,2,3,4,5],[5,4,3,2,1,0]])
    targets=np.roll(ids,-1,axis=1)
    _,loss=model.forward(ids,targets)
    assert np.isfinite(float(loss.data))
    upper=np.triu_indices(ids.shape[1],k=1)
    for head in model.last_attention:
        q,k=head['q_unit'],head['k_unit']
        # Independent atan2 angle formula, avoiding the model's acos implementation.
        dot=np.einsum('btd,bsd->bts',q,k)
        dot=np.clip(dot,-1.0+1e-7,1.0-1e-7)
        independently_computed=np.arctan2(np.sqrt(np.maximum(0.0,1.0-dot*dot)),dot)
        assert np.allclose(np.linalg.norm(q,axis=-1),1,atol=2e-6)
        assert np.allclose(np.linalg.norm(k,axis=-1),1,atol=2e-6)
        assert np.allclose(independently_computed,head['distance'],atol=2e-6)
        weights=head['weights']
        assert np.allclose(weights.sum(axis=-1),1,atol=1e-6)
        assert np.all(weights[:,upper[0],upper[1]]==0)
        assert np.all((independently_computed>=0)&(independently_computed<=np.pi))
    # Independent central-difference check of a geodesic attention projection gradient.
    loss.backward()
    param=model.p['block0.head0.q']; idx=(0,0); analytic=float(param.grad[idx])
    saved=float(param.data[idx]); eps=2e-3
    param.data[idx]=saved+eps; plus=float(model.forward(ids,targets)[1].data)
    param.data[idx]=saved-eps; minus=float(model.forward(ids,targets)[1].data)
    param.data[idx]=saved
    numeric=(plus-minus)/(2*eps)
    assert abs(numeric)>1e-6,(analytic,numeric)
    assert abs(analytic-numeric)<max(1e-7,abs(numeric)*.02),(analytic,numeric)
    print({'result':'PASS','attention_heads':len(model.last_attention),
           'gradient_analytic':analytic,'gradient_finite_difference':numeric})


if __name__=='__main__': main()
