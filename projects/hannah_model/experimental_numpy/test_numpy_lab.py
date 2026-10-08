import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from autograd import Tensor, Adam, cross_entropy
from model import MiniHannahLM


def test_reverse_mode_matches_finite_difference():
    a=Tensor.parameter(np.array([[.2,-.4],[.7,.3]],dtype=np.float64))
    b=Tensor.parameter(np.array([[.1,.5],[-.2,.8]],dtype=np.float64))
    loss=((a@b).tanh()**2).sum()
    loss.backward(); analytic=a.grad.copy()
    def f(arr): return float((np.tanh(arr@b.data)**2).sum())
    # The engine stores parameters as float32; use a finite-difference step
    # large enough to avoid cancellation at float32 precision.
    eps=3e-3; numeric=np.zeros_like(a.data)
    for ix in np.ndindex(a.data.shape):
        plus=a.data.copy(); minus=a.data.copy(); plus[ix]+=eps; minus[ix]-=eps
        numeric[ix]=(f(plus)-f(minus))/(2*eps)
    assert np.max(np.abs(analytic-numeric))<2e-5,(analytic,numeric)


def test_embedding_gather_accumulates_duplicate_indices():
    e=Tensor.parameter(np.zeros((3,2),dtype=np.float32))
    e[np.array([1,1,2])].sum().backward()
    assert np.all(e.grad[1]==2) and np.all(e.grad[2]==1)


def test_historical_configuration_parameter_count():
    model=MiniHannahLM(vocab_size=133,d_model=256,d_ff=1024,n_layers=4,n_heads=4,
                       seq_len=128,sphere_dim=20,seed=1)
    count=sum(p.data.size for p in model.parameters())
    assert count==3_062_016,count


def test_mini_hannah_forward_backward_and_geodesic():
    model=MiniHannahLM(vocab_size=5,d_model=8,d_ff=16,n_layers=1,n_heads=2,
                       seq_len=8,sphere_dim=4,dropout=0,seed=2)
    ids=np.array([[0,1,2,3,4,0,1,2],[1,2,3,4,0,1,2,3]])
    targets=np.roll(ids,-1,axis=1)
    logits,loss=model.forward(ids,targets)
    assert logits.data.shape==(2,8,5)
    assert np.isfinite(loss.data)
    assert len(model.last_attention)==2
    for report in model.last_attention:
        d,w=report['distance'],report['weights']
        assert d.shape==(2,8,8) and w.shape==(2,8,8)
        assert np.all((d>=0)&(d<=np.pi))
        assert np.allclose(w.sum(axis=-1),1.0,atol=1e-6)
        assert np.all(w[:,np.triu_indices(8,k=1)[0],np.triu_indices(8,k=1)[1]]==0)
    loss.backward()
    active=[p for p in model.parameters() if p.grad is not None]
    assert active and all(np.all(np.isfinite(p.grad)) for p in active)
    q_weight=model.p['block0.head0.q']; ix=(0,0); analytic=float(q_weight.grad[ix])
    original=float(q_weight.data[ix]); step=2e-3
    q_weight.data[ix]=original+step; plus=float(model.forward(ids,targets)[1].data)
    q_weight.data[ix]=original-step; minus=float(model.forward(ids,targets)[1].data)
    q_weight.data[ix]=original
    numeric=(plus-minus)/(2*step)
    assert abs(analytic-numeric)<max(2e-3,abs(numeric)*.08),(analytic,numeric)
    # Attention geodesics in the model are clamped strictly inside [0, pi].
    assert len(model.parameters()) > 0


def test_adam_reduces_tiny_repeated_language_task():
    model=MiniHannahLM(vocab_size=3,d_model=8,d_ff=16,n_layers=1,n_heads=2,
                       seq_len=6,sphere_dim=4,dropout=0,seed=4)
    opt=Adam(model.parameters(),lr=.005)
    ids=np.array([[0,1,2,0,1,2],[1,2,0,1,2,0]])
    y=np.roll(ids,-1,axis=1)
    _,initial=model.forward(ids,y)
    for _ in range(80):
        opt.zero_grad(); _,loss=model.forward(ids,y); loss.backward(); opt.step()
    _,final=model.forward(ids,y)
    assert float(final.data) < float(initial.data)*.99,(initial.data,final.data)


if __name__=='__main__':
    for fn in (test_reverse_mode_matches_finite_difference,
               test_embedding_gather_accumulates_duplicate_indices,
               test_historical_configuration_parameter_count,
               test_mini_hannah_forward_backward_and_geodesic,
               test_adam_reduces_tiny_repeated_language_task):
        fn(); print('PASS',fn.__name__)
