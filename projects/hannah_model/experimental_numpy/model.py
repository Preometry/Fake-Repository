"""Small Hannah trust-transformer implemented only with NumPy operations."""
from __future__ import annotations
import numpy as np
from autograd import Tensor, concat, cross_entropy

R_VALID=tuple(r for r in range(1,46,2) if r!=23)


def _weight(rng, shape, scale=0.02):
    return Tensor.parameter(rng.normal(0,scale,size=shape).astype(np.float32))


def _layer_norm(x, gamma, beta, eps=1e-5):
    centered=x-x.mean(axis=-1,keepdims=True)
    variance=(centered*centered).mean(axis=-1,keepdims=True)
    return centered/(variance+eps).sqrt()*gamma+beta


class MiniHannahLM:
    """One or more original-style blocks at configurable miniature dimensions.

    It retains learned token and HANNAH channel/position/winding embeddings,
    remainder projection and trust mixing, spherical geodesic attention,
    residual FFNs, final layer norm, and tied token/output weights.
    """
    def __init__(self,vocab_size,d_model=16,d_ff=32,n_layers=1,n_heads=2,
                 seq_len=16,sphere_dim=8,temperature=1.0,dropout=0.1,seed=7):
        assert d_model % n_heads==0
        self.vocab_size=vocab_size; self.d_model=d_model; self.d_ff=d_ff
        self.n_layers=n_layers; self.n_heads=n_heads; self.seq_len=seq_len
        self.sphere_dim=sphere_dim; self.temperature=temperature; self.dropout=dropout
        self.rng=np.random.default_rng(seed); self.training=True; self.p={}
        self.capture_geodesic_vectors=False
        self.last_attention=[]
        def w(name,shape,scale=.02): self.p[name]=_weight(self.rng,shape,scale); return self.p[name]
        def z(name,shape): self.p[name]=Tensor.parameter(np.zeros(shape,dtype=np.float32)); return self.p[name]
        self.tok=w('token_embedding',(vocab_size,d_model))
        self.channel=w('hannah_channel',(22,d_model)); self.position=w('hannah_position',(seq_len,d_model))
        self.winding=w('hannah_winding',(10,d_model)); self.rem_w=w('remainder_projection.weight',(d_model,d_model)); self.rem_b=z('remainder_projection.bias',(d_model,))
        self.mix_w=w('channel_mixer.weight',(d_model,d_model)); self.mix_b=z('channel_mixer.bias',(d_model,))
        # Static trust-distance matrix exactly follows the recovered cyclic 22-channel rule.
        idx=np.arange(seq_len)%22
        d=np.abs(idx[:,None]-idx[None,:]); d=np.minimum(d,22-d)*np.pi/22
        trust=np.exp(-(d*d)); self.trust_matrix=trust/(trust.sum(axis=-1,keepdims=True)+1e-8)
        self.blocks=[]
        for layer in range(n_layers):
            b={'ln1_g':Tensor.parameter(np.ones(d_model,dtype=np.float32)),
               'ln1_b':Tensor.parameter(np.zeros(d_model,dtype=np.float32)),
               'ln2_g':Tensor.parameter(np.ones(d_model,dtype=np.float32)),
               'ln2_b':Tensor.parameter(np.zeros(d_model,dtype=np.float32)),
               'attn_heads':[]}
            for head in range(n_heads):
                b['attn_heads'].append({
                    'q':w(f'block{layer}.head{head}.q',(d_model,sphere_dim)),
                    'k':w(f'block{layer}.head{head}.k',(d_model,sphere_dim)),
                    'v':w(f'block{layer}.head{head}.v',(d_model,d_model//n_heads)),
                    'o':w(f'block{layer}.head{head}.o',(d_model//n_heads,d_model//n_heads))})
            b['attn_out']=w(f'block{layer}.attention_output',(d_model,d_model))
            b['ff1']=w(f'block{layer}.ffn1',(d_model,d_ff)); b['ff2']=w(f'block{layer}.ffn2',(d_ff,d_model))
            self.blocks.append(b)
        self.final_g=Tensor.parameter(np.ones(d_model,dtype=np.float32)); self.final_b=Tensor.parameter(np.zeros(d_model,dtype=np.float32))

    def parameters(self): return list(self.p.values())+[v for b in self.blocks for v in (b['ln1_g'],b['ln1_b'],b['ln2_g'],b['ln2_b'])]+[self.final_g,self.final_b]

    def _drop(self,x):
        if not self.training or self.dropout<=0: return x
        keep=1.0-self.dropout; mask=(self.rng.random(x.data.shape)<keep).astype(np.float32)/keep
        return x*mask

    def _attention(self,x,b,causal):
        outputs=[]
        for h in b['attn_heads']:
            q=x@h['q']; k=x@h['k']; v=x@h['v']
            # Epsilon inside the squared norm preserves unit length for normal
            # vectors while keeping the zero-vector case finite.
            q=q/((q*q).sum(axis=-1,keepdims=True)+1e-16).sqrt()
            k=k/((k*k).sum(axis=-1,keepdims=True)+1e-16).sqrt()
            similarity=q@k.transpose(0,2,1)
            distance=similarity.clip(-1+1e-7,1-1e-7).acos()
            logits=-(distance*distance)/self.temperature
            logits=logits+np.where(causal,0.0,-1e9).astype(np.float32)
            weights=logits.softmax(axis=-1)
            record={'distance':distance.data.copy(),'weights':weights.data.copy()}
            if self.capture_geodesic_vectors:
                record.update(q_unit=q.data.copy(),k_unit=k.data.copy())
            self.last_attention.append(record)
            outputs.append((weights@v)@h['o'])
        return concat(outputs,axis=-1)@b['attn_out']

    def forward(self,ids,targets=None):
        ids=np.asarray(ids,dtype=np.int64); batch,t=ids.shape
        self.last_attention=[]
        if t>self.seq_len: raise ValueError(f"sequence length {t} exceeds {self.seq_len}")
        positions=np.arange(t); channel=positions%22
        # Local training windows use consecutive positions, so recovered winding_number is zero.
        x=self.tok[ids]+self.channel[channel][None,:,:]+self.position[positions][None,:,:]+self.winding[np.zeros(t,dtype=np.int64)][None,:,:]
        x=self._drop(x@self.rem_w+self.rem_b)
        trust=self.trust_matrix[:t,:t]
        x=(Tensor(trust[None,:,:])@x)@self.mix_w+self.mix_b
        causal=np.tril(np.ones((1,t,t),dtype=bool))
        for b in self.blocks:
            x=x+self._attention(x,b,causal)
            norm=_layer_norm(x,b['ln2_g'],b['ln2_b'])
            hidden=(norm@b['ff1']).gelu()
            x=x+self._drop(hidden@b['ff2'])
        x=_layer_norm(x,self.final_g,self.final_b)
        logits=x@self.tok.T
        loss=cross_entropy(logits,targets) if targets is not None else None
        return logits,loss
