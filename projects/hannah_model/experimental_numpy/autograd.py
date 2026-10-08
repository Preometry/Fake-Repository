"""Small NumPy reverse-mode autodiff engine for the Hannah training prototype.

This is intentionally a compact research implementation, not a replacement for
PyTorch. Operations are vectorized NumPy kernels with explicit reverse rules.
"""
from __future__ import annotations
import numpy as np


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and g.shape[axis] != 1:
            g = g.sum(axis=axis, keepdims=True)
    return g.reshape(shape)


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, parents=(), backward=None, requires_grad=False, name=None):
        self.data = np.asarray(data)
        self.parents = tuple(parents)
        self._backward = backward or (lambda: None)
        self.requires_grad = bool(requires_grad or any(p.requires_grad for p in self.parents))
        self.grad = None
        self.name = name

    @staticmethod
    def parameter(data, name=None):
        return Tensor(np.asarray(data, dtype=np.float32), requires_grad=True, name=name)

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        topo, seen = [], set()
        def visit(node):
            if id(node) in seen: return
            seen.add(id(node))
            for parent in node.parents: visit(parent)
            topo.append(node)
        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=self.data.dtype)
        for node in reversed(topo):
            node._backward()

    def _accum(self, value):
        if self.requires_grad:
            value = np.asarray(value, dtype=self.data.dtype)
            self.grad = value.copy() if self.grad is None else self.grad + value

    def __add__(self, other):
        other = as_tensor(other); out = Tensor(self.data + other.data, (self, other))
        def backward():
            self._accum(_unbroadcast(out.grad, self.data.shape))
            other._accum(_unbroadcast(out.grad, other.data.shape))
        out._backward = backward; return out
    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, (self,))
        out._backward = lambda: self._accum(-out.grad)
        return out
    def __sub__(self, other): return self + (-as_tensor(other))
    def __rsub__(self, other): return as_tensor(other) - self

    def __mul__(self, other):
        other = as_tensor(other); out = Tensor(self.data * other.data, (self, other))
        def backward():
            self._accum(_unbroadcast(out.grad * other.data, self.data.shape))
            other._accum(_unbroadcast(out.grad * self.data, other.data.shape))
        out._backward = backward; return out
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_tensor(other); out = Tensor(self.data / other.data, (self, other))
        def backward():
            self._accum(_unbroadcast(out.grad / other.data, self.data.shape))
            other._accum(_unbroadcast(-out.grad * self.data / (other.data ** 2), other.data.shape))
        out._backward = backward; return out
    def __rtruediv__(self, other): return as_tensor(other) / self

    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,))
        out._backward = lambda: self._accum(out.grad * power * self.data ** (power - 1))
        return out

    def __matmul__(self, other):
        other = as_tensor(other); out = Tensor(np.matmul(self.data, other.data), (self, other))
        def backward():
            ga = np.matmul(out.grad, np.swapaxes(other.data, -1, -2))
            gb = np.matmul(np.swapaxes(self.data, -1, -2), out.grad)
            self._accum(_unbroadcast(ga, self.data.shape))
            other._accum(_unbroadcast(gb, other.data.shape))
        out._backward = backward; return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))
        def backward():
            g = out.grad
            if axis is not None and not keepdims:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                for ax in sorted((a % self.data.ndim for a in axes)): g = np.expand_dims(g, ax)
            self._accum(np.broadcast_to(g, self.data.shape))
        out._backward = backward; return out

    def mean(self, axis=None, keepdims=False):
        if axis is None: count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.data.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)): shape = tuple(shape[0])
        out = Tensor(self.data.reshape(*shape), (self,))
        out._backward = lambda: self._accum(out.grad.reshape(self.data.shape))
        return out

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(*axes), (self,))
        inverse = np.argsort(axes)
        out._backward = lambda: self._accum(out.grad.transpose(*inverse))
        return out
    @property
    def T(self): return self.transpose()

    def __getitem__(self, index):
        out = Tensor(self.data[index], (self,))
        def backward():
            g = np.zeros_like(self.data)
            np.add.at(g, index, out.grad)
            self._accum(g)
        out._backward = backward; return out

    def exp(self):
        value = np.exp(self.data); out = Tensor(value, (self,))
        out._backward = lambda: self._accum(out.grad * value)
        return out
    def log(self):
        out = Tensor(np.log(self.data), (self,))
        out._backward = lambda: self._accum(out.grad / self.data)
        return out
    def tanh(self):
        value = np.tanh(self.data); out = Tensor(value, (self,))
        out._backward = lambda: self._accum(out.grad * (1 - value * value))
        return out
    def sqrt(self):
        value = np.sqrt(self.data); out = Tensor(value, (self,))
        out._backward = lambda: self._accum(out.grad * (0.5 / value))
        return out
    def acos(self):
        value = np.arccos(self.data); out = Tensor(value, (self,))
        out._backward = lambda: self._accum(-out.grad / np.sqrt(np.maximum(1 - self.data**2, 1e-20)))
        return out
    def clip(self, low, high):
        value = np.clip(self.data, low, high); out = Tensor(value, (self,))
        mask = (self.data > low) & (self.data < high)
        out._backward = lambda: self._accum(out.grad * mask)
        return out
    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z); value = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(value, (self,))
        out._backward = lambda: self._accum(value * (out.grad - (out.grad * value).sum(axis=axis, keepdims=True)))
        return out

    def gelu(self):
        # Vectorized tanh approximation, used as a practical NumPy equivalent.
        c = np.sqrt(2 / np.pi)
        u = c * (self.data + 0.044715 * self.data**3)
        t = np.tanh(u)
        value = 0.5 * self.data * (1 + t)
        du = c * (1 + 3 * 0.044715 * self.data**2)
        deriv = 0.5 * (1 + t) + 0.5 * self.data * (1 - t*t) * du
        out = Tensor(value, (self,))
        out._backward = lambda: self._accum(out.grad * deriv)
        return out

    def __repr__(self): return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"


def as_tensor(x): return x if isinstance(x, Tensor) else Tensor(x)


def concat(values, axis=-1):
    values=[as_tensor(v) for v in values]
    out=Tensor(np.concatenate([v.data for v in values],axis=axis),tuple(values))
    cuts=np.cumsum([v.data.shape[axis] for v in values])[:-1]
    def backward():
        pieces=np.split(out.grad,cuts,axis=axis)
        for value,piece in zip(values,pieces): value._accum(piece)
    out._backward=backward
    return out


class Adam:
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = list(parameters); self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self.m=[np.zeros_like(p.data) for p in self.parameters]
        self.v=[np.zeros_like(p.data) for p in self.parameters]; self.t=0
    def step(self):
        self.t += 1
        for i,p in enumerate(self.parameters):
            if p.grad is None: continue
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*p.grad
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(p.grad*p.grad)
            mh=self.m[i]/(1-self.b1**self.t); vh=self.v[i]/(1-self.b2**self.t)
            p.data -= self.lr*mh/(np.sqrt(vh)+self.eps)
    def zero_grad(self):
        for p in self.parameters: p.zero_grad()
    def state_dict(self):
        return {'t':self.t,'m':[x.copy() for x in self.m],
                'v':[x.copy() for x in self.v], 'lr':self.lr,
                'beta1':self.b1,'beta2':self.b2,'eps':self.eps}
    def load_state_dict(self, state):
        if len(state['m'])!=len(self.parameters) or len(state['v'])!=len(self.parameters):
            raise ValueError('Adam checkpoint parameter count does not match model')
        for param, m, v in zip(self.parameters,state['m'],state['v']):
            if param.data.shape!=m.shape or param.data.shape!=v.shape:
                raise ValueError('Adam checkpoint parameter shape does not match model')
        self.t=int(state['t']); self.lr=float(state['lr'])
        self.b1=float(state['beta1']); self.b2=float(state['beta2']); self.eps=float(state['eps'])
        self.m=[x.copy() for x in state['m']]; self.v=[x.copy() for x in state['v']]


def cross_entropy(logits: Tensor, targets):
    """Stable mean cross entropy with an explicit reverse rule."""
    targets=np.asarray(targets,dtype=np.int64)
    flat=logits.data.reshape(-1,logits.data.shape[-1]); y=targets.reshape(-1)
    shifted=flat-flat.max(axis=-1,keepdims=True)
    ex=np.exp(shifted); probs=ex/ex.sum(axis=-1,keepdims=True)
    loss=float(-np.log(np.maximum(probs[np.arange(len(y)),y],1e-30)).mean())
    out=Tensor(np.asarray(loss,dtype=logits.data.dtype),(logits,))
    grad=probs.copy(); grad[np.arange(len(y)),y]-=1; grad/=len(y)
    def backward(): logits._accum(out.grad*grad.reshape(logits.data.shape))
    out._backward=backward
    return out
