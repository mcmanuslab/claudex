"""Thin array-backend shim.

The reference implementation is written once, against this shim, so the same
vectorised code runs on NumPy (portable, used for tests and for development on
non-Apple hardware) and on MLX (the target for the M3 Ultra).  DESIGN.md 7.3
treats the backend choice as a hypothesis to be measured, not a conclusion, so
it has to be swappable.

Only the operations the rollout actually needs are exposed.  Anything that
would force a host synchronisation inside the rollout loop is deliberately
absent.
"""

from __future__ import annotations

import os

_BACKEND = os.environ.get("NEMO_BACKEND", "numpy").lower()

if _BACKEND == "mlx":  # pragma: no cover - requires Apple Silicon
    import mlx.core as _xp

    NAME = "mlx"

    def to_host(x):
        import numpy as _np

        return _np.array(x)

    def eval_(*xs):
        _xp.eval(*xs)

    def compile_(fn):
        return _xp.compile(fn)

    def default_float():
        return _xp.float32

else:
    import numpy as _xp

    NAME = "numpy"

    def to_host(x):
        return _xp.asarray(x)

    def eval_(*xs):  # numpy is eager
        return None

    def compile_(fn):
        return fn

    def default_float():
        return _xp.float32


xp = _xp

# Re-exports used across the codebase, so call sites never touch `xp` directly
# for anything whose MLX spelling differs from NumPy's.
array = _xp.array
zeros = _xp.zeros
arange = _xp.arange
concatenate = _xp.concatenate
maximum = _xp.maximum
sqrt = _xp.sqrt
matmul = _xp.matmul
reshape = _xp.reshape
float32 = _xp.float32
int32 = _xp.int32


def softmax(x, axis=-1):
    m = _xp.max(x, axis=axis, keepdims=True)
    e = _xp.exp(x - m)
    return e / _xp.sum(e, axis=axis, keepdims=True)


def sigmoid(x):
    # Numerically stable on both backends without branching on host values.
    return 0.5 * (1.0 + _xp.tanh(0.5 * x))


def relu(x):
    """One vectorised pass.

    Measured on the reference NumPy backend: a tanh-based GELU cost 11% of
    total rollout time, and `where(x>0, x, x*slope)` cost 7.8 ms per round --
    three passes over the activation plus two temporaries -- against 1.0 ms for
    `maximum(x, 0)`.  In a 16-wide evolved module neither the GELU curve nor
    the leaky slope earns that.  Dead units are handled by evolution: a module
    that stops contributing is deleted or its weights mutate.

    Note for the MLX backend: this gap is a NumPy limitation, not a property of
    the math.  MLX fuses elementwise chains under `mx.compile`, so the same
    expression costs far less there.  Do not read the NumPy timings as a
    prediction of MLX timings.
    """
    return _xp.maximum(x, 0.0)


def rms_norm(x, scale, eps: float = 1e-5):
    ms = _xp.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / _xp.sqrt(ms + eps)) * scale


def gather_slots(buf, idx):
    """buf: (L, n_slots, E, d);  idx: (L, G, K) -> (L, G, K, E, d).

    Implemented as ONE flat gather on a 2-D view rather than take_along_axis on
    a broadcast index.  The obvious take_along_axis spelling has to materialise
    an (L, G*K, E, d) index array -- 33 MB of int64 at pilot size -- and
    benchmarked at ~110x the cost of the module GEMMs it feeds.  Flattening the
    lane and slot axes turns it into a single (L*G*K,) index into an
    (L*n_slots, E*d) array, which both NumPy and MLX gather efficiently.
    """
    L, S, E, d = buf.shape
    G, K = idx.shape[1], idx.shape[2]
    flat_buf = buf.reshape(L * S, E * d)
    lane_off = (_xp.arange(L, dtype=idx.dtype) * S).reshape(L, 1, 1)
    flat_idx = (lane_off + idx).reshape(L * G * K)
    out = _xp.take(flat_buf, flat_idx, axis=0)
    return out.reshape(L, G, K, E, d)


def set_slots(buf, values, start: int, count: int):
    """Write `values` (L, count, E, d) into slots [start, start+count).

    Slice assignment rather than a three-way concatenate: the concatenate
    spelling copies the whole message buffer three times per round, four rounds
    per timestep.  Both backends support `a[:, i:j] = v`.

    MLX caveat: this mutates `buf`, and in-place mutation of a traced array is
    not generally safe inside `mx.compile`.  If the rollout is compiled, replace
    the body with the functional form below -- measure before assuming the
    concatenate cost matters there, since MLX fuses where NumPy cannot:

        return concatenate([buf[:, :start], values, buf[:, start+count:]], 1)
    """
    buf[:, start:start + count] = values
    return buf


def broadcast_to(x, shape):
    return _xp.broadcast_to(x, shape)


def max_(x, axis=None, keepdims=False):
    return _xp.max(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=-1):
    return _xp.argmax(x, axis=axis)


def clip(x, lo, hi):
    return _xp.clip(x, lo, hi)


def tanh(x):
    return _xp.tanh(x)
