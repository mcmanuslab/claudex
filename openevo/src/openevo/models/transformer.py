"""Tiny causal transformer, batched over a whole population bucket.

Every array carries a leading ``P`` axis indexing organisms that share an architecture
signature. At 5K parameters the cost of evaluating one organism is dominated by
per-call dispatch overhead rather than arithmetic (roughly three orders of magnitude
of headroom), so the only way to make the experiment affordable is to evaluate many
organisms inside a single call. Organisms are therefore bucketed by
``ArchGenome.signature()`` and each bucket runs as one batched graph.

Two execution paths:

* :func:`rollout_step` -- incremental, KV-cached. Used while acting in a world, where
  step ``t``'s action affects step ``t+1``'s observation and so cannot be batched over
  time. Sequential in ``T`` but batched over ``(organisms x episodes)``.
* :func:`forward_full` / :func:`backward` -- teacher-forced over the recorded
  trajectory. Used by the gradient ("slow", in-weights) learning channel, where the
  actions are already fixed and the whole sequence can go in one pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .genome import ArchGenome

DTYPE = np.float32
RMS_EPS = 1e-6
Params = dict[str, np.ndarray]

# --------------------------------------------------------------------------- init


def init_params(
    g: ArchGenome, rng: np.random.Generator, n: int = 1, scale: float = 1.0
) -> Params:
    """Initialise `n` organisms of architecture `g`.

    Output projections (``Wo``, ``W2``) start at zero, making each block an exact
    identity at birth. This matches how :mod:`openevo.models.morphisms` introduces new
    blocks, so a newborn ancestor and a newly grown block are on the same footing.
    """
    d, da, f, A = g.d_model, g.d_attn, g.d_ff, g.n_act

    def normal(*shape: int, std: float) -> np.ndarray:
        return (rng.normal(0.0, std * scale, size=(n, *shape))).astype(DTYPE)

    p: Params = {
        "E_obs": normal(g.n_obs, d, std=0.5),
        "E_act": normal(g.n_act, d, std=0.5),
        "E_rew": normal(g.n_rew, d, std=0.5),
        "g_final": np.ones((n, d), dtype=DTYPE),
        "W_head": normal(d, A, std=1.0 / np.sqrt(d)),
        "b_head": np.zeros((n, A), dtype=DTYPE),
    }
    for b in range(g.n_blocks):
        p[f"b{b}.g1"] = np.ones((n, d), dtype=DTYPE)
        p[f"b{b}.Wq"] = normal(d, da, std=1.0 / np.sqrt(d))
        p[f"b{b}.Wk"] = normal(d, da, std=1.0 / np.sqrt(d))
        p[f"b{b}.Wv"] = normal(d, da, std=1.0 / np.sqrt(d))
        p[f"b{b}.Wo"] = np.zeros((n, da, d), dtype=DTYPE)
        p[f"b{b}.g2"] = np.ones((n, d), dtype=DTYPE)
        p[f"b{b}.W1"] = normal(d, f, std=1.0 / np.sqrt(d))
        p[f"b{b}.W2"] = np.zeros((n, f, d), dtype=DTYPE)
    return p


def param_keys(g: ArchGenome) -> list[str]:
    base = ["E_obs", "E_act", "E_rew", "g_final", "W_head", "b_head"]
    for b in range(g.n_blocks):
        base += [f"b{b}.{k}" for k in ("g1", "Wq", "Wk", "Wv", "Wo", "g2", "W1", "W2")]
    return base


def count_params(p: Params) -> int:
    """Parameter count of a single organism inside a (possibly batched) dict."""
    return int(sum(int(np.prod(v.shape[1:])) for v in p.values()))


def unstack(p: Params, i: int) -> Params:
    return {k: np.array(v[i], dtype=DTYPE) for k, v in p.items()}


def stack(items: list[Params]) -> Params:
    return {k: np.stack([it[k] for it in items]).astype(DTYPE) for k in items[0]}


def add_batch_axis(p: Params) -> Params:
    return {k: v[None].astype(DTYPE) for k, v in p.items()}


def nbytes(p: Params) -> int:
    return int(sum(v.nbytes for v in p.values()))


# ------------------------------------------------------------------------ pieces


def rmsnorm(x: np.ndarray, gain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """gain shape (P, d); x shape (P, ..., d). Returns (y, inv_r)."""
    ms = np.mean(np.square(x), axis=-1, keepdims=True) + RMS_EPS
    inv_r = 1.0 / np.sqrt(ms)
    gshape = (gain.shape[0],) + (1,) * (x.ndim - 2) + (gain.shape[1],)
    return (x * inv_r * gain.reshape(gshape)).astype(DTYPE), inv_r.astype(DTYPE)


def rmsnorm_backward(
    dy: np.ndarray, x: np.ndarray, inv_r: np.ndarray, gain: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    d = x.shape[-1]
    gshape = (gain.shape[0],) + (1,) * (x.ndim - 2) + (gain.shape[1],)
    g = gain.reshape(gshape)
    u = dy * g
    dot = np.sum(u * x, axis=-1, keepdims=True)
    dx = u * inv_r - x * (dot * inv_r**3 / d)
    axes = tuple(range(1, x.ndim - 1))
    dgain = np.sum(dy * x * inv_r, axis=axes)
    return dx.astype(DTYPE), dgain.astype(DTYPE)


_GELU_C = np.float32(np.sqrt(2.0 / np.pi))


def gelu(x: np.ndarray) -> np.ndarray:
    inner = _GELU_C * (x + 0.044715 * x**3)
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(DTYPE)


def gelu_backward(dy: np.ndarray, x: np.ndarray) -> np.ndarray:
    inner = _GELU_C * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    dinner = _GELU_C * (1.0 + 3 * 0.044715 * x**2)
    return (dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t**2) * dinner)).astype(DTYPE)


def _radical_inverse(n: int) -> np.ndarray:
    """Van der Corput sequence in base 2: 0, 1/2, 1/4, 3/4, 1/8, 5/8, ..."""
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v, denom, k = 0.0, 0.5, i
        while k:
            v += (k & 1) * denom
            denom *= 0.5
            k >>= 1
        out[i] = v
    return out


def rope_tables(seq_len: int, head_dim: int, base: float = 10000.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """RoPE tables whose frequency *set* nests as head_dim grows.

    Standard RoPE assigns pair ``i`` of ``half`` the exponent ``i/half``, so changing
    head_dim re-assigns the frequency of every existing pair and silently rotates the
    whole query/key space -- which would make ``widen_head`` impossible to make
    function-preserving. Ordering the exponents by the van der Corput sequence instead
    means the exponent set for ``half`` rungs is an exact *prefix* of the set for any
    larger ``half``: growing head_dim leaves existing pairs untouched and only appends
    new frequencies, while still covering the [0, 1] exponent range at every size.
    """
    half = head_dim // 2
    inv = base ** (-_radical_inverse(half))
    ang = np.arange(seq_len, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(DTYPE), np.sin(ang).astype(DTYPE)


def rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, inverse: bool = False
               ) -> np.ndarray:
    """x shape (..., T, hd); cos/sin shape (T, hd/2)."""
    a, b = x[..., 0::2], x[..., 1::2]
    c = cos.reshape((1,) * (x.ndim - 2) + cos.shape)
    s = sin.reshape((1,) * (x.ndim - 2) + sin.shape)
    if inverse:
        s = -s
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(DTYPE)


def softmax_backward(dp: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p * (dp - np.sum(dp * p, axis=-1, keepdims=True))).astype(DTYPE)


def embed(p: Params, obs: np.ndarray, prev_act: np.ndarray, prev_rew: np.ndarray
          ) -> np.ndarray:
    """obs/prev_act/prev_rew are int arrays shape (P, B, T) -> (P, B, T, d)."""
    P = obs.shape[0]
    idx = np.arange(P)[:, None, None]
    return (
        p["E_obs"][idx, obs] + p["E_act"][idx, prev_act] + p["E_rew"][idx, prev_rew]
    ).astype(DTYPE)


# ----------------------------------------------------------------- full forward


def forward_full(
    p: Params, g: ArchGenome, obs: np.ndarray, prev_act: np.ndarray,
    prev_rew: np.ndarray, *, want_cache: bool = False,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Teacher-forced causal forward over the whole sequence.

    obs/prev_act/prev_rew: int arrays (P, B, T). Returns logits (P, B, T, n_act).
    """
    P, B, T = obs.shape
    H, hd = g.n_heads, g.head_dim
    cos, sin = rope_tables(T, hd)
    mask = np.triu(np.full((T, T), -1e30, dtype=DTYPE), k=1)

    x = embed(p, obs, prev_act, prev_rew)
    cache: dict[str, Any] = {"x_in": x, "cos": cos, "sin": sin, "T": T} if want_cache else {}

    for b in range(g.n_blocks):
        h, inv_r1 = rmsnorm(x, p[f"b{b}.g1"])
        q = np.einsum("pbtd,pdh->pbth", h, p[f"b{b}.Wq"])
        k = np.einsum("pbtd,pdh->pbth", h, p[f"b{b}.Wk"])
        v = np.einsum("pbtd,pdh->pbth", h, p[f"b{b}.Wv"])
        qh = q.reshape(P, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        kh = k.reshape(P, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        vh = v.reshape(P, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        qr = rope_apply(qh, cos, sin)
        kr = rope_apply(kh, cos, sin)
        scores = np.einsum("pbhtd,pbhsd->pbhts", qr, kr) / np.sqrt(hd, dtype=DTYPE)
        probs = softmax(scores + mask)
        ctx = np.einsum("pbhts,pbhsd->pbhtd", probs, vh)
        ctx_m = ctx.transpose(0, 1, 3, 2, 4).reshape(P, B, T, H * hd)
        attn_out = np.einsum("pbth,phd->pbtd", ctx_m, p[f"b{b}.Wo"])
        x_mid = (x + attn_out).astype(DTYPE)

        h2, inv_r2 = rmsnorm(x_mid, p[f"b{b}.g2"])
        pre = np.einsum("pbtd,pdf->pbtf", h2, p[f"b{b}.W1"])
        act = gelu(pre)
        ff = np.einsum("pbtf,pfd->pbtd", act, p[f"b{b}.W2"])
        x_next = (x_mid + ff).astype(DTYPE)

        if want_cache:
            cache[f"b{b}"] = dict(
                x=x, h=h, inv_r1=inv_r1, qr=qr, kr=kr, qh=qh, kh=kh, vh=vh,
                probs=probs, ctx_m=ctx_m, x_mid=x_mid, h2=h2, inv_r2=inv_r2,
                pre=pre, act=act,
            )
        x = x_next

    xf, inv_rf = rmsnorm(x, p["g_final"])
    logits = (np.einsum("pbtd,pda->pbta", xf, p["W_head"])
              + p["b_head"][:, None, None, :]).astype(DTYPE)
    if want_cache:
        cache["final"] = dict(x=x, xf=xf, inv_rf=inv_rf)
        return logits, cache
    return logits, None


def backward(
    p: Params, g: ArchGenome, cache: dict[str, Any], dlogits: np.ndarray,
    obs: np.ndarray, prev_act: np.ndarray, prev_rew: np.ndarray,
) -> Params:
    """Reverse-mode gradients for :func:`forward_full`. Validated by finite differences."""
    P, B, T = obs.shape
    H, hd = g.n_heads, g.head_dim
    cos, sin = cache["cos"], cache["sin"]
    grads: Params = {k: np.zeros_like(v) for k, v in p.items()}

    fin = cache["final"]
    grads["b_head"] = dlogits.sum(axis=(1, 2))
    grads["W_head"] = np.einsum("pbtd,pbta->pda", fin["xf"], dlogits)
    dxf = np.einsum("pbta,pda->pbtd", dlogits, p["W_head"])
    dx, dg_final = rmsnorm_backward(dxf, fin["x"], fin["inv_rf"], p["g_final"])
    grads["g_final"] = dg_final

    for b in reversed(range(g.n_blocks)):
        c = cache[f"b{b}"]
        # --- FFN branch (residual: x_next = x_mid + ff)
        dff = dx
        grads[f"b{b}.W2"] = np.einsum("pbtf,pbtd->pfd", c["act"], dff)
        dact = np.einsum("pbtd,pfd->pbtf", dff, p[f"b{b}.W2"])
        dpre = gelu_backward(dact, c["pre"])
        grads[f"b{b}.W1"] = np.einsum("pbtd,pbtf->pdf", c["h2"], dpre)
        dh2 = np.einsum("pbtf,pdf->pbtd", dpre, p[f"b{b}.W1"])
        dx_mid_n, dg2 = rmsnorm_backward(dh2, c["x_mid"], c["inv_r2"], p[f"b{b}.g2"])
        grads[f"b{b}.g2"] = dg2
        dx_mid = dx + dx_mid_n

        # --- attention branch (residual: x_mid = x + attn_out)
        dattn = dx_mid
        grads[f"b{b}.Wo"] = np.einsum("pbth,pbtd->phd", c["ctx_m"], dattn)
        dctx_m = np.einsum("pbtd,phd->pbth", dattn, p[f"b{b}.Wo"])
        dctx = dctx_m.reshape(P, B, T, H, hd).transpose(0, 1, 3, 2, 4)
        dprobs = np.einsum("pbhtd,pbhsd->pbhts", dctx, c["vh"])
        dvh = np.einsum("pbhts,pbhtd->pbhsd", c["probs"], dctx)
        dscores = softmax_backward(dprobs, c["probs"]) / np.sqrt(hd, dtype=DTYPE)
        dqr = np.einsum("pbhts,pbhsd->pbhtd", dscores, c["kr"])
        dkr = np.einsum("pbhts,pbhtd->pbhsd", dscores, c["qr"])
        dqh = rope_apply(dqr, cos, sin, inverse=True)
        dkh = rope_apply(dkr, cos, sin, inverse=True)

        def merge(t: np.ndarray) -> np.ndarray:
            return t.transpose(0, 1, 3, 2, 4).reshape(P, B, T, H * hd)

        dq, dk, dv = merge(dqh), merge(dkh), merge(dvh)
        grads[f"b{b}.Wq"] = np.einsum("pbtd,pbth->pdh", c["h"], dq)
        grads[f"b{b}.Wk"] = np.einsum("pbtd,pbth->pdh", c["h"], dk)
        grads[f"b{b}.Wv"] = np.einsum("pbtd,pbth->pdh", c["h"], dv)
        dh = (np.einsum("pbth,pdh->pbtd", dq, p[f"b{b}.Wq"])
              + np.einsum("pbth,pdh->pbtd", dk, p[f"b{b}.Wk"])
              + np.einsum("pbth,pdh->pbtd", dv, p[f"b{b}.Wv"]))
        dx_n, dg1 = rmsnorm_backward(dh, c["x"], c["inv_r1"], p[f"b{b}.g1"])
        grads[f"b{b}.g1"] = dg1
        dx = dx_mid + dx_n

    # --- embeddings (scatter-add)
    pidx = np.repeat(np.arange(P), B * T)
    flat = dx.reshape(P * B * T, -1)
    for name, idx in (("E_obs", obs), ("E_act", prev_act), ("E_rew", prev_rew)):
        np.add.at(grads[name], (pidx, idx.reshape(-1)), flat)
    return grads


# ------------------------------------------------------------ incremental rollout


def new_kv_cache(p: Params, g: ArchGenome, batch: int, seq_len: int) -> dict[str, Any]:
    P = next(iter(p.values())).shape[0]
    return {
        "k": [np.zeros((P, batch, g.n_heads, seq_len, g.head_dim), dtype=DTYPE)
              for _ in range(g.n_blocks)],
        "v": [np.zeros((P, batch, g.n_heads, seq_len, g.head_dim), dtype=DTYPE)
              for _ in range(g.n_blocks)],
        "t": 0,
        "rope": rope_tables(seq_len, g.head_dim),
    }


def rollout_step(
    p: Params, g: ArchGenome, kv: dict[str, Any],
    obs: np.ndarray, prev_act: np.ndarray, prev_rew: np.ndarray,
) -> np.ndarray:
    """One timestep. obs/prev_act/prev_rew are int arrays (P, B). Returns logits (P, B, A)."""
    t = kv["t"]
    cos, sin = kv["rope"]
    P, B = obs.shape
    H, hd = g.n_heads, g.head_dim

    x = embed(p, obs[:, :, None], prev_act[:, :, None], prev_rew[:, :, None])[:, :, 0]
    ct, st = cos[t : t + 1], sin[t : t + 1]

    for b in range(g.n_blocks):
        h, _ = rmsnorm(x, p[f"b{b}.g1"])
        q = np.einsum("pbd,pdh->pbh", h, p[f"b{b}.Wq"]).reshape(P, B, H, 1, hd)
        k = np.einsum("pbd,pdh->pbh", h, p[f"b{b}.Wk"]).reshape(P, B, H, 1, hd)
        v = np.einsum("pbd,pdh->pbh", h, p[f"b{b}.Wv"]).reshape(P, B, H, 1, hd)
        qr = rope_apply(q, ct, st)
        kr = rope_apply(k, ct, st)
        kv["k"][b][:, :, :, t : t + 1, :] = kr
        kv["v"][b][:, :, :, t : t + 1, :] = v
        K = kv["k"][b][:, :, :, : t + 1, :]
        V = kv["v"][b][:, :, :, : t + 1, :]
        scores = np.einsum("pbhtd,pbhsd->pbhts", qr, K) / np.sqrt(hd, dtype=DTYPE)
        probs = softmax(scores)
        ctx = np.einsum("pbhts,pbhsd->pbhtd", probs, V).reshape(P, B, H * hd)
        x = (x + np.einsum("pbh,phd->pbd", ctx, p[f"b{b}.Wo"])).astype(DTYPE)
        h2, _ = rmsnorm(x, p[f"b{b}.g2"])
        act = gelu(np.einsum("pbd,pdf->pbf", h2, p[f"b{b}.W1"]))
        x = (x + np.einsum("pbf,pfd->pbd", act, p[f"b{b}.W2"])).astype(DTYPE)

    xf, _ = rmsnorm(x, p["g_final"])
    kv["t"] = t + 1
    return (np.einsum("pbd,pda->pba", xf, p["W_head"]) + p["b_head"][:, None, :]).astype(DTYPE)
