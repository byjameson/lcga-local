"""LCGA core attention primitives: dense reference, gather reference, Triton block-sparse kernel."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ----------------------------------------------------------------------------
# Reference implementations (torch, fp32 math, used for teacher targets & testing)
# ----------------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, G, T, D) -> (B, G*n_rep, T, D)"""
    if n_rep == 1:
        return x
    B, G, T, D = x.shape
    return x[:, :, None, :, :].expand(B, G, n_rep, T, D).reshape(B, G * n_rep, T, D)


def dense_attention_ref(q, k, v, scale, block=64, chunk=128):
    """Dense causal attention in fp32.

    q: (B, H, T, D) fp16/fp32, k/v: (B, G, T, D).
    Returns out (B, H, T, D, same dtype as q) and block_mass (B, H, nb, nb) fp32,
    where block_mass[b,h,I,J] = fraction of attention mass from query block I to key block J.
    """
    B, H, T, D = q.shape
    G = k.shape[1]
    assert T % block == 0
    nb = T // block
    kk = repeat_kv(k, H // G).float()
    vv = repeat_kv(v, H // G).float()
    out = torch.empty_like(q)
    mass = torch.zeros(B, H, nb, nb, device=q.device, dtype=torch.float32)
    kpos = torch.arange(T, device=q.device)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        s = torch.matmul(q[:, :, t0:t1].float(), kk.transpose(-1, -2)) * scale  # (B,H,c,T)
        qpos = torch.arange(t0, t1, device=q.device)
        s = s.masked_fill(kpos.view(1, 1, 1, T) > qpos.view(1, 1, -1, 1), float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, :, t0:t1] = torch.matmul(p, vv).to(q.dtype)
        # block mass: aggregate query rows into query blocks
        assert (t1 - t0) % block == 0 and t0 % block == 0
        p6 = p.view(B, H, (t1 - t0) // block, block, nb, block)
        mass[:, :, t0 // block: t1 // block, :] += p6.sum(3).sum(4)
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
    return out, mass


def gather_sparse_attention(q, k, v, idx, scale, block=64, null_logit=None,
                            return_stats=False):
    """Reference sparse block attention (fp32 math).

    q: (B,H,T,D), k/v: (B,G,T,D), idx: (B, NB, MMAX) int32 with -1 for empty slots.
    Returns out (B,H,T,D) in q.dtype.

    Formulation: num = renormalized attention over the selected keys (sums to 1),
    out = num * sigmoid(lse - null_logit)  [null key of value 0 with logit nl].
    """
    B, H, T, D = q.shape
    G = k.shape[1]
    assert T % block == 0
    NB = T // block
    bs = block
    MMAX = idx.shape[2]
    kk = repeat_kv(k, H // G)
    vv = repeat_kv(v, H // G)
    qb = q.view(B, H, NB, bs, D)
    safe = idx.clamp_min(0).long()                     # (B,NB,MMAX)
    idx6 = safe.view(B, 1, NB, MMAX, 1, 1).expand(B, H, NB, MMAX, bs, D)
    kb = torch.gather(kk.view(B, H, NB, bs, D).unsqueeze(3).expand(B, H, NB, MMAX, bs, D), 2, idx6)
    vb = torch.gather(vv.view(B, H, NB, bs, D).unsqueeze(3).expand(B, H, NB, MMAX, bs, D), 2, idx6)
    scores = torch.einsum("bhmid,bhmnjd->bhmnij", qb.float(), kb.float()) * scale
    # validity: causal (key pos <= query pos) and non-padding slot
    qpos = (torch.arange(NB, device=q.device).view(1, NB, 1, 1, 1) * bs
            + torch.arange(bs, device=q.device).view(1, 1, 1, bs, 1))         # (1,NB,1,bs,1)
    kpos = (safe.view(B, NB, MMAX, 1, 1) * bs
            + torch.arange(bs, device=q.device).view(1, 1, 1, 1, bs))         # (B,NB,MMAX,1,bs)
    valid = (kpos <= qpos) & (idx.view(B, NB, MMAX, 1, 1) >= 0)               # (B,NB,MMAX,bs,bs)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)              # (B,H,NB,MMAX,bs) per block
    lse = torch.nan_to_num(lse, neginf=-1e9)
    lse_all = torch.logsumexp(lse, dim=3)              # (B,H,NB,bs) over blocks
    p = torch.exp(scores - lse_all[..., None, :, None])  # (B,H,NB,MMAX,bs,bs)
    p = torch.nan_to_num(p, nan=0.0)
    num = torch.einsum("bhmnij,bhmnjd->bhmnid", p, vb.float()).sum(dim=3)  # (B,H,NB,bs,D)
    scale_attn = None
    if null_logit is not None:
        nl = null_logit.view(1, H, 1, 1).float()       # (1,H,1,1)
        scale_attn = torch.sigmoid(lse_all - nl)
        num = num * scale_attn.unsqueeze(-1)
    out = num.view(B, H, T, D).to(q.dtype)
    if return_stats:
        with torch.no_grad():
            lse_f = lse_all.flatten()
            stats = {"lse_mean": float(lse_f.mean()),
                     "lse_p05": float(lse_f.quantile(0.05)),
                     "lse_p95": float(lse_f.quantile(0.95))}
            if scale_attn is not None:
                stats["scale_mean"] = float(scale_attn.mean())
                stats["scale_min"] = float(scale_attn.min())
                stats["scale_p05"] = float(scale_attn.flatten().quantile(0.05))
        return out, stats
    return out


# ----------------------------------------------------------------------------
# Triton block-sparse causal attention (forward only, GQA)
# ----------------------------------------------------------------------------
@triton.jit
def _lcsa_fwd_kernel(
    Q, K, V, OUT, IDX, NULLL,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kg, stride_kt, stride_kd,
    stride_vb, stride_vg, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_ib, stride_in,
    H, GQA_R, T, NB,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, MMAX: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    g = h // GQA_R

    offs_m = pid_q * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    q_ptrs = Q + b * stride_qb + h * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)  # (M, D) fp16

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)

    for j in range(MMAX):
        kv = tl.load(IDX + b * stride_ib + pid_q * stride_in + j)
        valid_b = kv >= 0
        kv_safe = tl.where(valid_b, kv, 0)
        offs_n = kv_safe * BLOCK_N + tl.arange(0, BLOCK_N)
        k_ptrs = K + b * stride_kb + g * stride_kg + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V + b * stride_vb + g * stride_vg + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        load_mask = (offs_n[:, None] >= 0) & (offs_d[None, :] >= 0)
        kt = tl.load(k_ptrs, mask=load_mask, other=0.0)
        s = tl.dot(q, tl.trans(kt)) * scale  # (M, N) fp32

        causal_ok = offs_n[None, :] <= offs_m[:, None]
        s = tl.where(causal_ok & valid_b, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        alpha = tl.where(m_new == float("-inf"), 1.0, alpha)
        p = tl.exp(s - m_new[:, None])
        p = tl.where(m_new[:, None] == float("-inf"), 0.0, p)
        vt = tl.load(v_ptrs, mask=load_mask, other=0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vt)
        m_i = m_new

    nl = tl.load(NULLL + h)
    m2 = tl.maximum(m_i, nl)
    l_tot = l_i * tl.exp(m_i - m2) + tl.exp(nl - m2)
    acc = acc * tl.exp(m_i - m2)[:, None]
    out = acc / l_tot[:, None]

    o_ptrs = OUT + b * stride_ob + h * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(tl.float16))


def sparse_attention_triton(q, k, v, idx, scale, null_logit=None, block=64, num_warps=4, num_stages=2):
    """Block-sparse causal attention forward via Triton.

    q: (B,H,T,D) fp16 contiguous, k/v: (B,G,T,D) fp16 contiguous,
    idx: (B, NB, MMAX) int32/int64 with -1 = empty slot.
    Returns out (B,H,T,D) fp16.
    """
    B, H, T, D = q.shape
    G = k.shape[1]
    assert T % block == 0
    NB = T // block
    MMAX = idx.shape[2]
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    idx = idx.contiguous().to(torch.int32)
    if null_logit is None:
        null_logit = torch.zeros(H, device=q.device, dtype=torch.float32)
    else:
        null_logit = null_logit.contiguous().float()
    out = torch.empty_like(q)
    grid = (NB, B * H)
    _lcsa_fwd_kernel[grid](
        q, k, v, out, idx, null_logit,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        idx.stride(0), idx.stride(1),
        H, H // G, T, NB,
        float(scale),
        BLOCK_M=block, BLOCK_N=block, D=D, MMAX=MMAX,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def build_simple_idx(T, block, budget, device):
    """Dense-causal idx (all blocks <= I, at most budget*NB selected newest-first?) for tests/baselines.

    This builds a *contiguous* selection: for query block I, keep the most recent
    m = max(1, round(budget*(I+1))) blocks ending at I (always including block 0 and I).
    Returns idx (1, NB, MMAX) int32.
    """
    NB = T // block
    m_list = [max(1, min(I + 1, max(2, int(round(budget * (I + 1)))))) if I > 0 else 1 for I in range(NB)]
    MMAX = max(m_list)
    idx = torch.full((1, NB, MMAX), -1, dtype=torch.int32, device=device)
    for I in range(NB):
        m = m_list[I]
        sel = set()
        sel.add(0)
        for j in range(0, m):
            sel.add(max(0, I - j))
        sel = sorted(sel)[:m]
        idx[0, I, : len(sel)] = torch.tensor(sel, dtype=torch.int32, device=device)
    return idx
