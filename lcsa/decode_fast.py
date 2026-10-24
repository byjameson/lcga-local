"""C phase: optimized decode harness.

Differences vs lcsa/decode.py (naive benchmark):
  * dense (expanded): K/V stored already expanded to H query heads — repeat_kv
    runs ONCE after prefill instead of at every decode step.
  * dense (gqa): K/V kept at G heads and SDPA is called with enable_gqa=True —
    lets the fused kernel read the cache once per KV head (if the backend
    supports it; the bench script falls back gracefully).
  * sparse: only the m selected blocks are gathered (from the GQA cache) and
    expanded for the attention call; the router keeps incremental per-block
    k-side statistics, so no full-cache work per step.

Clean-config assumption: null_logit <= -1e8 (i.e. the null key is disabled);
in that case attention over the selected keys is a plain softmax and SDPA can
be used directly. With a finite null_logit the manual path in lcsa/decode.py
remains the reference.

Parity: scripts/decode_ref.py checks step logits against the naive harness
on identical prefills.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import repeat_kv
from .model import LCAGAttention, _apply_rope


class FastCache:
    def __init__(self, max_len, n_layers):
        self.max_len = max_len
        self.layers = [None] * n_layers

    def init_layer(self, i, B, G, D, H_len, K, nb_max, device, dtype):
        self.layers[i] = {
            "k": torch.zeros(B, G, self.max_len, D, device=device, dtype=dtype),
            "v": torch.zeros(B, G, self.max_len, D, device=device, dtype=dtype),
            "k_exp": None,
            "v_exp": None,
            "pk_sum": torch.zeros(B, nb_max, K, device=device, dtype=torch.float32),
            "k_sum": torch.zeros(B, G, nb_max, D, device=device, dtype=torch.float32),
            "v_sum": torch.zeros(B, G, nb_max, D, device=device, dtype=torch.float32),
            "counts": torch.zeros(nb_max, device=device, dtype=torch.float32),
            "len": 0,
        }

    @torch.no_grad()
    def expand_kv(self, H):
        """One-time repeat_kv of the stored prefill (outside the timed loop).

        Allocates full max_len so decode steps can append expanded tokens.
        """
        for cl in self.layers:
            L = cl["len"]
            B, G, _, D = cl["k"].shape
            k_exp = torch.zeros(B, H, self.max_len, D, device=cl["k"].device, dtype=cl["k"].dtype)
            v_exp = torch.zeros_like(k_exp)
            k_exp[:, :, :L] = repeat_kv(cl["k"][:, :, :L], H // G)
            v_exp[:, :, :L] = repeat_kv(cl["v"][:, :, :L], H // G)
            cl["k_exp"] = k_exp
            cl["v_exp"] = v_exp


@torch.no_grad()
def prefill_fast(model, input_ids, block=64, max_new_tokens=0):
    """Dense prefill recording GQA KV caches + router block statistics.

    Same math as lcsa.decode.prefill_and_record (identical cache contents).
    """
    base = model.model
    cfg = model.config
    B, T = input_ids.shape
    device = input_ids.device
    H = cfg.num_attention_heads
    G = cfg.num_key_value_heads
    D = cfg.hidden_size // H
    nb_max = (T + max_new_tokens + block - 1) // block + 1
    max_len = ((T + max_new_tokens + block - 1) // block) * block
    K = 64

    cache = FastCache(max_len, len(base.layers))
    h = base.embed_tokens(input_ids)
    pos = torch.arange(T, device=device)
    cos, sin = base.rotary_emb(h, pos.unsqueeze(0))

    for i, layer in enumerate(base.layers):
        attn = layer.self_attn
        orig = attn.orig if isinstance(attn, LCAGAttention) else attn
        hn = layer.input_layernorm(h)
        q = orig.q_proj(hn).view(B, T, H, D).transpose(1, 2)
        k = orig.k_proj(hn).view(B, T, G, D).transpose(1, 2)
        v = orig.v_proj(hn).view(B, T, G, D).transpose(1, 2)
        qr, kr = _apply_rope(q, k, cos, sin)
        attn_out, _ = orig(hidden_states=hn, position_embeddings=(cos, sin),
                           attention_mask=None, past_key_values=None)
        h = h + attn_out
        h = h + layer.mlp(layer.post_attention_layernorm(h))

        cache.init_layer(i, B, G, D, T, K, nb_max, device, kr.dtype)
        cl = cache.layers[i]
        cl["k"][:, :, :T] = kr
        cl["v"][:, :, :T] = v
        cl["len"] = T
        if isinstance(attn, LCAGAttention):
            pk = F.softmax(attn.router.k_proj(hn.float()), dim=-1)  # (B,T,K)
            nfb = (T + block - 1) // block
            cl["pk_sum"][:, :nfb] = pk.view(B, nfb, block, K).sum(2)
            cl["k_sum"][:, :, :nfb] = k.float().view(B, G, nfb, block, D).sum(3)
            cl["v_sum"][:, :, :nfb] = v.float().view(B, G, nfb, block, D).sum(3)
            for j in range(nfb):
                cl["counts"][j] = min(block, T - j * block)
    logits = model.lm_head(base.norm(h))
    return logits, cache


@torch.no_grad()
def decode_step_fast(model, token_id, cache, pos, mode="sparse", block=64, budget=0.25,
                     use_expanded=False):
    """One fast decode step.

    mode: "dense" or "sparse". Dense uses the expanded cache when
    use_expanded=True, else GQA SDPA (k/v at G heads, enable_gqa=True).
    Returns logits (B,1,V).
    """
    base = model.model
    cfg = model.config
    B = token_id.shape[0]
    device = token_id.device
    H = cfg.num_attention_heads
    G = cfg.num_key_value_heads
    D = cfg.hidden_size // H

    h = base.embed_tokens(token_id)
    cos, sin = base.rotary_emb(h, torch.tensor([[pos]], device=device))

    for i, layer in enumerate(base.layers):
        attn = layer.self_attn
        orig = attn.orig if isinstance(attn, LCAGAttention) else attn
        cl = cache.layers[i]
        L = cl["len"]
        hn = layer.input_layernorm(h)
        q = orig.q_proj(hn).view(B, 1, H, D).transpose(1, 2)
        k = orig.k_proj(hn).view(B, 1, G, D).transpose(1, 2)
        v = orig.v_proj(hn).view(B, 1, G, D).transpose(1, 2)
        qr, kr = _apply_rope(q, k, cos, sin)
        cl["k"][:, :, L:L + 1] = kr
        cl["v"][:, :, L:L + 1] = v
        if use_expanded:
            assert cl["k_exp"] is not None, "call cache.expand_kv(H) once after prefill"
            cl["k_exp"][:, :, L] = kr[:, :, 0].repeat_interleave(H // G, dim=1)
            cl["v_exp"][:, :, L] = v[:, :, 0].repeat_interleave(H // G, dim=1)
        blk = L // block
        if isinstance(attn, LCAGAttention):
            pk = F.softmax(attn.router.k_proj(hn.float()), dim=-1)
            cl["pk_sum"][:, blk] += pk[:, 0]
            cl["k_sum"][:, :, blk] += k.float()[:, :, 0]
            cl["v_sum"][:, :, blk] += v.float()[:, :, 0]
            cl["counts"][blk] += 1

        if mode == "dense":
            if use_expanded:
                k_all = cl["k_exp"][:, :, :L + 1]
                v_all = cl["v_exp"][:, :, :L + 1]
                out = F.scaled_dot_product_attention(qr, k_all, v_all, is_causal=False)
            else:
                out = F.scaled_dot_product_attention(qr, cl["k"][:, :, :L + 1],
                                                     cl["v"][:, :, :L + 1],
                                                     is_causal=False, enable_gqa=True)
        else:
            assert isinstance(attn, LCAGAttention), "sparse decode needs a patched model"
            assert B == 1, "fast sparse decode supports batch 1"
            assert float(attn.null_logit.min()) <= -1e8, "fast path assumes null disabled"
            nb_now = blk + 1
            pq = F.softmax(attn.router.q_proj(hn.float()), dim=-1)[:, 0]  # (B,K)
            pk_bar = cl["pk_sum"][:, :nb_now] / cl["counts"][:nb_now].view(1, -1, 1).clamp_min(1)
            s = ((pq * attn.router.graph_diag.view(1, -1)).unsqueeze(1)
                 @ pk_bar.transpose(1, 2)).squeeze(1) \
                + ((pq @ attn.router.graph_u)
                   @ (pk_bar @ attn.router.graph_v).transpose(1, 2)).squeeze(1)
            m_sel = max(2, min(nb_now, int(round(budget * nb_now))))
            s_mask = s.clone()
            s_mask[:, blk] = 1e4                       # always include current block
            sel_blocks = torch.topk(s_mask, m_sel, dim=-1).indices[0]  # (m,)
            cl["last_sel"] = sel_blocks
            kb = cl["k"].view(B, G, -1, block, D)[:, :, sel_blocks]
            vb = cl["v"].view(B, G, -1, block, D)[:, :, sel_blocks]
            ksel = repeat_kv(kb.reshape(B, G, m_sel * block, D), H // G)
            vsel = repeat_kv(vb.reshape(B, G, m_sel * block, D), H // G)
            # slot-level causality (matches gather_sparse_attention / the graph decoder):
            # future slots of the partial current block (unwritten zeros) are masked out.
            kpos_sel = (sel_blocks.view(-1, 1) * block
                        + torch.arange(block, device=device).view(1, block)).reshape(1, 1, -1)
            gmask = torch.where(kpos_sel <= L,
                                torch.zeros((), dtype=qr.dtype, device=device),
                                torch.full((), -1e4, dtype=qr.dtype, device=device))
            out = F.scaled_dot_product_attention(qr, ksel, vsel, attn_mask=gmask)
        out_cat = out.transpose(1, 2).reshape(B, 1, H * D).to(h.dtype)
        h = h + orig.o_proj(out_cat)
        h = h + layer.mlp(layer.post_attention_layernorm(h))

    for cl in cache.layers:
        cl["len"] = cl["len"] + 1
    return model.lm_head(base.norm(h))
