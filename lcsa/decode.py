"""Decode (KV-cache) benchmark harness for LCGA sparse attention vs dense SDPA.

A hand-rolled decoder so both paths share identical loop overhead:
- prefill: dense pass over the prompt that also records per-layer K/V caches and
  router block statistics (concept-probability sums, K/V block sums for the coarse path)
- step: single-token decode, dense = SDPA over the full cache,
  sparse = concept-graph block selection + Triton kernel over selected blocks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import repeat_kv, sparse_attention_triton
from .model import LCAGAttention, _apply_rope


class DecoderCache:
    def __init__(self, max_len, n_layers):
        self.max_len = max_len
        self.layers = [None] * n_layers

    def init_layer(self, i, B, G, D, H_len, K, nb_max, device, dtype):
        self.layers[i] = {
            "k": torch.zeros(B, G, self.max_len, D, device=device, dtype=dtype),
            "v": torch.zeros(B, G, self.max_len, D, device=device, dtype=dtype),
            "pk_sum": torch.zeros(B, nb_max, K, device=device, dtype=torch.float32),
            "k_sum": torch.zeros(B, G, nb_max, D, device=device, dtype=torch.float32),
            "v_sum": torch.zeros(B, G, nb_max, D, device=device, dtype=torch.float32),
            "counts": torch.zeros(nb_max, device=device, dtype=torch.float32),
            "len": 0,
        }


@torch.no_grad()
def prefill_and_record(model, input_ids, block=64, max_new_tokens=0):
    """Dense prefill recording KV caches + router block stats."""
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

    cache = DecoderCache(max_len, len(base.layers))
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
        # propagate with the ORIGINAL dense attention (same start state for both paths)
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
            pk_blk = pk.view(B, nfb, block, K).sum(2)
            cl["pk_sum"][:, :nfb] = pk_blk
            kf = k.float().view(B, G, nfb, block, D).sum(3)
            vf = v.float().view(B, G, nfb, block, D).sum(3)
            cl["k_sum"][:, :, :nfb] = kf
            cl["v_sum"][:, :, :nfb] = vf
            for j in range(nfb):
                cl["counts"][j] = min(block, T - j * block)
    logits = model.lm_head(base.norm(h))
    return logits, cache


@torch.no_grad()
def decode_step(model, token_id, cache, pos, sparse=True, block=64, budget=0.25, use_ksum=True):
    """One decode step. token_id: (B,1) long. Returns logits (B,1,V)."""
    base = model.model
    cfg = model.config
    B = token_id.shape[0]
    device = token_id.device
    H = cfg.num_attention_heads
    G = cfg.num_key_value_heads
    D = cfg.hidden_size // H
    scale = D ** -0.5

    h = base.embed_tokens(token_id)  # (B,1,D)
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
        blk = L // block
        if isinstance(attn, LCAGAttention):
            pk = F.softmax(attn.router.k_proj(hn.float()), dim=-1)  # (B,1,K)
            cl["pk_sum"][:, blk] += pk[:, 0]
            cl["k_sum"][:, :, blk] += k.float()[:, :, 0]
            cl["v_sum"][:, :, blk] += v.float()[:, :, 0]
            cl["counts"][blk] += 1

        if not sparse:
            k_all = cl["k"][:, :, :L + 1]
            v_all = cl["v"][:, :, :L + 1]
            out = F.scaled_dot_product_attention(qr, repeat_kv(k_all, H // G),
                                                 repeat_kv(v_all, H // G), is_causal=False)
        else:
            assert isinstance(attn, LCAGAttention), "sparse decode needs a patched model"
            assert B == 1, "sparse decode bench supports batch 1"
            nb_now = blk + 1
            pq = F.softmax(attn.router.q_proj(hn.float()), dim=-1)[:, 0]  # (B,K)
            pk_bar = cl["pk_sum"][:, :nb_now] / cl["counts"][:nb_now].view(1, -1, 1).clamp_min(1)
            s = ((pq * attn.router.graph_diag.view(1, -1)).unsqueeze(1)
                 @ pk_bar.transpose(1, 2)).squeeze(1) \
                + ((pq @ attn.router.graph_u)
                   @ (pk_bar @ attn.router.graph_v).transpose(1, 2)).squeeze(1)  # (B,nb_now)
            m_sel = max(2, min(nb_now, int(round(budget * nb_now))))
            s_mask = s.clone()
            s_mask[:, blk] = 1e4          # always include current block
            topv, topi = torch.topk(s_mask, m_sel, dim=-1)
            sel_blocks = topi[0].clamp_min(0)   # (m,)
            cl["last_sel"] = sel_blocks
            kb = cl["k"].view(B, G, -1, block, D)[:, :, sel_blocks]   # (B,G,m,block,D)
            vb = cl["v"].view(B, G, -1, block, D)[:, :, sel_blocks]
            ksel = repeat_kv(kb.reshape(B, G, m_sel * block, D), H // G)
            vsel = repeat_kv(vb.reshape(B, G, m_sel * block, D), H // G)
            scores = torch.matmul(qr, ksel.transpose(-1, -2)) * scale  # (B,H,1,m*block)
            # slot-level causality (same semantics as gather_sparse_attention / triton /
            # decode_graph): future slots of the partial current block are not valid keys.
            # Without this, unwritten zero slots (score 0 -> softmax weight exp(0)=1)
            # pollute the attention output — a deviation from the training-time attention.
            kpos_sel = (sel_blocks.view(-1, 1) * block
                        + torch.arange(block, device=device).view(1, block)).reshape(1, 1, -1)
            scores = scores.masked_fill(~(kpos_sel <= L), float("-inf"))
            nl = attn.null_logit.view(1, H, 1, 1)
            scores_ext = torch.cat([scores, nl.expand(B, H, 1, 1)], dim=-1)
            p = F.softmax(scores_ext.float(), dim=-1)[..., :-1].to(qr.dtype)
            out_exact = torch.matmul(p, vsel)  # (B,H,1,D)
            # coarse path
            if use_ksum:
                kb2 = cl["k_sum"][:, :, :nb_now] / cl["counts"][:nb_now].view(1, 1, -1, 1).clamp_min(1)
                vb2 = cl["v_sum"][:, :, :nb_now] / cl["counts"][:nb_now].view(1, 1, -1, 1).clamp_min(1)
                kb2h = repeat_kv(kb2, H // G).float()
                vb2h = repeat_kv(vb2, H // G).float()
                lc = torch.matmul(qr.float(), kb2h.transpose(-1, -2)) * scale + attn.lambda_router.float() * s.unsqueeze(1)
                lc = torch.cat([lc, attn.coarse_null.view(1, H, 1, 1).expand(B, H, 1, 1)], dim=-1)
                pc = F.softmax(lc, dim=-1)
                vb2p = torch.cat([vb2h, torch.zeros(B, H, 1, D, device=device)], dim=2)
                out_coarse = torch.einsum("bhij,bhjd->bhid", pc, vb2p)
                g = torch.sigmoid(attn.gate).view(1, H, 1, 1)
                out = g * out_exact + (1 - g) * out_coarse.to(qr.dtype)
            else:
                out = out_exact
        out_cat = out.transpose(1, 2).reshape(B, 1, H * D).to(h.dtype)
        h = h + orig.o_proj(out_cat)

        h = h + layer.mlp(layer.post_attention_layernorm(h))
    # advance cache length
    for cl in cache.layers:
        cl["len"] = cl["len"] + 1
    logits = model.lm_head(base.norm(h))
    return logits
