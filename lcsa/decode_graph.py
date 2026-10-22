"""C phase: whole-step CUDA-graph decode.

The eager harness measures ~65-90 ms/token at every context length (1k..8k):
on this machine the decode step is launch-overhead-bound (~300+ tiny kernels
per step, WDDM), not KV-traffic-bound. This module captures one COMPLETE
decode step (all layers) as a single CUDA graph and replays it, so per-step
host cost is a handful of buffer writes plus one graph launch.

Two modes (single request, batch=1, fp16, base weights frozen):
  dense  : one SDPA per layer over the full expanded KV cache (preallocated to
           max_len); future slots are masked additively (mask is enabled one
           position per replay inside the graph via scatter_).
  sparse : the router's per-block statistics are updated in-graph, the top-M
           blocks are re-selected every step (M = round(budget * n_blocks),
           fixed so shapes are static), gathered from the expanded cache, and
           attended with a gathered mask built from the selected block indices.

Assumes the clean inference config (null key disabled, null_logit <= -1e8):
attention over selected keys is a plain softmax, so SDPA can be used.

Buffers are all preallocated; the only per-step host work is two tiny H2D
copies (token id and position). Correctness is checked by the benchmark
(scripts/decode_ref.py --graph) against the eager fast paths on identical
prefills.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention import repeat_kv
from .decode_fast import FastCache
from .model import LCAGAttention, _apply_rope

_MASK_NEG = -1e4  # fp16-safe additive mask value (exp -> 0)


class GraphDecoder:
    def __init__(self, model, max_len: int, mode: str = "dense", block: int = 64,
                 budget: float = 0.25, device: str = "cuda"):
        assert mode in ("dense", "sparse")
        assert max_len % block == 0, "max_len must be a multiple of block"
        self.model = model
        self.base = model.model
        self.cfg = model.config
        self.device = device
        self.mode = mode
        self.block = block
        self.max_len = int(max_len)
        self.nb = self.max_len // block

        cfg = self.cfg
        self.H = cfg.num_attention_heads
        self.G = cfg.num_key_value_heads
        self.D = cfg.hidden_size // self.H
        assert self.H % self.G == 0

        # ---- input buffers (updated from host each step) -------------------
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.pos = torch.zeros(1, 1, dtype=torch.long, device=device)
        self.tok_pinned = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        self.pos_pinned = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        self.pos_val = 0
        self._zero = torch.zeros((), dtype=torch.float16, device=device)
        self._neg = torch.full((), _MASK_NEG, dtype=torch.float16, device=device)

        # ---- KV caches (expanded to H heads, preallocated) ------------------
        self.kbuf = []
        self.vbuf = []
        self.routers = []
        for layer in self.base.layers:
            self.kbuf.append(torch.zeros(1, self.H, self.max_len, self.D,
                                         dtype=torch.float16, device=device))
            self.vbuf.append(torch.zeros(1, self.H, self.max_len, self.D,
                                         dtype=torch.float16, device=device))
            attn = layer.self_attn
            if mode == "sparse":
                assert isinstance(attn, LCAGAttention), "sparse graph needs patched model"
                assert float(attn.null_logit.min()) <= -1e8, "graph assumes null disabled"
                self.routers.append(attn.router)

        # ---- output buffer ---------------------------------------------------
        self.logits = torch.zeros(1, 1, cfg.vocab_size, dtype=torch.float16, device=device)

        if mode == "dense":
            self.mask = torch.full((1, 1, 1, self.max_len), _MASK_NEG,
                                   dtype=torch.float16, device=device)
            self.mask_flat = self.mask.view(-1)
        else:
            self.M = max(2, int(round(budget * self.nb)))
            self.block_ids = torch.arange(self.nb, device=device)
            self.ar_block = torch.arange(block, device=device)
            self.pk_sum = []
            self.counts = []
            for attn in (layer.self_attn for layer in self.base.layers):
                Kc = attn.router.graph_diag.shape[0]
                self.pk_sum.append(torch.zeros(1, self.nb, Kc, device=device))
                self.counts.append(torch.zeros(self.nb, device=device))
            self.ones_b = torch.ones(1, device=device)
            # per-layer selection buffer, filled in-graph at every replay (parity evidence)
            self.idx_dbg = [torch.zeros(self.M, dtype=torch.long, device=device)
                            for _ in self.base.layers]

        self.graph = None
        self._capture()

    # ------------------------------------------------------------------ step
    @torch.no_grad()
    def _step(self):
        base, cfg = self.base, self.cfg
        B = 1
        H, G, D = self.H, self.G, self.D
        blk = (self.pos // self.block).view(-1)                      # (B,)

        if self.mode == "dense":
            self.mask_flat.scatter_(0, self.pos.view(-1), 0.0)

        h = base.embed_tokens(self.tok)
        cos, sin = base.rotary_emb(h, self.pos)

        for i, layer in enumerate(base.layers):
            attn = layer.self_attn
            orig = attn.orig if isinstance(attn, LCAGAttention) else attn
            hn = layer.input_layernorm(h)
            q = orig.q_proj(hn).view(B, 1, H, D).transpose(1, 2)
            k = orig.k_proj(hn).view(B, 1, G, D).transpose(1, 2)
            v = orig.v_proj(hn).view(B, 1, G, D).transpose(1, 2)
            qr, kr = _apply_rope(q, k, cos, sin)
            ke = kr[:, :, 0].repeat_interleave(H // G, dim=1).unsqueeze(2)   # (B,H,1,D)
            ve = v[:, :, 0].repeat_interleave(H // G, dim=1).unsqueeze(2)
            self.kbuf[i].index_copy_(2, self.pos.view(-1), ke)
            self.vbuf[i].index_copy_(2, self.pos.view(-1), ve)

            if self.mode == "sparse":
                router = self.routers[i]
                # in-graph router statistics update for the new token
                pk = F.softmax(router.k_proj(hn.float()), dim=-1)            # (B,1,K)
                self.pk_sum[i].index_add_(1, blk, pk[:, 0].unsqueeze(1))
                self.counts[i].index_add_(0, blk, self.ones_b)
                # block scores (same math as lcsa/decode_fast.py)
                pq = F.softmax(router.q_proj(hn.float()), dim=-1)[:, 0]      # (B,K)
                c = self.counts[i].clamp_min(1.0).view(1, self.nb, 1)
                pk_bar = self.pk_sum[i] / c                                  # (B,nb,K)
                s = ((pq * router.graph_diag.view(1, -1)).unsqueeze(1)
                     @ pk_bar.transpose(1, 2)).squeeze(1) \
                    + ((pq @ router.graph_u)
                       @ (pk_bar @ router.graph_v).transpose(1, 2)).squeeze(1)  # (B,nb)
                valid = self.block_ids.view(1, self.nb) <= blk.view(B, 1)
                s = s.masked_fill(~valid, _MASK_NEG)
                s = s.masked_fill(self.block_ids.view(1, self.nb) == blk.view(B, 1), 1e4)
                idx = torch.topk(s, self.M, dim=-1).indices[0]               # (M,)
                self.idx_dbg[i].copy_(idx)
                kg = self.kbuf[i].view(B, H, self.nb, self.block, D)[:, :, idx]
                vg = self.vbuf[i].view(B, H, self.nb, self.block, D)[:, :, idx]
                kg = kg.reshape(B, H, self.M * self.block, D)
                vg = vg.reshape(B, H, self.M * self.block, D)
                kpos = (idx.view(self.M, 1) * self.block + self.ar_block.view(1, self.block))
                kpos = kpos.reshape(1, 1, self.M * self.block)
                gmask = torch.where(kpos <= self.pos.view(B, 1, 1), self._zero, self._neg)
                out = F.scaled_dot_product_attention(qr, kg, vg, attn_mask=gmask)
            else:
                out = F.scaled_dot_product_attention(qr, self.kbuf[i], self.vbuf[i],
                                                     attn_mask=self.mask)

            out_cat = out.transpose(1, 2).reshape(B, 1, H * D).to(h.dtype)
            h = h + orig.o_proj(out_cat)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        self.logits.copy_(self.model.lm_head(base.norm(h)))

    # --------------------------------------------------------------- capture
    def _capture(self):
        """Warm up on a side stream, then capture the whole step.

        Tries the default SDPA backend first, then memory-efficient and math
        fallbacks (whichever captures first is replayed thereafter).
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        last_err = None
        for backends in (None, [SDPBackend.EFFICIENT_ATTENTION], [SDPBackend.MATH]):
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    if backends is None:
                        self._step()
                    else:
                        with sdpa_kernel(backends):
                            self._step()
                self.graph = g
                torch.cuda.synchronize()
                return
            except RuntimeError as e:  # capture failed with this backend
                last_err = e
                torch.cuda.synchronize()
        raise RuntimeError(f"CUDA graph capture failed for all SDPA backends: {last_err}")

    # ------------------------------------------------------------- interface
    @torch.no_grad()
    def load_prefill(self, cache: FastCache, T0: int):
        """Copy an eager prefill (GQA caches + router stats) into the graph buffers."""
        H, G = self.H, self.G
        for i, cl in enumerate(cache.layers):
            self.kbuf[i][:, :, :T0] = repeat_kv(cl["k"][:, :, :T0], H // G)
            self.vbuf[i][:, :, :T0] = repeat_kv(cl["v"][:, :, :T0], H // G)
            if self.mode == "sparse":
                self.pk_sum[i].zero_()
                self.counts[i].zero_()
                n = min(self.nb, cl["pk_sum"].shape[1])
                self.pk_sum[i][:, :n].copy_(cl["pk_sum"][:, :n])
                self.counts[i][:n].copy_(cl["counts"][:n])
        if self.mode == "dense":
            self.mask.fill_(_MASK_NEG)
            self.mask[..., :T0] = 0.0
        self.pos_val = int(T0)

    def step(self, token_id: int = 100):
        """Advance one token; returns the (1,1,V) fp16 logits buffer."""
        self.tok_pinned[0, 0] = token_id
        self.pos_pinned[0, 0] = self.pos_val
        self.tok.copy_(self.tok_pinned, non_blocking=True)
        self.pos.copy_(self.pos_pinned, non_blocking=True)
        self.graph.replay()
        self.pos_val += 1
        return self.logits
