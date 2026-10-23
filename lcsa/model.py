"""LCGA attention module (wraps original Qwen2 attention) + model patching."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (dense_attention_ref, gather_sparse_attention,
                        sparse_attention_triton, repeat_kv)
from .concepts import ConceptGraphRouter

_STATE = {"enabled": False, "capture": False, "captured": [], "selection_mode": "topk",
          "profile": False, "profile_acc": {}, "torch_exact": False,
          "stage2": False, "dense_only": False, "dense_targets": {}, "aux_losses": [],
          "ckpt_gather": False, "router_graph": False, "capture_qk": False}


def set_torch_exact(flag: bool):
    """Use the torch gather reference for the exact path (slower; for stats/parity)."""
    _STATE["torch_exact"] = bool(flag)


def set_capture_qk(flag: bool):
    """Also capture post-RoPE q/k in the capture entries.

    Used by exp61 to build a token-level teacher: block-averaged dense mass dilutes
    a 3-token answer across a 64-token block (~1/20), which destroys the retrieval
    signal. With q/k the teacher can be the actual attention of the answer tokens.
    """
    _STATE["capture_qk"] = bool(flag)


def set_router_graph(flag: bool):
    """A2: capture the eval-time router pipeline into CUDA graphs (inference-only).

    Enable only after weights are final (post load-ckpt / LoRA merge); auto-off
    for topk+gumbel / training / capture paths. The first run double-checks the
    graph against the eager pipeline and falls back on any mismatch.
    """
    _STATE["router_graph"] = bool(flag)


def set_stage2_mode(flag: bool, dense_only: bool = False, keep_targets: bool = False):
    """Stage-2 training modes.

    dense_only=True : pass-1. Each wrapper computes ONLY the dense teacher
        (block mass + attention output) and stores it in _STATE["dense_targets"]
        keyed by layer index; nothing else runs.
    flag=True       : pass-2. The exact path uses the differentiable gather
        reference, aux losses (KL against stored mass, MSE against stored dense
        output) are appended to _STATE["aux_losses"], output is not detached.
    keep_targets    : do not clear dense targets (pass-2 after pass-1).
    """
    _STATE["stage2"] = bool(flag)
    _STATE["dense_only"] = bool(dense_only)
    _STATE["aux_losses"] = []
    if not keep_targets:
        _STATE["dense_targets"] = {}


def get_aux_losses():
    return list(_STATE["aux_losses"])


def set_gather_checkpoint(flag: bool):
    """Stage-2 memory saver: recompute the gather attention in backward."""
    _STATE["ckpt_gather"] = bool(flag)


def set_selection_mode(mode: str):
    """'topk' (learned), 'random', 'local' (mandatory blocks only), 'window'
    (sink + recent blocks), or 'oracle' (top-m by TRUE dense block mass;
    diagnostics upper bound only)."""
    assert mode in ("topk", "random", "local", "window", "oracle")
    _STATE["selection_mode"] = mode


def set_profiling(flag: bool):
    _STATE["profile"] = bool(flag)
    if flag:
        _STATE["profile_acc"] = {}


def get_profile():
    return dict(_STATE["profile_acc"])


def _profile_add(key, ms):
    acc = _STATE["profile_acc"]
    acc[key] = acc.get(key, 0.0) + ms


def set_sparse_enabled(flag: bool):
    _STATE["enabled"] = bool(flag)


def sparse_enabled() -> bool:
    return _STATE["enabled"]


def begin_capture():
    _STATE["capture"] = True
    _STATE["captured"] = []


def end_capture():
    _STATE["capture"] = False
    return _STATE["captured"]


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


class LCAGAttention(nn.Module):
    """Wraps a frozen Qwen2 attention module with concept-graph sparse attention."""

    def __init__(self, orig: nn.Module, n_concepts: int = 64, graph_rank: int = 16,
                 block: int = 64, budget: float = 0.25, dynamic: bool = False,
                 content_rank: int = 0, gumbel_sigma: float = 0.0,
                 capture_dense: bool = True, use_coarse: bool = True,
                 null_init: float = -3.0, gate_init: float = 0.85, rel_bias: bool = True,
                 recall_gain: bool = False):
        super().__init__()
        self.orig = orig
        cfg = orig.config
        self.H = cfg.num_attention_heads
        self.G = cfg.num_key_value_heads
        self.D = orig.head_dim
        self.scale = orig.scaling
        self.block = block
        self.budget = budget
        self.gumbel_sigma = gumbel_sigma
        self.capture_dense = capture_dense
        self.use_coarse = use_coarse
        d = cfg.hidden_size
        self.router = ConceptGraphRouter(d, n_concepts, graph_rank,
                                         content_rank=content_rank, dynamic=dynamic,
                                         rel_bias=rel_bias)
        self.gate = nn.Parameter(torch.full((self.H,), float(gate_init)))
        self.null_logit = nn.Parameter(torch.full((self.H,), float(null_init)))
        self.coarse_null = nn.Parameter(torch.full((self.H,), float(null_init)))
        self.lambda_router = nn.Parameter(torch.tensor(1.0))
        # stage-2 output gain: out = exact * exp(gain_logit) * recall_hat^gain_alpha
        self.use_recall_gain = recall_gain
        if recall_gain:
            self.gain_logit = nn.Parameter(torch.zeros(self.H))
            self.gain_alpha = nn.Parameter(torch.zeros(self.H))
        # A2: per-layer CUDA-graph router runner (lazy, inference-only)
        self._rg = None

    def _router_graph_forward(self, hs):
        """A2: router pipeline via CUDA graph; on any problem return (None, None)."""
        try:
            if self._rg is None:
                from .router_graph import RouterGraph
                self._rg = RouterGraph(self.router, self.block, self.budget)
            res = self._rg.run(hs)
        except Exception as e:  # driver/torch quirks -> eager fallback for the process
            print(f"[lcga] router graph disabled ({type(e).__name__}: {e})", flush=True)
            set_router_graph(False)
            self._rg = None
            return None, None
        if res is None:
            set_router_graph(False)   # gate already disabled the module permanently
            return None, None
        return res

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kwargs):
        if not _STATE["enabled"]:
            return self.orig(hidden_states, position_embeddings, attention_mask,
                             past_key_values, **kwargs)
        # NOTE: the sparse path processes full sequences; a KV cache (if passed by the
        # caller) is intentionally ignored. Decode uses lcsa/decode.py instead.
        hs = hidden_states
        B, T, _ = hs.shape
        assert T % self.block == 0, f"seq len {T} not divisible by block {self.block}"
        o = self.orig
        H, G, D = self.H, self.G, self.D
        bs = self.block
        NB = T // bs

        q = o.q_proj(hs).view(B, T, H, D).transpose(1, 2)
        k = o.k_proj(hs).view(B, T, G, D).transpose(1, 2)
        v = o.v_proj(hs).view(B, T, G, D).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = _apply_rope(q, k, cos, sin)

        if _STATE["dense_only"]:
            # stage-2 pass 1: produce and store the dense teacher for this layer.
            with torch.no_grad():
                dout, dmass = dense_attention_ref(q.detach(), k.detach(), v.detach(),
                                                  self.scale, block=bs)
                dout_cat = dout.transpose(1, 2).reshape(B, T, H * D).to(hs.dtype)
                dense_out = o.o_proj(dout_cat)
            _STATE["dense_targets"][self.layer_idx] = {"mass": dmass,
                                                       "out": dense_out.detach()}
            return dense_out, None

        sigma = self.gumbel_sigma if self.training else 0.0
        mode = _STATE["selection_mode"]
        use_graph = (_STATE["router_graph"] and mode == "topk" and sigma == 0.0
                     and not _STATE["stage2"] and not _STATE["torch_exact"]
                     and not _STATE["capture"])
        idx = probs = None
        if _STATE["profile"]:
            ev0 = torch.cuda.Event(True)
            ev1 = torch.cuda.Event(True)
            ev2 = torch.cuda.Event(True)
            ev3 = torch.cuda.Event(True)
            ev0.record()
        if use_graph:
            idx, probs = self._router_graph_forward(hs)
        if idx is None:  # eager path (also the graph fallback)
            hs32 = hs.float()
            ql, kl = self.router.concept_logits(hs32)          # (B,T,K) fp32, requires grad
            scores = self.router.block_scores(ql, kl, hs32, block=bs)  # (B,NB,NB)
            if mode == "topk":
                idx, sel_probs = self.router.select_blocks(scores, block=bs, budget=self.budget,
                                                           gumbel_sigma=sigma)
                if scores.requires_grad:
                    # differentiable probs (KL signal / recall-gain gradient highway)
                    probs = self.router.masked_probs(scores, block=bs)
                else:
                    # eval-time reuse: select_blocks already computed bitwise-identical
                    # probs (same masked softmax with tau) -> skip the second softmax
                    probs = sel_probs
            elif mode == "random":
                idx = self.router.random_selection(scores, block=bs, budget=self.budget)
                probs = self.router.masked_probs(scores, block=bs)
            elif mode == "window":
                idx = self.router.window_selection(scores, block=bs, budget=self.budget)
                probs = self.router.masked_probs(scores, block=bs)
            elif mode == "oracle":
                with torch.no_grad():
                    _, dmass = dense_attention_ref(q.detach(), k.detach(), v.detach(),
                                                   self.scale, block=bs)
                    idx = self.router.oracle_selection(dmass.mean(1), budget=self.budget)
                probs = self.router.masked_probs(scores, block=bs)
            else:  # local
                idx = self.router.local_selection(scores, block=bs)
                probs = self.router.masked_probs(scores, block=bs)
        if _STATE["profile"]:
            ev1.record()

        est = None
        if _STATE["stage2"]:
            if _STATE["ckpt_gather"]:
                exact = torch.utils.checkpoint.checkpoint(
                    gather_sparse_attention, q, k, v, idx, self.scale, bs,
                    use_reentrant=False)
            else:
                exact = gather_sparse_attention(q, k, v, idx, self.scale, block=bs)
        elif _STATE["torch_exact"]:
            with torch.no_grad():
                exact, est = gather_sparse_attention(q, k, v, idx, self.scale, block=bs,
                                                     null_logit=self.null_logit,
                                                     return_stats=True)
        else:
            with torch.no_grad():
                exact = sparse_attention_triton(q.detach().contiguous(),
                                                k.detach().contiguous(),
                                                v.detach().contiguous(), idx, self.scale,
                                                null_logit=self.null_logit.detach())  # (B,H,T,D) fp16
        if _STATE["profile"]:
            ev2.record()

        if self.use_recall_gain:
            # stage-2 output gain: scale = exp(gain_logit) * recall_hat^gain_alpha
            # recall_hat = router mass captured by the selected blocks (differentiable
            # through probs). At init (gain_logit=0, gain_alpha=0) the scale is 1.0.
            with torch.no_grad():
                sel = torch.zeros_like(probs, dtype=torch.bool)
                sel.scatter_(2, idx.clamp_min(0).long(), True)
            recall_hat = (probs * sel).sum(-1).clamp(0.05, 1.0)          # (B, NB)
            fac = torch.exp(self.gain_logit).view(1, H, 1) * \
                recall_hat.view(B, 1, NB) ** self.gain_alpha.view(1, H, 1)
            fac = fac.view(B, H, NB, 1, 1).expand(B, H, NB, bs, 1).reshape(B, H, T, 1)
            exact = (exact.float() * fac).to(exact.dtype)

        if self.use_coarse:
            kbar = k.float().reshape(B, G, NB, bs, D).mean(3)      # (B,G,NB,D)
            vbar = v.float().reshape(B, G, NB, bs, D).mean(3)
            qbar = q.float().reshape(B, H, NB, bs, D).mean(3)
            kbar_h = repeat_kv(kbar, H // G)                        # (B,H,NB,D)
            if os.environ.get("LCGA_DEBUG"):
                print(f"[dbg] B={B} H={H} G={G} qbar={tuple(qbar.shape)} "
                      f"kbar={tuple(kbar.shape)} kbar_h={tuple(kbar_h.shape)}", flush=True)
            logits_c = torch.matmul(qbar, kbar_h.transpose(-1, -2)) * self.scale \
                + self.lambda_router.float() * scores.unsqueeze(1)   # (B,H,NB,NB)
            allowed = torch.tril(torch.ones(NB, NB, device=hs.device, dtype=torch.bool))
            logits_c = logits_c.masked_fill(~allowed, float("-inf"))
            nl = self.coarse_null.view(1, H, 1, 1).expand(B, H, NB, 1)
            logits_c = torch.cat([logits_c, nl], dim=-1)            # (B,H,NB,NB+1)
            pc = torch.softmax(logits_c, dim=-1)
            vpad = torch.cat([vbar, torch.zeros(B, G, 1, D, device=hs.device, dtype=vbar.dtype)], dim=2)
            vpad_h = repeat_kv(vpad, H // G)                        # (B,H,NB+1,D)
            coarse = torch.einsum("bhij,bhjd->bhid", pc, vpad_h)    # (B,H,NB,D) fp32
            g = torch.sigmoid(self.gate).view(1, H, 1, 1)
            g16 = g.to(hs.dtype)
            coarse16 = coarse.to(hs.dtype)
            coarse16 = coarse16.unsqueeze(3).expand(B, H, NB, bs, D).reshape(B, H, T, D)
            out_pre = g16 * exact + (1.0 - g16) * coarse16
        else:
            out_pre = exact

        out_cat = out_pre.transpose(1, 2).reshape(B, T, H * D)
        out_final = o.o_proj(out_cat)

        if _STATE["stage2"]:
            tgt = _STATE["dense_targets"].get(self.layer_idx)
            if tgt is not None:
                kl = self.router.kl_mass_loss(tgt["mass"], probs)
                mse = F.mse_loss(out_final.float(), tgt["out"].float())
                _STATE["aux_losses"].append((kl, mse))

        if _STATE["profile"]:
            ev3.record()
            torch.cuda.synchronize()
            _profile_add("router_ms", ev0.elapsed_time(ev1))
            _profile_add("kernel_ms", ev1.elapsed_time(ev2))
            _profile_add("coarse_ms", ev2.elapsed_time(ev3))

        if _STATE["capture"]:
            entry = {"idx": idx.detach(), "probs": probs,
                     "scores": scores.detach(), "student": out_final}
            if _STATE["capture_qk"]:
                entry["q"] = q.detach()
                entry["k"] = k.detach()
            if est is not None:
                entry["exact_stats"] = est
            if self.capture_dense:
                with torch.no_grad():
                    dout, dmass = dense_attention_ref(q.detach(), k.detach(), v.detach(),
                                                      self.scale, block=bs)
                    dout_cat = dout.transpose(1, 2).reshape(B, T, H * D).to(hs.dtype)
                    entry["dense"] = o.o_proj(dout_cat).detach()
                    entry["mass"] = dmass
            _STATE["captured"].append(entry)
        if _STATE["stage2"]:
            return out_final, None
        return out_final.detach(), None


def patch_model(model, layers=None, **kwargs):
    """Replace attention modules of the given layers with LCAGAttention wrappers."""
    base = model.model if hasattr(model, "model") else model
    n_layers = len(base.layers)
    if layers is None:
        layers = list(range(n_layers))
    wrappers = {}
    for i in layers:
        layer = base.layers[i]
        attn = layer.self_attn
        if isinstance(attn, LCAGAttention):
            if kwargs.get("recall_gain") and not hasattr(attn, "gain_logit"):
                attn.gain_logit = nn.Parameter(torch.zeros(attn.H, device=attn.gate.device))
                attn.gain_alpha = nn.Parameter(torch.zeros(attn.H, device=attn.gate.device))
            if kwargs:
                for key, val in kwargs.items():
                    if key == "recall_gain":
                        attn.use_recall_gain = val
                    else:
                        setattr(attn, key, val)
            attn.layer_idx = i
            wrappers[i] = attn
            continue
        attn.config._attn_implementation = getattr(attn.config, "_attn_implementation", "sdpa")
        w = LCAGAttention(attn, **kwargs)
        ref = next(attn.parameters())
        w = w.to(device=ref.device)   # router/gate params stay fp32 for stability
        w.layer_idx = i
        layer.self_attn = w
        wrappers[i] = w
    return wrappers


def router_parameters(model):
    """Yields router/gate parameters of all LCAGAttention modules (excluding frozen orig weights)."""
    base = model.model if hasattr(model, "model") else model
    for layer in base.layers:
        attn = layer.self_attn
        if isinstance(attn, LCAGAttention):
            orig_ids = {id(p) for p in attn.orig.parameters()}
            for p in attn.parameters():
                if id(p) not in orig_ids:
                    yield p


def count_router_params(model):
    return sum(p.numel() for p in router_parameters(model))
