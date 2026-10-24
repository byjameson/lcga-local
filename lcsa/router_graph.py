"""A2: CUDA-graph replay for the eval-time LCGA router pipeline (inference-only).

The router forward is ~25 tiny CUDA ops per layer; on a 6GB consumer GPU the CPU
launch overhead dominates (~18ms/forward @T=1024 for 24 layers). Capturing the
pipeline (cast -> concept logits -> block scores -> select -> probs) into one
CUDA graph per (layer, B, T) collapses it to ~one replay + an input copy, with
bitwise-identical kernels.

Safety model:
  * opt-in only (lcsa.model.set_router_graph(True)); auto-off for gumbel /
    training / capture paths. Enable AFTER weights are final (post load_ckpt /
    LoRA merge); the first exception disables the mode for the process.
  * all layers share one memory pool + one fp16 input buffer per (B, T): graphs
    are replayed strictly sequentially on the caller's stream, so pool addresses
    are reused safely; downstream readers run in stream order before the next
    replay overwrites them.
  * the first real input doubles as a correctness gate: eager pipeline result is
    compared against a replay (idx bitwise, probs within 1e-5); mismatch ->
    mode disabled, eager fallback for everything.
"""
from __future__ import annotations

import torch

_STATE = {"disabled": False, "disabled_reason": "", "pool": None, "bufs": {}}


def is_disabled():
    return _STATE["disabled"]


def _shared_pool():
    if _STATE["pool"] is None:
        _STATE["pool"] = torch.cuda.graphs.graph_pool_handle()
    return _STATE["pool"]


def _input_buf(B, T, d, device):
    key = (B, T, d, str(device))
    buf = _STATE["bufs"].get(key)
    if buf is None:
        buf = torch.zeros(B, T, d, dtype=torch.float16, device=device)
        _STATE["bufs"][key] = buf
    return buf


class RouterGraph:
    def __init__(self, router, block, budget):
        self.router = router
        self.block = block
        self.budget = budget
        self.entries = {}

    def _pipeline(self, buf16):
        h32 = buf16.float()
        ql, kl = self.router.concept_logits(h32)
        sc = self.router.block_scores(ql, kl, h32, block=self.block)
        idx, probs = self.router.select_blocks(sc, block=self.block, budget=self.budget)
        return idx, probs

    def run(self, hs16):
        """Return (idx, probs) from graph replay, or None -> caller falls back to eager."""
        if _STATE["disabled"]:
            return None
        B, T, d = hs16.shape
        buf = _input_buf(B, T, d, hs16.device)
        buf.copy_(hs16)
        ent = self.entries.get((B, T))
        if ent is None:
            ent = self._capture(B, T, d, buf)
            if ent is None:
                return None
            if not self._gate(hs16, ent):
                return None
            return ent["idx"], ent["probs"]  # gate replay already ran on this input
        ent["g"].replay()
        return ent["idx"], ent["probs"]

    @torch.no_grad()
    def _capture(self, B, T, d, buf):
        NB = T // self.block
        self.router._nb_buffers(NB, buf.device)          # pre-create outside the pool
        self.router._budget_buffers(NB, self.budget, buf.device)
        side = torch.cuda.Stream(device=buf.device)
        side.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._pipeline(buf)
        finally:
            torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=_shared_pool()):
                idx, probs = self._pipeline(buf)
        except Exception as e:  # depends on driver/torch build
            _STATE["disabled"] = True
            _STATE["disabled_reason"] = f"{type(e).__name__}: {e}"
            print(f"[lcga] router graph capture failed -> eager fallback "
                  f"({_STATE['disabled_reason']})", flush=True)
            return None
        self.entries[(B, T)] = ent = {"g": g, "idx": idx, "probs": probs}
        return ent

    @torch.no_grad()
    def _gate(self, hs16, ent):
        """Eager-vs-graph equality check on the first real input."""
        h32 = hs16.float()
        ql, kl = self.router.concept_logits(h32)
        sc = self.router.block_scores(ql, kl, h32, block=self.block)
        idx_e, pr_e = self.router.select_blocks(sc, block=self.block, budget=self.budget)
        ent["g"].replay()
        ok_idx = torch.equal(idx_e, ent["idx"])
        ok_pr = torch.allclose(pr_e, ent["probs"], atol=1e-5, rtol=1e-4)
        if not (ok_idx and ok_pr):
            _STATE["disabled"] = True
            _STATE["disabled_reason"] = (
                f"gate mismatch (idx_ok={ok_idx}, probs_ok={ok_pr}, "
                f"probs_maxdiff={(pr_e - ent['probs']).abs().max().item():.2e})")
            print(f"[lcga] router graph {_STATE['disabled_reason']} -> eager fallback",
                  flush=True)
            return False
        return True
