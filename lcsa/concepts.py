"""Latent Concept Graph Router for LCGA.

Tokens are projected into a distribution over K latent concepts. The block-level
attention relevance between key-block J and query-block I is predicted through a
learned concept graph G (diagonal + low rank), optionally modulated dynamically
by the query block content. Predictions are *probabilities* (softmax over key
blocks), used for:
  - hard block selection (top-m + mandatory local/sink blocks)
  - the differentiable coarse attention path (gradient highway to the router)
  - a KL loss against the dense attention's block-mass distribution (training)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptGraphRouter(nn.Module):
    def __init__(self, d_model: int, n_concepts: int = 64, graph_rank: int = 16,
                 content_rank: int = 0, tau: float = 1.0, dynamic: bool = False,
                 rel_bias: bool = True, q_pool: str = "mean"):
        super().__init__()
        self.d_model = d_model
        self.K = n_concepts
        self.graph_rank = graph_rank
        self.content_rank = content_rank
        self.dynamic = dynamic
        self.rel_bias = rel_bias
        self.q_pool = q_pool

        # concept projections (query side / key side)
        self.q_proj = nn.Linear(d_model, n_concepts, bias=False)
        self.k_proj = nn.Linear(d_model, n_concepts, bias=False)
        # concept graph: G = diag(diag) + U V^T
        self.graph_diag = nn.Parameter(torch.zeros(n_concepts))
        self.graph_u = nn.Parameter(torch.randn(n_concepts, graph_rank) * 0.02)
        self.graph_v = nn.Parameter(torch.randn(n_concepts, graph_rank) * 0.02)
        # learnable recency bias over block distance: -softplus(alpha) * log1p(I-J)
        if rel_bias:
            self.rel_alpha = nn.Parameter(torch.tensor(-2.25))  # softplus = ~0.1
        # input-dependent modulation of query-side concept weights
        if dynamic:
            self.dyn_proj = nn.Linear(d_model, n_concepts, bias=False)
            nn.init.zeros_(self.dyn_proj.weight)
        # optional bilinear content shortcut (ablation switch)
        if content_rank > 0:
            self.content_q = nn.Linear(d_model, content_rank, bias=False)
            self.content_k = nn.Linear(d_model, content_rank, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau)))))
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        # A1: memoized shape-dependent constants (masks / bias / rel table per NB,
        # selection tables per (NB, budget)). Numerically identical; kills the
        # small-op launch storm this router pays on 6GB consumer GPUs.
        self._nb_buf = {}
        self._budget_buf = {}

    @property
    def tau(self):
        return torch.exp(self.log_tau).clamp(0.05, 10.0)

    def _nb_buffers(self, NB, device):
        """Cached constants for a block count NB: causal mask, topk bias, rel table."""
        key = (NB, device)
        buf = self._nb_buf.get(key)
        if buf is None:
            allowed = torch.ones(NB, NB, device=device, dtype=torch.bool).tril()
            bias = torch.zeros(1, NB, NB, device=device, dtype=torch.float32)
            bias[:, :, 0] = 1e4
            ar = torch.arange(NB, device=device)
            bias[:, ar, ar] = 1e4
            I = torch.arange(NB, device=device).view(NB, 1)
            J = torch.arange(NB, device=device).view(1, NB)
            buf = {"not_allowed3": (~allowed).unsqueeze(0),  # (1,NB,NB)
                   "bias_topk": bias,
                   "rel_log1p": torch.log1p((I - J).clamp_min(0).float())}
            self._nb_buf[key] = buf
        return buf

    def _budget_buffers(self, NB, budget, device):
        """Cached per-row budgets m_I and position table for a given budget."""
        key = (NB, float(budget), device)
        buf = self._budget_buf.get(key)
        if buf is None:
            m_list = [1 if I == 0 else max(2, min(I + 1, int(round(budget * (I + 1)))))
                      for I in range(NB)]
            MMAX = max(m_list)
            buf = {"m_list": m_list, "MMAX": MMAX,
                   "m_tensor": torch.tensor(m_list, device=device).view(1, NB, 1),
                   "pos": torch.arange(MMAX, device=device).view(1, 1, MMAX)}
            self._budget_buf[key] = buf
        return buf

    def concept_logits(self, h):
        """h: (B, T, d) fp32 -> (q_logits, k_logits) each (B, T, K)"""
        return self.q_proj(h), self.k_proj(h)

    def block_scores(self, q_logits, k_logits, h=None, block=64):
        """Predict block-level relevance scores (B, NB, NB).

        Concept distributions are aggregated per block; scores flow through the
        concept graph. Optional dynamic modulation + content shortcut.

        Query pooling (`self.q_pool`): "mean" averages all 64 positions of the block;
        "last" uses only the last position, which is the one position in a causal
        attention block that can attend to the whole block (and therefore the one that
        actually decides the next token).
        """
        B, T, K = q_logits.shape
        NB = T // block
        pq = F.softmax(q_logits, dim=-1)  # (B,T,K)
        pk = F.softmax(k_logits, dim=-1)
        if getattr(self, "q_pool", "mean") == "last":
            pq_b = pq.view(B, NB, block, K)[:, :, -1, :]        # (B,NB,K)
        else:
            pq_b = pq.view(B, NB, block, K).mean(2)              # (B,NB,K)
        pk_b = pk.view(B, NB, block, K).mean(2)
        if self.dynamic:
            gate = 1.0 + torch.tanh(self.dyn_proj(h))  # (B,T,K)
            gate_b = gate.view(B, NB, block, K).mean(2)
            pq_b = pq_b * gate_b
        # G = diag(diag) + U V^T
        term_diag = (pq_b * self.graph_diag.view(1, 1, K)) @ pk_b.transpose(1, 2)  # (B,NB,NB)
        qu = pq_b @ self.graph_u          # (B,NB,r)
        kv = pk_b @ self.graph_v          # (B,NB,r)
        term_rank = qu @ kv.transpose(1, 2)
        scores = term_diag + term_rank
        if self.rel_bias:
            rel_log1p = self._nb_buffers(NB, scores.device)["rel_log1p"]
            scores = scores - F.softplus(self.rel_alpha) * rel_log1p
        if self.content_rank > 0 and h is not None:
            hb = h.view(B, NB, block, -1)               # (B,NB,block,d)
            h_b_k = hb.mean(2)                           # (B,NB,d) keys: block content
            # Query side: "mean" averages all 64 positions of the block; "last" uses the
            # final position, the only one in a causal block that can attend to the whole
            # block and therefore the one that decides the next token.
            h_b_q = hb[:, :, -1, :] if self.q_pool == "last" else h_b_k
            cq = self.content_q(h_b_q)
            ck = self.content_k(h_b_k)
            scores = scores + cq @ ck.transpose(1, 2)
        return scores

    def masked_probs(self, scores, block=64, with_mandatory_bias=0.0):
        """Softmax over allowed key blocks J <= I. Returns probs (B,NB,NB) fp32.

        Rows only contain probability mass for J <= I.
        """
        B, NB, _ = scores.shape
        not_allowed3 = self._nb_buffers(NB, scores.device)["not_allowed3"]
        # NOTE: divide by tau BEFORE masking with -inf; masking first makes the tau
        # gradient pick up 0 * (-inf) = NaN terms in backward.
        s = scores / self.tau
        if with_mandatory_bias > 0:
            bias = torch.zeros_like(s)
            bias[:, :, 0] = with_mandatory_bias          # sink block
            idx = torch.arange(NB, device=s.device)
            bias[:, idx, idx] = with_mandatory_bias       # diagonal
            s = s + bias
        s = s.masked_fill(not_allowed3, float("-inf"))
        return F.softmax(s, dim=-1)

    @torch.no_grad()
    def select_blocks(self, scores, block=64, budget=0.25, gumbel_sigma: float = 0.0):
        """Top-m block selection per query block.

        m_I = max(mandatory, round(budget * (I+1))) with mandatory = {0, I}.
        Returns idx (B, NB, MMAX) int32 (-1 = empty) and probs (B,NB,NB).
        """
        B, NB, _ = scores.shape
        dev = scores.device
        nbuf = self._nb_buffers(NB, dev)
        bbuf = self._budget_buffers(NB, budget, dev)
        MMAX = bbuf["MMAX"]
        not_allowed3 = nbuf["not_allowed3"]
        s = scores.masked_fill(not_allowed3, float("-inf"))
        # mandatory bias (sink block + diagonal; both lie inside the causal region,
        # so the cached bias needs no extra masking)
        s = s + nbuf["bias_topk"]
        if gumbel_sigma > 0:
            g = -torch.log(-torch.log(torch.rand_like(s).clamp_min(1e-9)).clamp_min(1e-9))
            s = s + g * gumbel_sigma
        # topk over full width; entries with -inf (or beyond the per-row budget) -> -1
        top_vals, top_idx = torch.topk(s, MMAX, dim=-1)
        keep = torch.isfinite(top_vals) & (bbuf["pos"] < bbuf["m_tensor"])
        idx = torch.where(keep, top_idx, torch.full_like(top_idx, -1)).to(torch.int32)
        probs = F.softmax(scores.masked_fill(not_allowed3, float("-inf")) / self.tau, dim=-1)
        return idx, probs

    @torch.no_grad()
    def window_selection(self, scores, block=64, budget=0.25):
        """Control: sink block + the most recent (m-1) blocks (contiguous local window)."""
        B, NB, _ = scores.shape
        dev = scores.device
        m_list = [1 if I == 0 else max(2, min(I + 1, int(round(budget * (I + 1)))))
                  for I in range(NB)]
        MMAX = max(m_list)
        idx = torch.full((B, NB, MMAX), -1, dtype=torch.int32, device=dev)
        for I in range(NB):
            m = m_list[I]
            if I == 0:
                sel = [0]
            else:
                sel = [0] + list(range(max(1, I - (m - 2)), I + 1))
            idx[:, I, :len(sel)] = torch.tensor(sel, dtype=torch.int32, device=dev)
        return idx

    @torch.no_grad()
    def oracle_selection(self, mass, budget=0.25):
        """Diagnostic upper bound: top-m blocks per row by TRUE dense block mass.

        mass: (B, NB, NB) normalized dense attention block mass.
        """
        B, NB, _ = mass.shape
        dev = mass.device
        m_list = [1 if I == 0 else max(2, min(I + 1, int(round(budget * (I + 1)))))
                  for I in range(NB)]
        MMAX = max(m_list)
        idx = torch.full((B, NB, MMAX), -1, dtype=torch.int32, device=dev)
        for I in range(NB):
            m = min(m_list[I], I + 1)
            row = mass[:, I, :I + 1].clone()
            row[:, 0] = 1e9
            row[:, I] = 1e9
            top = torch.topk(row, m, dim=-1).indices
            idx[:, I, :m] = top.to(torch.int32)
        return idx

    @torch.no_grad()
    def random_selection(self, scores, block=64, budget=0.25):
        """Control: random block selection (mandatory + random extras) at the same budget."""
        B, NB, _ = scores.shape
        dev = scores.device
        m_list = [1 if I == 0 else max(2, min(I + 1, int(round(budget * (I + 1))))) for I in range(NB)]
        MMAX = max(m_list)
        idx = torch.full((B, NB, MMAX), -1, dtype=torch.int32, device=dev)
        for I in range(NB):
            m = m_list[I]
            sel = {0, I}
            pool = [j for j in range(I + 1) if j not in sel]
            while len(sel) < m and pool:
                pick = pool.pop(int(torch.randint(0, len(pool), (1,)).item()))
                sel.add(pick)
            s = sorted(sel)
            idx[:, I, :len(s)] = torch.tensor(s, dtype=torch.int32, device=dev)
        return idx

    @torch.no_grad()
    def local_selection(self, scores, block=64):
        """Control: only mandatory sink + diagonal blocks."""
        B, NB, _ = scores.shape
        dev = scores.device
        idx = torch.full((B, NB, 2), -1, dtype=torch.int32, device=dev)
        idx[:, :, 0] = 0
        idx[:, :, 1] = torch.arange(NB, device=dev, dtype=torch.int32).view(1, NB)
        idx[:, 0, 1] = -1
        return idx

    def kl_mass_loss(self, teacher_mass, probs, eps=1e-9):
        """KL(teacher block-mass || router probs) averaged over rows.

        teacher_mass: (B,H,NB,NB) normalized over J (dense attention block mass)
        probs: (B,NB,NB) router probabilities
        """
        t = teacher_mass.mean(1)  # average over heads (B,NB,NB)
        t = t / t.sum(-1, keepdim=True).clamp_min(eps)
        r = probs.clamp_min(eps)
        kl = (t * (torch.log(t.clamp_min(eps)) - torch.log(r))).sum(-1)
        return kl.mean()
