"""M2: expert paging for the MoE-lite MiniLM (top-1 Switch experts).

Shared weights (token embedding, attention, norms, router) stay on the GPU.
Expert MLPs live in CPU RAM; per token only the *routed* experts are copied
into a small GPU slot (or an LRU cache of `capacity` experts per layer).

This is the "load only the units a query needs" primitive at token granularity;
for a dense model there are no such units, which is why the MoE path exists.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import torch
import torch.nn.functional as F

from lcsa.mini_lm import MiniLM


class PagedMoE:
    def __init__(self, ckpt_path: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16, capacity: int = 0,
                 preload: bool = False, prefetch: bool = False,
                 pin_experts: bool = True, verbose: bool = True):
        d = torch.load(ckpt_path, map_location="cpu")
        cfg = d["cfg"]
        self.cfg = cfg
        m = MiniLM(vocab=cfg["vocab"], dim=cfg["dim"], layers=cfg["layers"],
                   heads=cfg["heads"], ctx=cfg["ctx"],
                   mlp_hidden=cfg["mlp_hidden"], loops=cfg.get("loops", 1),
                   moe_experts=cfg["moe"])
        m.load_state_dict(d["model"])
        m.eval()
        self.model = m
        self.meta = d.get("meta", {})
        self.device = torch.device(device)
        self.dtype = dtype
        self.E = int(cfg["moe"])
        self.L = int(cfg["layers"])
        self.hidden = int(cfg["dim"])
        self.ctx = int(cfg["ctx"])
        self.capacity = int(capacity)
        self.verbose = verbose

        # shared / small modules on GPU
        tied = m.lm_head.weight is m.tok.weight
        m.tok.to(self.device, dtype=dtype)
        if tied:
            # .to() can replace the Parameter object; re-tie explicitly
            m.lm_head.weight = m.tok.weight
        m.norm.to(self.device, dtype=dtype)
        self.cos = m.cos.to(self.device, dtype=dtype)
        self.sin = m.sin.to(self.device, dtype=dtype)
        for blk in m.blocks:
            blk.n1.to(self.device, dtype=dtype)
            blk.n2.to(self.device, dtype=dtype)
            blk.attn.to(self.device, dtype=dtype)
            blk.mlp.router.to(self.device, dtype=dtype)
            for e in range(self.E):  # experts stay in CPU RAM (bf16)
                blk.mlp.w12[e].to(dtype=dtype)
                blk.mlp.w3[e].to(dtype=dtype)
        if pin_experts and self.device.type == "cuda":
            for blk in m.blocks:  # pinned H2D copies (non_blocking prefetch)
                for e in range(self.E):
                    blk.mlp.w12[e].weight.data = blk.mlp.w12[e].weight.data.pin_memory()
                    blk.mlp.w3[e].weight.data = blk.mlp.w3[e].weight.data.pin_memory()
        if m.lm_head.weight.device.type != self.device.type:
            raise RuntimeError("tied lm_head did not follow token embedding to GPU")

        self.slots = [None] * self.L
        self.last = [-1] * self.L
        self.cache = [OrderedDict() for _ in range(self.L)]
        self.bytes_copied = 0
        self.copies = 0
        self.hits = 0
        self.usage = [torch.zeros(self.E) for _ in range(self.L)]

        # M3: prefetch state
        self.prefetch = bool(prefetch)
        if self.prefetch and self.capacity > 2:
            raise ValueError("prefetch supports capacity <= 2")
        if self.prefetch and preload:
            raise ValueError("prefetch and preload are mutually exclusive")
        self.pslots = [None] * self.L
        self.trans = [torch.zeros(self.E, self.E, dtype=torch.long) for _ in range(self.L)]
        self.pending = [-1] * self.L
        self.prefetch_copies = 0
        self.prefetch_bytes = 0
        self.prefetch_used = 0

        if preload:
            if self.capacity < self.E:
                raise ValueError("preload needs capacity >= E")
            for i in range(self.L):
                for e in range(self.E):
                    self._get(i, e)
            self.reset_stats()

        if verbose:
            exp_mb = sum(blk.mlp.w12[e].weight.numel() + blk.mlp.w3[e].weight.numel()
                         for blk in m.blocks for e in range(self.E)) * 2 / 2**20
            print(f"[moe] E={self.E} layers={self.L} experts_cpu={exp_mb:.1f}MB "
                  f"capacity={self.capacity} preload={preload} prefetch={self.prefetch} "
                  f"device={self.device}")

    # ------------------------------------------------------------------ experts
    def _expert_cpu(self, i: int, e: int):
        mlp = self.model.blocks[i].mlp
        return mlp.w12[e].weight.detach(), mlp.w3[e].weight.detach()

    def _get(self, i: int, e: int):
        cpu12, cpu3 = self._expert_cpu(i, e)
        if self.prefetch:
            return self._get_prefetch(i, e, cpu12, cpu3)
        if self.capacity <= 1:
            slot = self.slots[i]
            if slot is None:
                slot = (torch.empty_like(cpu12, device=self.device, dtype=self.dtype),
                        torch.empty_like(cpu3, device=self.device, dtype=self.dtype))
                self.slots[i] = slot
            if self.last[i] == e:
                self.hits += 1
            else:
                slot[0].copy_(cpu12)
                slot[1].copy_(cpu3)
                self.last[i] = e
                self.copies += 1
                self.bytes_copied += (cpu12.numel() + cpu3.numel()) * cpu12.element_size()
            self.usage[i][e] += 1
            return slot

        cache = self.cache[i]
        if e in cache:
            cache.move_to_end(e)
            self.hits += 1
            self.usage[i][e] += 1
            return cache[e]
        if len(cache) >= self.capacity:
            cache.popitem(last=False)
        w12 = torch.empty_like(cpu12, device=self.device, dtype=self.dtype)
        w3 = torch.empty_like(cpu3, device=self.device, dtype=self.dtype)
        w12.copy_(cpu12)
        w3.copy_(cpu3)
        cache[e] = (w12, w3)
        self.copies += 1
        self.bytes_copied += (cpu12.numel() + cpu3.numel()) * cpu12.element_size()
        self.usage[i][e] += 1
        return w12, w3

    # ------------------------------------------------------------------ M3 prefetch
    def _get_prefetch(self, i: int, e: int, cpu12, cpu3):
        """Two resident slots per layer + transition-based prefetch.

        trans[i][prev, next] counts observed expert transitions within a forward
        pass. After serving e, the most likely successor P(next | e) is copied
        into the other slot with a non_blocking pinned H2D copy.
        """
        st = self.pslots[i]
        if st is None:
            st = {"bufs": [None, None], "owners": [-1, -1]}
            self.pslots[i] = st
        prev = self.last[i]
        if prev >= 0:
            self.trans[i][prev, e] += 1
        self.last[i] = e

        hit_idx = -1
        for si in (0, 1):
            if st["owners"][si] == e:
                hit_idx = si
                break
        if hit_idx >= 0:
            self.hits += 1
            if self.pending[i] == e:
                self.prefetch_used += 1
            self.usage[i][e] += 1
            slot = st["bufs"][hit_idx]
        else:
            victim = 0 if st["owners"][0] != self.pending[i] else 1
            if st["bufs"][victim] is None:
                st["bufs"][victim] = (torch.empty_like(cpu12, device=self.device, dtype=self.dtype),
                                      torch.empty_like(cpu3, device=self.device, dtype=self.dtype))
            st["bufs"][victim][0].copy_(cpu12, non_blocking=True)
            st["bufs"][victim][1].copy_(cpu3, non_blocking=True)
            st["owners"][victim] = e
            self.copies += 1
            self.bytes_copied += (cpu12.numel() + cpu3.numel()) * cpu12.element_size()
            self.usage[i][e] += 1
            slot = st["bufs"][victim]

        self.pending[i] = -1
        row = self.trans[i][e]
        if int(row.sum()) > 0:
            p = int(row.argmax())
            if p != e and p not in st["owners"]:
                other = 1 if st["owners"][0] == e else 0
                pc12, pc3 = self._expert_cpu(i, p)
                if st["bufs"][other] is None:
                    st["bufs"][other] = (torch.empty_like(pc12, device=self.device, dtype=self.dtype),
                                         torch.empty_like(pc3, device=self.device, dtype=self.dtype))
                st["bufs"][other][0].copy_(pc12, non_blocking=True)
                st["bufs"][other][1].copy_(pc3, non_blocking=True)
                st["owners"][other] = p
                self.prefetch_copies += 1
                self.prefetch_bytes += (pc12.numel() + pc3.numel()) * pc12.element_size()
                self.pending[i] = p
        return slot

    # ------------------------------------------------------------------ forward
    def _moe(self, i: int, x: torch.Tensor) -> torch.Tensor:
        mlp = self.model.blocks[i].mlp
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        probs = torch.softmax(mlp.router(xf).float(), dim=-1)
        top_p, top_i = probs.max(dim=-1)
        out = torch.empty_like(xf)
        for e in torch.unique(top_i).tolist():
            m = top_i == e
            w12, w3 = self._get(i, e)
            a, b = F.linear(xf[m], w12).chunk(2, dim=-1)
            y = top_p[m, None] * F.linear(F.silu(a) * b, w3)
            out[m] = y.to(out.dtype)
        return out.reshape(B, T, C)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        ids = ids.to(self.device)
        x = self.model.tok(ids)
        for i, blk in enumerate(self.model.blocks):
            x = x + blk.attn(blk.n1(x), self.cos, self.sin)
            x = x + self._moe(i, blk.n2(x))
        x = self.model.norm(x)
        return self.model.lm_head(x)

    # ------------------------------------------------------------------ stats
    def reset_stats(self):
        self.bytes_copied = 0
        self.copies = 0
        self.hits = 0
        self.prefetch_copies = 0
        self.prefetch_bytes = 0
        self.prefetch_used = 0
        self.pending = [-1] * self.L
        self.usage = [torch.zeros(self.E) for _ in range(self.L)]

    def stats(self) -> dict:
        calls = self.copies + self.hits
        return {
            "expert_copies": self.copies,
            "expert_hits": self.hits,
            "hit_rate": (self.hits / calls) if calls else None,
            "expert_mb": round(self.bytes_copied / 2**20, 3),
            "prefetch_copies": self.prefetch_copies,
            "prefetch_mb": round(self.prefetch_bytes / 2**20, 3),
            "prefetch_used": self.prefetch_used,
            "prefetch_used_frac": (self.prefetch_used / self.prefetch_copies)
                                   if self.prefetch_copies else None,
            "usage": [u.tolist() for u in self.usage],
        }

    # ------------------------------------------------------------------ generate / eval
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, n_new: int, temp: float = 0.8,
                 top_k: int = 100, seed: int = 1234) -> tuple[torch.Tensor, float]:
        ids = prompt_ids.to(self.device)
        gen = torch.Generator(device=self.device).manual_seed(seed)
        t0 = time.time()
        for _ in range(n_new):
            x = ids[:, -self.ctx:]
            logits = self.forward(x)[:, -1, :].float()
            if temp <= 1e-6:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temp
                if top_k and top_k < logits.numel():
                    v, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=gen)
            ids = torch.cat([ids, nxt], dim=1)
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        dt = time.time() - t0
        return ids, n_new / max(dt, 1e-9)

    @torch.no_grad()
    def eval_ce(self, tokens: torch.Tensor, chunk: int = 256) -> float:
        tokens = tokens.to(self.device)
        total, n = 0.0, 0
        for a in range(0, tokens.numel() - 1, chunk):
            ids = tokens[a:a + chunk].unsqueeze(0)
            if ids.shape[1] < 2:
                continue
            logits = self.forward(ids)
            tgt = ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                   tgt, reduction="sum")
            total += float(loss)
            n += tgt.numel()
        return total / max(n, 1)
