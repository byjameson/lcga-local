"""P3: kendi mini dil modelimiz (sifirdan, TinyStories icin).

Decoder-only transformer: byte-level BPE (8k) -> RMSNorm + RoPE + causal SDPA +
SwiGLU MLP, tied embeddings. ~17M param (dim 384 / 8 katman / 6 head / ctx 512).

Basit ve denetlenebilir: egitim exp11_own_lm.py; ornek uretim MiniLM.generate().
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def build_rope(head_dim, ctx, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(ctx).float()
    freqs = torch.outer(t, inv)                       # (ctx, head_dim/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd); eslesik (interleaved) RoPE."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, : x1.size(-2), :].to(x.dtype)
    s = sin[None, None, : x1.size(-2), :].to(x.dtype)
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.h, self.hd = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.h, self.hd).transpose(1, 2)
        k = k.view(B, T, self.h, self.hd).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class MoESwiGLU(nn.Module):
    """exp15 (MoE-lite): E uzmanli top-1 MoE (Switch stili).

    y = p_top(x) * Uzman_top(x); router gradyani p_top uzerinden akar.
    Yuk dengeleme aux'u loss'a eklenir (alpha=0.01; Switch formulu)."""

    def __init__(self, dim, hidden, experts):
        super().__init__()
        self.E = int(experts)
        self.router = nn.Linear(dim, self.E, bias=True)
        self.w12 = nn.ModuleList([nn.Linear(dim, 2 * hidden, bias=False)
                                  for _ in range(self.E)])
        self.w3 = nn.ModuleList([nn.Linear(hidden, dim, bias=False)
                                 for _ in range(self.E)])
        self.alpha = 0.01
        self.aux = None
        self.last_usage = None

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        probs = torch.softmax(self.router(xf).float(), dim=-1)
        top_p, top_i = probs.max(dim=-1)
        onehot = F.one_hot(top_i, self.E).float()
        f = onehot.mean(0)
        self.aux = self.alpha * self.E * (f * probs.mean(0)).sum()
        self.last_usage = f.detach()
        out = torch.empty_like(xf)
        for e in range(self.E):
            m = (top_i == e)
            if bool(m.any()):
                a, b = self.w12[e](xf[m]).chunk(2, dim=-1)
                out[m] = (top_p[m, None] * self.w3[e](F.silu(a) * b)).to(out.dtype)
        return out.reshape(B, T, C)


class Block(nn.Module):
    def __init__(self, dim, heads, hidden, experts=0):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.n2 = RMSNorm(dim)
        self.mlp = (SwiGLU(dim, hidden) if experts <= 0
                    else MoESwiGLU(dim, hidden, experts))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.mlp(self.n2(x))
        return x


class MiniLM(nn.Module):
    def __init__(self, vocab=8192, dim=384, layers=8, heads=6, ctx=512,
                 mlp_hidden=1024, loops=1, moe_experts=0):
        super().__init__()
        self.vocab, self.ctx = vocab, ctx
        # P4: dongulu derinlik -- ayni bloklar `loops` kez uygulanir (parametre sabit,
        # hesap artar). loops=1 -> eski davranis BIREBIR ayni.
        self.loops = int(loops)
        # exp15 (MoE-lite): moe_experts>0 -> her MLP yerine E uzmanli top-1 MoE;
        # mlp_hidden bu durumda UZMAN BASINA gizli boyuttur (esit param icin ~/E).
        self.moe_experts = int(moe_experts)
        self.tok = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_hidden,
                                           experts=self.moe_experts)
                                     for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.tok.weight            # tied
        cos, sin = build_rope(dim // heads, ctx)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init)
        std_r = 0.02 / math.sqrt(2 * layers)             # residual proj olcekleme
        for b in self.blocks:
            nn.init.normal_(b.attn.proj.weight, std=std_r)
            w3s = b.mlp.w3 if self.moe_experts > 0 else [b.mlp.w3]
            for w3 in w3s:
                nn.init.normal_(w3.weight, std=std_r)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def cfg(self):
        b = self.blocks[0]
        w12 = b.mlp.w12[0] if self.moe_experts > 0 else b.mlp.w12
        return {"arch": "minilm", "vocab": self.vocab,
                "dim": self.tok.embedding_dim, "layers": len(self.blocks),
                "heads": b.attn.h, "ctx": self.ctx,
                "mlp_hidden": w12.out_features // 2,
                "loops": self.loops, "moe": self.moe_experts}

    def forward(self, idx, return_hidden=False):
        x = self.tok(idx)
        cos, sin = self.cos, self.sin
        for _ in range(self.loops):                      # loops=1: eski yol (birebir)
            for blk in self.blocks:
                x = blk(x, cos, sin)
        x = self.norm(x)
        if return_hidden:                                # loss icin: lm_head'siz
            return x
        return self.lm_head(x)

    def loss(self, idx, chunk=4096, add_aux=False):
        """Ortalama next-token CE; lm_head de chunk'li (bellek-guvenli).

        add_aux=True (exp15/MoE): katmanlarin yuk-dengeleme aux kaybi eklenir.
        P3-perf: eskiden lm_head TAM (B*T, vocab) logits uretiyordu (~0.4G bf16
        + ~0.4G backward) -> 6GB kartta WDDM tasmasi. Simdi logits chunk chunk
        hesaplanir; matematik BIREBIR ayni (CE zaten chunk'liydi)."""
        h = self.forward(idx[:, :-1], return_hidden=True)
        tgt = idx[:, 1:].reshape(-1)
        hf = h.reshape(-1, h.size(-1))
        w = self.lm_head.weight
        tot = None
        for i in range(0, tgt.numel(), chunk):
            lg = hf[i: i + chunk] @ w.t()                # (c, vocab) autocast bf16
            # .float() YOK: bf16 logits'i dogrudan CE'ye ver; CE icinde fp32 akumulasyon var.
            c = F.cross_entropy(lg, tgt[i: i + chunk], reduction="sum")
            tot = c if tot is None else tot + c
        out = tot / tgt.numel()
        if add_aux:
            aux = None
            for blk in self.blocks:
                a = getattr(blk.mlp, "aux", None)
                if a is not None:
                    aux = a if aux is None else aux + a
            if aux is not None:
                out = out + aux
        return out

    @torch.no_grad()
    def generate(self, prompt_ids, n_new, temp=0.8, top_k=100, eos_id=None):
        """prompt_ids: (1, T0) int64 (CPU veya GPU). Naif (KV-cache'siz)."""
        dev = next(self.parameters()).device
        ids = prompt_ids.to(dev)
        use_amp = dev.type == "cuda"
        for _ in range(n_new):
            x = ids[:, -self.ctx:]
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=use_amp):
                logits = self.forward(x)[:, -1, :].float()
            logits = logits / max(temp, 1e-6)
            if top_k and top_k < logits.numel():
                v, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < v[..., -1, None], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            if eos_id is not None and int(nxt) == eos_id:
                break
        return ids
