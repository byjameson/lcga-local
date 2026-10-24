"""P2: tokenizer'sız byte-level prototip — düz (flat) byte LM vs iki-katmanlı
sabit-patch byte LM ("BytePatch", BLT-lite).

Fikir: sözlük = 256 ham UTF-8 byte'ı. İki-katmanlı model, byte'ları P'lik
sabit patch'lere bölüp önce yerel bir encoder ile patch latentine indirger
(global adım sayısı = byte sayısı / P), global causal transformer patch
latentleri üzerinde koşar, sonra küçük bir yerel decoder bir sonraki patch'in
byte'larını autoregressive üretir. Kayıp = sonraki-patch byte CE; doğrudan
bit/byte (bpb) olarak okunabilir. Düz model aynı bütçe/ölçekte bir baseline.

Tüm modeller küçük (≈5M param), wikitext-2 ham byte'ları üzerinde eğitilir.
Dürüstlük notları:
- P=4'te global adım/byte = 0.25; yerel decoder adımları hesabı ayrı.
- Yerel encoder patch içinde çift yönlüdür (BLT ile aynı tercih); patch'ler
  arası ve patch içi çözümleme (decoder) nedenseldir → tahmin sızıntısı yok.
- Bu bir prototip: dinamik parçalama, KV-cache'li streaming decode ve büyük
  ölçek sonraki işler (bkz. P2_RAPOR.md).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

LN2 = math.log(2.0)


# ---------------------------------------------------------------- data utils
def build_byte_data(data_dir, force=False):
    """wikitext-2 (train/valid/test) ham metnini UTF-8 byte tensor'larına indir."""
    from .data import get_wikitext
    os.makedirs(data_dir, exist_ok=True)
    out = {}
    jobs = [("wt2_train", "train"), ("wt2_valid", "validation"), ("wt2_test", "test")]
    for name, split in jobs:
        path = os.path.join(data_dir, f"bytes_{name}.pt")
        if os.path.exists(path) and not force:
            out[name] = path
            continue
        text = get_wikitext("wikitext-2-raw-v1", split)
        b = bytearray(text.encode("utf-8"))
        torch.save(torch.frombuffer(b, dtype=torch.uint8).clone(), path)
        print(f"[bytes] {name}: {len(b)} -> {path}", flush=True)
        out[name] = path
    return out


def load_bytes(path):
    return torch.load(path, weights_only=True)


def sample_windows(data, W, batch, generator):
    """Rastgele ardışık pencereler -> (batch, W) int64."""
    max_start = len(data) - W - 1
    starts = torch.randint(0, max_start, (batch,), generator=generator).tolist()
    return torch.stack([data[s: s + W].long() for s in starts])


def fixed_windows(data, W, n):
    return data[: n * W].view(n, W).long()


def bpb(nll_sum, count):
    return float(nll_sum) / float(count) / LN2


@torch.no_grad()
def eval_bpb(model, data, W, n_windows, batch, device):
    model.eval()
    wins = fixed_windows(data, W, n_windows)
    tot, cnt = 0.0, 0
    for i in range(0, len(wins), batch):
        x = wins[i: i + batch].to(device)
        nll, c = model.byte_loss(x)
        tot += float(nll)
        cnt += int(c)
    model.train()
    return bpb(tot, cnt)


# --------------------------------------------------------------------- block
def _init_weights(m):
    """küçük-LM init'i: std 0.02 lineer/embedding, sıfır bias (nanoGPT tarzı)."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_mult=4, causal=True):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.causal = causal
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, mlp_mult * dim)
        self.fc2 = nn.Linear(mlp_mult * dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = q.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, D // self.heads).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, D))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


# -------------------------------------------------------------- flat byte LM
class FlatByteLM(nn.Module):
    """Standart nedensel transformer, doğrudan byte dizisi üzerinde."""

    def __init__(self, W=512, dim=256, layers=6, heads=8):
        super().__init__()
        self.W = W
        self.emb = nn.Embedding(256, dim)
        self.pos = nn.Embedding(W, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.apply(_init_weights)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def cfg(self):
        return {"arch": "flat", "W": self.W, "dim": self.emb.embedding_dim,
                "layers": len(self.blocks), "heads": self.blocks[0].heads}

    def forward(self, x):
        B, W = x.shape
        assert W <= self.W
        h = self.emb(x) + self.pos.weight[:W].view(1, W, -1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return torch.einsum("btd,vd->btv", h, self.emb.weight)

    def byte_loss(self, x):
        """(sum NLL natsı, sayılan byte sayısı) — ilk byte (bağlamsız) hariç."""
        logits = self.forward(x).float()
        tgt = x[:, 1:]
        nll = F.cross_entropy(logits[:, :-1].reshape(-1, 256), tgt.reshape(-1),
                              reduction="sum")
        return nll, int(tgt.numel())


# ----------------------------------------------------------- patch (BLT-lite)
class BytePatchLM(nn.Module):
    """İki-katmanlı byte LM: yerel encoder -> patch latentleri -> global
    transformer -> yerel autoregressive decoder (sonraki patch byte'ları)."""

    def __init__(self, P=4, W=512, byte_dim=64, dim=256, layers=6, heads=8,
                 lo_layers=2, lo_heads=4):
        super().__init__()
        assert W % P == 0
        self.P, self.W = P, W
        self.L = W // P
        self.emb = nn.Embedding(256, byte_dim)
        # yerel encoder (patch içi, çift yönlü)
        self.enc_pos = nn.Embedding(P, byte_dim)
        self.enc = nn.ModuleList([Block(byte_dim, lo_heads, causal=False)
                                  for _ in range(lo_layers)])
        self.patch_proj = nn.Linear(byte_dim, dim)
        # global transformer (patch'ler arası, nedensel)
        self.pos = nn.Embedding(self.L, dim)
        self.start = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm_in = nn.LayerNorm(dim)
        # yerel decoder (patch içi, nedensel; h_{i-1} + önceki byte'lar)
        self.h_proj = nn.Linear(dim, byte_dim)
        self.dec_pos = nn.Embedding(P, byte_dim)
        self.dec = nn.ModuleList([Block(byte_dim, lo_heads) for _ in range(lo_layers)])
        self.norm_out = nn.LayerNorm(byte_dim)
        self.apply(_init_weights)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * len(self.blocks)))

    def cfg(self):
        return {"arch": "patch", "P": self.P, "W": self.W,
                "byte_dim": self.emb.embedding_dim, "dim": self.pos.embedding_dim,
                "layers": len(self.blocks), "heads": self.blocks[0].heads,
                "lo_layers": len(self.enc), "lo_heads": self.enc[0].heads}

    def patch_states(self, xp):
        """xp (B,L,P) -> h (B,L,dim): her patch'in global latent temsili."""
        B, L, P = xp.shape
        e = self.emb(xp) + self.enc_pos.weight.view(1, 1, P, -1)
        e = e.reshape(B * L, P, -1)
        for blk in self.enc:
            e = blk(e)
        pe = self.patch_proj(e.mean(dim=1)).view(B, L, -1)
        h = pe + self.pos.weight[:L].view(1, L, -1)
        for blk in self.blocks:
            h = blk(h)
        return self.norm_in(h)

    def decode_logits(self, h_in, xp):
        """h_in (B,L,dim): patch i'yi bu h ile tahmin et (h_{-1}=start).
        xp (B,L,P): teacher-forcing byte'ları. -> logits (B,L,P,256)."""
        B, L, P = xp.shape
        ctx = self.h_proj(h_in).unsqueeze(2)                      # (B,L,1,bd)
        prev = self.emb(xp[:, :, :-1])                            # (B,L,P-1,bd)
        din = torch.cat([ctx, prev], dim=2)                       # (B,L,P,bd)
        din = din + self.dec_pos.weight.view(1, 1, P, -1)
        din = din.reshape(B * L, P, -1)
        for blk in self.dec:
            din = blk(din)
        din = self.norm_out(din)
        logits = torch.einsum("npe,ve->npv", din, self.emb.weight)
        return logits.view(B, L, P, -1)

    def forward(self, x):
        B, W = x.shape
        assert W == self.W
        xp = x.view(B, self.L, self.P)
        h = self.patch_states(xp)
        h_in = torch.cat([self.start.expand(B, 1, -1), h[:, :-1]], dim=1)
        return self.decode_logits(h_in, xp)

    def byte_loss(self, x):
        """(sum NLL natsı, sayılan byte sayısı) — pencerenin ilk byte'ı hariç
        (düz modelle aynı hedef kümesi)."""
        B, W = x.shape
        logits = self.forward(x).float()                          # (B,L,P,256)
        tgt = x.view(B, -1)                                       # (B,L*P)
        nll = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1),
                              reduction="none").view(B, self.L, self.P)
        m = torch.ones_like(nll)
        m[:, 0, 0] = 0.0
        return (nll * m).sum(), int(m.sum())


# -------------------------------------------------------------- generation
def sample_topk(logits, temp, top_k, gen):
    logits = logits.float() / max(temp, 1e-6)
    if top_k and top_k < logits.numel():
        v, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < v[-1], float("-inf"))
    probs = torch.softmax(logits, -1).cpu()
    return int(torch.multinomial(probs, 1, generator=gen))


@torch.no_grad()
def generate_flat(model, n_bytes, device, temp=0.8, top_k=50, seed=0,
                  prompt=b"\n"):
    """Naif O(n^2) örnekleme (demo amaçlı; KV-cache yok)."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    buf = bytearray(prompt)
    while len(buf) < n_bytes:
        window = bytearray(buf[-model.W:])
        x = torch.frombuffer(window, dtype=torch.uint8).long().view(1, -1).to(device)
        logits = model(x)[0, -1]
        buf.append(sample_topk(logits, temp, top_k, g))
    return bytes(buf)


@torch.no_grad()
def generate_patch(model, n_bytes, device, temp=0.8, top_k=50, seed=0,
                   prompt=b"\n" * 4):
    """Patch-at-a-time üretim (naif; her patch'te tam recompute)."""
    model.eval()
    P = model.P
    g = torch.Generator().manual_seed(seed)
    buf = bytearray(prompt[: len(prompt) // P * P])
    while len(buf) < n_bytes:
        ctx_bytes = bytearray(buf[-model.W:])                # patch-hizalı kaydırma
        L = len(ctx_bytes) // P
        if L == 0:
            h_last = model.start                             # (1,1,dim)
        else:
            x = torch.frombuffer(ctx_bytes, dtype=torch.uint8).long()
            xp = x.view(1, L, P).to(device)
            h = model.patch_states(xp)
            h_last = h[:, -1]                                # (1,dim)
        ctx = model.h_proj(h_last).view(1, 1, -1)            # (1,1,bd)
        cur = []
        for j in range(P):
            parts = [ctx]
            if j > 0:
                pb = torch.tensor(cur, dtype=torch.long, device=device)
                parts.append(model.emb(pb).view(1, -1, model.emb.embedding_dim))
            din = torch.cat(parts, dim=1)
            din = din + model.dec_pos.weight[: j + 1].view(1, -1,
                                                           model.emb.embedding_dim)
            for blk in model.dec:
                din = blk(din)
            din = model.norm_out(din)
            logits = torch.einsum("npe,ve->npv", din[:, -1:],
                                  model.emb.weight)[0, 0]
            cur.append(sample_topk(logits, temp, top_k, g))
        buf.extend(cur)
    model.train()
    return bytes(buf)
