"""Evaluation metrics: PPL, KL, top-k agreement, routing diagnostics, timing."""
from __future__ import annotations

import json
import os
import statistics
import time

import torch
import torch.nn.functional as F


def cuda_time_ms(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True)
        e = torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return {"median": round(statistics.median(ts), 4), "min": round(min(ts), 4),
            "mean": round(statistics.mean(ts), 4)}


@torch.no_grad()
def eval_pair(teacher_logits_fn, student_logits_fn, chunks, device, max_chunks=None,
              pos_chunk=256):
    """Compare teacher vs student next-token distributions over token chunks.

    Returns dict with CE/PPL for both, KL(teacher||student) and top-k agreements.
    """
    stats = {"ce_t": 0.0, "ce_s": 0.0, "kl_ts": 0.0, "kl_st": 0.0,
             "top1": 0.0, "top5": 0.0, "n": 0}
    n_chunks = len(chunks) if max_chunks is None else min(len(chunks), max_chunks)
    for ci in range(n_chunks):
        ids = chunks[ci].to(device).unsqueeze(0)  # (1,T)
        t_logits = teacher_logits_fn(ids)
        s_logits = student_logits_fn(ids)
        # predict tokens 1..T-1
        labels = ids[0, 1:]
        lt = t_logits[0, :-1]
        ls = s_logits[0, :-1]
        T = labels.shape[0]
        for p0 in range(0, T, pos_chunk):
            p1 = min(T, p0 + pos_chunk)
            a = lt[p0:p1].float()
            b = ls[p0:p1].float()
            lab = labels[p0:p1]
            la = F.log_softmax(a, dim=-1)
            lb = F.log_softmax(b, dim=-1)
            stats["ce_t"] += F.nll_loss(la, lab, reduction="sum").item()
            stats["ce_s"] += F.nll_loss(lb, lab, reduction="sum").item()
            stats["kl_ts"] += F.kl_div(lb, la, log_target=True, reduction="sum").item()
            stats["kl_st"] += F.kl_div(la, lb, log_target=True, reduction="sum").item()
            t1 = a.argmax(-1)
            s1 = b.argmax(-1)
            stats["top1"] += (t1 == s1).sum().item()
            tk = torch.topk(a, 5, dim=-1).indices
            stats["top5"] += (tk == s1.unsqueeze(-1)).any(-1).sum().item()
            stats["n"] += lab.numel()
        del t_logits, s_logits
    n = max(stats["n"], 1)
    out = {
        "ntok": n,
        "ce_teacher": stats["ce_t"] / n,
        "ce_student": stats["ce_s"] / n,
        "ppl_teacher": float(torch.exp(torch.tensor(stats["ce_t"] / n))),
        "ppl_student": float(torch.exp(torch.tensor(stats["ce_s"] / n))),
        "kl_teacher_student": stats["kl_ts"] / n,
        "kl_student_teacher": stats["kl_st"] / n,
        "top1_agreement": stats["top1"] / n,
        "top5_agreement": stats["top5"] / n,
    }
    return out


def routing_diagnostics(captured, n_layers=None):
    """Block-mass recall of the selected blocks per layer (from capture entries)."""
    per_layer = []
    for entry in captured:
        if "mass" not in entry or "idx" not in entry:
            continue
        mass = entry["mass"]           # (B,H,NB,NB) normalized over J
        idx = entry["idx"]             # (B,NB,MMAX)
        B, H, NB, _ = mass.shape
        MMAX = idx.shape[2]
        sel = torch.zeros(B, 1, NB, NB, device=mass.device, dtype=torch.bool)
        valid = idx >= 0
        for j in range(MMAX):
            jj = idx[:, :, j].clamp_min(0).long()  # (B,NB)
            sel.scatter_(3, jj[:, None, :, None], True)
        sel = sel & valid.any(-1)[:, None, :, None]
        recall = (mass * sel).sum(-1) / mass.sum(-1).clamp_min(1e-9)  # (B,H,NB)
        per_layer.append({"recall_mean": float(recall.mean()), "recall_min": float(recall.min())})
    return per_layer


def append_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")
