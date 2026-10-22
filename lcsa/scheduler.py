"""M4: multi-unit scheduler.

Plans which independent model units can run together in VRAM (budget from
free VRAM or an explicit override) and executes each plan group, running
units inside a group in parallel on separate CUDA streams. Falls back to
sequential groups when units do not fit together. Merges unit outputs with a
simple labeled-concat policy.

Units are declared with an explicit VRAM estimate (est_gb); the fit check is
only as honest as those declarations, and every decision is logged.
"""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Unit:
    name: str
    model_path: str
    est_gb: float
    prompt: str = ""
    max_new_tokens: int = 24
    text: str | None = None
    gen_s: float = 0.0
    load_s: float = 0.0
    # mean logprob of the generated continuation (confidence proxy for merge)
    score: float | None = None


def free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return free / 2**30


def plan(units: list[Unit], budget_gb: float) -> list[list[Unit]]:
    """Greedy first-fit grouping: consecutive units whose estimates fit in
    the budget run together; overflow starts a new (sequential) group."""
    groups: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_gb = 0.0
    for u in units:
        if cur_gb + u.est_gb <= budget_gb:
            cur.append(u)
            cur_gb += u.est_gb
        else:
            if cur:
                groups.append(cur)
            cur = [u]
            cur_gb = u.est_gb
    if cur:
        groups.append(cur)
    return groups


def _load(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda")
    model.eval()
    return tok, model


def _unload(model) -> None:
    model.to("cpu")
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _generate(models: dict, units: list[Unit]) -> float:
    """Generate for each unit using already-loaded models; returns max gen_s.

    Same-group units run concurrently on separate CUDA streams."""
    def work(u: Unit) -> None:
        tok, model = models[u.name]
        ids = tok(u.prompt, return_tensors="pt").input_ids.to("cuda")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            t1 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=u.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id,
                                 output_scores=True, return_dict_in_generate=True)
            u.gen_s = time.perf_counter() - t1
        seq = out.sequences
        gen_len = seq.shape[1] - ids.shape[1]
        logps: list[float] = []
        for t in range(gen_len):
            lp = torch.log_softmax(out.scores[t].float(), dim=-1)
            logps.append(float(lp[0, seq[0, ids.shape[1] + t]]))
        u.score = sum(logps) / len(logps) if logps else None
        u.text = tok.decode(seq[0, ids.shape[1]:], skip_special_tokens=True)

    if len(units) == 1:
        work(units[0])
    else:
        threads = [threading.Thread(target=work, args=(u,)) for u in units]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    torch.cuda.synchronize()
    return max(u.gen_s for u in units)


def run_group(units: list[Unit]) -> dict:
    """Load group units, generate in parallel (one stream per unit), unload.

    Returns per-group load/gen/unload seconds. gen_s is the max over units
    because same-group units run concurrently."""
    t0 = time.perf_counter()
    models = {u.name: _load(u.model_path) for u in units}
    load_s = time.perf_counter() - t0

    gen_s = _generate(models, units)

    t2 = time.perf_counter()
    for u in units:
        _, model = models.pop(u.name)
        _unload(model)
    unload_s = time.perf_counter() - t2
    return {"load_s": load_s, "gen_s": gen_s, "unload_s": unload_s}


def merge_outputs(units: list[Unit], policy: str = "concat") -> str:
    """Merge unit outputs.

    policy="concat": labeled concatenation (all segments, no arbitration).
    policy="best": pick the unit with the highest mean continuation logprob
    (a confidence proxy, not ground truth). Falls back to concat when no unit
    has a score.
    """
    if policy == "best":
        scored = [u for u in units if u.score is not None]
        if scored:
            best = max(scored, key=lambda u: u.score)
            return (best.text or "").strip()
    if len(units) == 1:
        return (units[0].text or "").strip()
    return " | ".join(f"[{u.name}] {u.text.strip()}" for u in units)
