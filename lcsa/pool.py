"""M4b: model pool with LRU eviction.

The base scheduler loads and unloads every request; with repeated traffic the load
time dominates. The pool keeps models resident on the GPU and moves the least
recently used ones to host RAM when a new model would exceed the VRAM budget.
Reloading an evicted model is a host->device copy (no disk), measured separately.

Honest limits: est_gb is declared by the caller, not measured; evicted models still
occupy host RAM; allocator fragmentation is not modelled.
"""
from __future__ import annotations

import gc
import os
import time
from collections import OrderedDict

import torch

from lcsa.scheduler import _generate, _load


class ModelPool:
    def __init__(self, budget_gb: float, device: str = "cuda",
                 verbose: bool = True):
        self.budget_gb = budget_gb
        self.device = device
        self.verbose = verbose
        self.entries: OrderedDict[str, dict] = OrderedDict()
        self.counters = {"evictions": 0, "reloads": 0, "disk_loads": 0,
                         "reload_s": 0.0, "load_s": 0.0}

    def _resident_gb(self) -> float:
        return sum(e["gb"] for e in self.entries.values() if e["on_gpu"])

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[pool] {msg}", flush=True)

    def _evict_one(self, exclude: str | None = None) -> bool:
        """Move the least-recently-used GPU resident (other than `exclude`) to host.

        Returns False when there is no evictable resident left.
        """
        victim = next((p for p, x in self.entries.items()
                       if x["on_gpu"] and p != exclude), None)
        if victim is None:
            return False
        old = self.entries[victim]
        t0 = time.perf_counter()
        old["model"].to("cpu")
        old["on_gpu"] = False
        gc.collect()
        torch.cuda.empty_cache()
        self.counters["evictions"] += 1
        self._log(f"evict {os.path.basename(victim)} "
                  f"({time.perf_counter()-t0:.2f}s, gpu->cpu, kept in pool)")
        self.entries.move_to_end(victim)
        return True

    def get(self, path: str, est_gb: float) -> dict:
        e = self.entries.get(path)
        if e is not None:
            self.entries.move_to_end(path)
            if not e["on_gpu"]:
                # the budget applies to reloads too, not only to disk loads
                while self._resident_gb() + e["gb"] > self.budget_gb:
                    if not self._evict_one(exclude=path):
                        break
                t0 = time.perf_counter()
                e["model"].to(self.device)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                e["on_gpu"] = True
                self.counters["reloads"] += 1
                self.counters["reload_s"] += dt
                self._log(f"reload {os.path.basename(path)} ({dt:.2f}s, host->gpu; "
                          f"resident {self._resident_gb():.2f}GB)")
            return e

        # evict least-recently-used GPU residents until the new model fits
        while self._resident_gb() + est_gb > self.budget_gb:
            if not self._evict_one(exclude=path):
                break

        t0 = time.perf_counter()
        tok, model = _load(path)
        dt = time.perf_counter() - t0
        self.counters["disk_loads"] += 1
        self.counters["load_s"] += dt
        e = {"tok": tok, "model": model, "gb": est_gb, "on_gpu": True}
        self.entries[path] = e
        self._log(f"load {os.path.basename(path)} ({dt:.2f}s, disk->gpu; "
                  f"resident {self._resident_gb():.2f}GB)")
        return e

    def run(self, units: list) -> dict:
        """Serve a group through the pool (no unload at the end)."""
        t0 = time.perf_counter()
        models = {}
        for u in units:
            e = self.get(u.model_path, u.est_gb)
            models[u.name] = (e["tok"], e["model"])
        load_s = time.perf_counter() - t0
        gen_s = _generate(models, units)
        return {"load_s": load_s, "gen_s": gen_s, "unload_s": 0.0}

    def stats(self) -> dict:
        base = os.path.basename
        return {
            "budget_gb": self.budget_gb,
            "resident": [base(p) for p, e in self.entries.items() if e["on_gpu"]],
            "host": [base(p) for p, e in self.entries.items() if not e["on_gpu"]],
            "resident_gb": round(self._resident_gb(), 2),
            **{k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in self.counters.items()},
        }

    def evict_all(self) -> float:
        """Move every GPU-resident model to host RAM (keeps entries for fast reload)."""
        t0 = time.perf_counter()
        for e in self.entries.values():
            if e["on_gpu"]:
                e["model"].to("cpu")
                e["on_gpu"] = False
        gc.collect()
        torch.cuda.empty_cache()
        dt = time.perf_counter() - t0
        self.counters["evictions"] += 1
        self._log(f"evict all -> host ({dt:.2f}s)")
        return dt

    def clear(self) -> None:
        for e in self.entries.values():
            if e["on_gpu"]:
                e["model"].to("cpu")
                e["on_gpu"] = False
        self.entries.clear()
        gc.collect()
        torch.cuda.empty_cache()
