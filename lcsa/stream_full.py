"""Full-parameter block-streaming trainer (P1 extension).

P1's StreamingLoRATrainer froze the base model and trained LoRA adapters. This
trainer keeps the *entire* model trainable while only one decoder block is on
the GPU at a time:

  forward : embed on GPU -> [block_i to GPU, run, cache input, to CPU] x L
  loss    : chunked CE through the tied lm_head (bounded logits)
  backward: [block_i to GPU, recompute with grad, backward, UPDATE on GPU,
             to CPU] x L (reverse)
  embed   : input-embedding gradient recomputed once at the end (dense grad)
  update  : SGD or Adafactor (factored second moment; state ~O(rows+cols))

Peak VRAM = one block + its activations + embed/head + one layer's grads.
Adafactor keeps only row/col factor vectors (a few MB for 3B), so it fits the
no-optimizer-state design; SGD remains the fallback for exact reproduction.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from lcsa.stream_train import _layer_out, chunked_lm_loss_and_backward


class StreamingSGDTrainer:
    def __init__(self, model, device: str = "cuda", loss_chunk: int = 128,
                 lr: float = 2e-5, clip: float = 1.0, weight_decay: float = 0.0,
                 opt: str = "sgd", adaf_beta2: float = 0.999,
                 adaf_eps: float = 1e-8, warmup: int = 0, master: bool = False,
                 verbose: bool = False, profile: bool = False,
                 prefetch: bool = False, buffered: bool = False,
                 pinned: bool = False):
        self.device = device
        self.loss_chunk = loss_chunk
        self.lr = lr
        self.clip = clip
        self.wd = weight_decay
        self.opt = opt
        self.adaf_beta2 = adaf_beta2
        self.adaf_eps = adaf_eps
        self.warmup = warmup
        self.profile = profile
        self.prof: dict = {}
        self.prefetch = prefetch
        self._pf_stream = torch.cuda.Stream() if prefetch else None
        self._pf_pending = False
        self.buffered = bool(buffered or pinned)
        self.pinned = bool(pinned)
        if self.buffered and master:
            raise ValueError("buffered/pinned mode does not support --master")
        if self.buffered and prefetch:
            raise ValueError("buffered/pinned mode does not support --prefetch")
        self._cpu_w: dict = {}
        self._gpu_pool: dict = {}
        self._pin_pool: dict = {}
        self.opt_state: dict = {}
        self.opt_state_cpu: dict = {}
        self.use_master = master
        self.lr_kind = "none"
        self.min_lr = 0.0
        self.total_steps = 0
        self.verbose = verbose
        self.model = model

        base = model.model
        self.layers = list(base.layers)
        self.embed = base.embed_tokens
        self.rotary = base.rotary_emb
        self.norm = base.norm
        self.lm_head = model.lm_head

        model.to("cpu")
        for p in model.parameters():
            p.requires_grad_(True)

        # buffered/pinned mode keeps per-block GPU buffers and swaps p.data, so
        # the forward pass never needs to copy weights back to the host.
        if self.buffered:
            for layer in self.layers:
                for n, p in layer.named_parameters():
                    self._cpu_w[id(p)] = p.data
            print(f"[buffered] pinned={self.pinned} layers={len(self.layers)}", flush=True)

        # small/shared modules stay resident on GPU
        self.embed.to(device)
        self.rotary.to(device)
        self.norm.to(device)
        self.tied = self.lm_head.weight is self.embed.weight
        if not self.tied:
            self.lm_head.to(device)

        self.n_params = sum(p.numel() for p in model.parameters())
        self.steps_done = 0

        # fp32 master weights on CPU (optional): updates accumulate in fp32 so
        # they are not lost to bf16 weight quantization; bf16 params are
        # refreshed from the master after each update.
        self.master_state: dict = {}
        if master:
            for i, layer in enumerate(self.layers):
                for n, p in layer.named_parameters():
                    self.master_state[f"L{i}.{n}"] = p.detach().float().cpu().clone()
            self.master_state["embed.weight"] = self.embed.weight.detach().float().cpu().clone()
            self.master_state["norm.weight"] = self.norm.weight.detach().float().cpu().clone()
            if not self.tied:
                self.master_state["head.weight"] = self.lm_head.weight.detach().float().cpu().clone()
            gb = sum(t.numel() for t in self.master_state.values()) * 4 / 1e9
            try:
                import psutil
                avail = psutil.virtual_memory().available / 1e9
                print(f"[master] fp32 state={gb:.2f}GB host_ram_available={avail:.1f}GB",
                      flush=True)
            except ImportError:
                print(f"[master] fp32 state={gb:.2f}GB", flush=True)

    # ------------------------------------------------------------------ forward
    def _to(self, mod, dst: str) -> None:
        """Move a module and (when profiling) accumulate transfer wall time."""
        if not self.profile:
            mod.to(dst)
            return
        t = time.perf_counter()
        mod.to(dst)
        key = "h2d_s" if dst == self.device else "d2h_s"
        self.prof[key] = self.prof.get(key, 0.0) + (time.perf_counter() - t)

    # ---------------------------------------------------------------- buffered
    def _key(self, n, p):
        return (n, tuple(p.shape), p.dtype)

    def _load(self, mod) -> None:
        """Bring one block's weights to the GPU (buffered) or move the module."""
        if not self.buffered:
            self._to(mod, self.device)
            return
        t0 = time.perf_counter()
        t_host = 0.0
        for n, p in mod.named_parameters():
            key = self._key(n, p)
            src = self._cpu_w[id(p)]
            gpu = self._gpu_pool.get(key)
            if gpu is None:
                gpu = torch.empty(p.shape, dtype=p.dtype, device=self.device)
                self._gpu_pool[key] = gpu
            if self.pinned:
                pin = self._pin_pool.get(key)
                if pin is None:
                    pin = torch.empty(p.shape, dtype=p.dtype, pin_memory=True)
                    self._pin_pool[key] = pin
                t = time.perf_counter()
                pin.copy_(src)
                t_host += time.perf_counter() - t
                gpu.copy_(pin, non_blocking=True)
            else:
                gpu.copy_(src)
            p.data = gpu
        if self.pinned:
            torch.cuda.synchronize()
        if self.profile:
            self.prof["h2d_s"] = self.prof.get("h2d_s", 0.0) + (
                time.perf_counter() - t0 - t_host)
            self.prof["host_copy_s"] = self.prof.get("host_copy_s", 0.0) + t_host

    def _unload(self, mod, writeback: bool) -> None:
        """Release a block's GPU buffers (optionally writing updated weights back)."""
        if not self.buffered:
            self._to(mod, "cpu")
            return
        if writeback:
            t0 = time.perf_counter()
            t_host = 0.0
            if self.pinned:
                for n, p in mod.named_parameters():
                    self._pin_pool[self._key(n, p)].copy_(p.data, non_blocking=True)
                torch.cuda.synchronize()
                for n, p in mod.named_parameters():
                    t = time.perf_counter()
                    self._cpu_w[id(p)].copy_(self._pin_pool[self._key(n, p)])
                    t_host += time.perf_counter() - t
            else:
                for n, p in mod.named_parameters():
                    self._cpu_w[id(p)].copy_(p.data)
            if self.profile:
                self.prof["d2h_s"] = self.prof.get("d2h_s", 0.0) + (
                    time.perf_counter() - t0 - t_host)
                self.prof["host_copy_s"] = self.prof.get("host_copy_s", 0.0) + t_host
        for n, p in mod.named_parameters():
            p.data = self._cpu_w[id(p)]
            p.grad = None

    # ---------------------------------------------------------------- prefetch
    def _prefetch(self, mod) -> None:
        """Start an async host->device copy of `mod` on a side stream."""
        with torch.cuda.stream(self._pf_stream):
            mod.to(self.device, non_blocking=True)
        self._pf_pending = True

    def _wait_prefetch(self) -> None:
        """Make the compute stream wait for a pending prefetch copy."""
        if not self.prefetch:
            return
        if self._pf_pending:
            torch.cuda.current_stream().wait_stream(self._pf_stream)
            self._pf_pending = False

    @torch.no_grad()
    def _forward_cached(self, ids):
        dev = self.device
        h = self.embed(ids.to(dev))
        pos_ids = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        pos = self.rotary(h, pos_ids)
        cache = []
        for i, layer in enumerate(self.layers):
            self._wait_prefetch()
            self._load(layer)
            if self.prefetch and i + 1 < len(self.layers):
                self._prefetch(self.layers[i + 1])
            cache.append(h)
            t = time.perf_counter()
            h = _layer_out(layer(h, position_embeddings=pos))
            if self.profile:
                torch.cuda.synchronize()
                self.prof["fwd_s"] = self.prof.get("fwd_s", 0.0) + (time.perf_counter() - t)
            self._unload(layer, writeback=False)
            if self.verbose and (i + 1) % 8 == 0:
                print(f"    [fwd] {i + 1}/{len(self.layers)} "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        return h, cache, pos

    # ------------------------------------------------------------------ update
    def set_schedule(self, total: int, kind: str = "cos", min_lr: float = 0.0) -> None:
        self.total_steps = total
        self.lr_kind = kind
        self.min_lr = min_lr

    def _lr_now(self) -> float:
        t = self.steps_done + 1
        lr = self.lr
        if self.lr_kind == "cos":
            prog = min(1.0, t / max(1, self.total_steps))
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (1 + math.cos(math.pi * prog))
        if self.warmup and self.warmup > 0:
            lr = lr * min(1.0, t / self.warmup)
        return lr

    def _update(self, named):
        """Dispatch a list of (state_key, param) pairs to the chosen optimizer."""
        ps = [(n, p) for n, p in named if p.requires_grad and p.grad is not None]
        if not ps:
            return
        if self.clip and self.clip > 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in ps], self.clip)
        lr = self._lr_now()
        if self.opt == "sgd":
            with torch.no_grad():
                for _, p in ps:
                    if self.wd:
                        p.mul_(1.0 - lr * self.wd)
                    p.add_(p.grad, alpha=-lr)
                    p.grad = None
        elif self.opt == "adafactor":
            self._adafactor(ps, self.steps_done + 1, lr)
        else:
            raise ValueError(f"unknown opt: {self.opt}")

    def _adafactor(self, ps, t, lr):
        """Adafactor with bias correction: factored second moment (row/col means).

        upd = g / sqrt(v_hat), v_hat = v / (1 - b2**t). The update is applied
        in-place on a single fp32 grad copy (scale factors instead of building
        the full outer-product matrix) to bound temporary VRAM.
        """
        b2 = self.adaf_beta2
        eps = self.adaf_eps
        bc = 1.0 - b2 ** t
        with torch.no_grad():
            for n, p in ps:
                gf = p.grad.float()
                if p.ndim >= 2:
                    st = self.opt_state.get(n)
                    if st is None or st[0].shape[0] != p.shape[0]:
                        st = (torch.zeros(p.shape[0], device=p.device, dtype=torch.float32),
                              torch.zeros(p.shape[1], device=p.device, dtype=torch.float32))
                    row, col = st
                    row.mul_(b2).add_(gf.pow(2).mean(dim=1), alpha=1 - b2)
                    col.mul_(b2).add_(gf.pow(2).mean(dim=0), alpha=1 - b2)
                    self.opt_state[n] = (row, col)
                    row_hat = row / bc
                    col_hat = col / bc
                    rm = row_hat.mean().clamp_min(eps)
                    srow = (rm / row_hat.clamp_min(eps)).sqrt()
                    scol = col_hat.clamp_min(eps).rsqrt()
                    gf.mul_(scol.unsqueeze(0)).mul_(srow.unsqueeze(1))
                else:
                    v = self.opt_state.get(n)
                    if v is None or v.ndim != 0:
                        v = torch.zeros((), device=p.device, dtype=torch.float32)
                    v.mul_(b2).add_(gf.pow(2).mean(), alpha=1 - b2)
                    self.opt_state[n] = v
                    gf.mul_(((v / bc).clamp_min(eps)).rsqrt())
                if self.wd:
                    p.mul_(1.0 - lr * self.wd)
                p.add_(gf.to(p.dtype), alpha=-lr)
                p.grad = None

    # ---------------------------------------------------------- fp32 master
    def _master_step(self, grads, lr, t):
        """Apply an update to fp32 CPU masters (SGD or Adafactor, all-CPU math).

        grads: list of (state_key, cpu bf16/fp32 grad or None). Weights are
        refreshed from the masters separately via _refresh().
        """
        b2 = self.adaf_beta2
        eps = self.adaf_eps
        bc = 1.0 - b2 ** t
        with torch.no_grad():
            for n, g in grads:
                if g is None:
                    continue
                m = self.master_state[n]
                gf = g.float()
                if self.opt == "adafactor":
                    if m.ndim >= 2:
                        st = self.opt_state_cpu.get(n)
                        if st is None:
                            st = (torch.zeros(m.shape[0], dtype=torch.float32),
                                  torch.zeros(m.shape[1], dtype=torch.float32))
                        row, col = st
                        row.mul_(b2).add_(gf.pow(2).mean(dim=1), alpha=1 - b2)
                        col.mul_(b2).add_(gf.pow(2).mean(dim=0), alpha=1 - b2)
                        self.opt_state_cpu[n] = (row, col)
                        row_hat = row / bc
                        col_hat = col / bc
                        rm = row_hat.mean().clamp_min(eps)
                        gf.mul_((rm / row_hat.clamp_min(eps)).sqrt().unsqueeze(1))
                        gf.mul_(col_hat.clamp_min(eps).rsqrt().unsqueeze(0))
                    else:
                        v = self.opt_state_cpu.get(n)
                        if v is None:
                            v = torch.zeros((), dtype=torch.float32)
                        v.mul_(b2).add_(gf.pow(2).mean(), alpha=1 - b2)
                        self.opt_state_cpu[n] = v
                        gf.mul_(((v / bc).clamp_min(eps)).rsqrt())
                if self.wd:
                    m.mul_(1.0 - lr * self.wd)
                m.add_(gf, alpha=-lr)

    def _refresh(self, param, key):
        param.data.copy_(self.master_state[key])

    # --------------------------------------------------------------------- step
    def step(self, xs, ys, update: bool = True):
        torch.cuda.synchronize()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        dev = self.device
        xs = xs.to(dev)
        ys = ys.to(dev)

        with torch.no_grad():
            h_blocks, cache, pos = self._forward_cached(xs)
        t = time.perf_counter()
        h_b = h_blocks.detach().requires_grad_(True)
        z_g = self.norm(h_b)
        z = z_g.detach().requires_grad_(True)
        _, loss = chunked_lm_loss_and_backward(self.lm_head, z, ys, self.loss_chunk)
        z_g.backward(z.grad)
        grad = h_b.grad
        if self.profile:
            torch.cuda.synchronize()
            self.prof["loss_s"] = self.prof.get("loss_s", 0.0) + (time.perf_counter() - t)

        lr = self._lr_now()
        t_step = self.steps_done + 1

        # reverse block pass: recompute, backward, update this block in place
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            self._wait_prefetch()
            self._load(layer)
            if self.prefetch and i - 1 >= 0:
                self._prefetch(self.layers[i - 1])
            h_in = cache[i]
            h_in.requires_grad_(True)
            t = time.perf_counter()
            out = _layer_out(layer(h_in, position_embeddings=pos))
            out.backward(grad)
            if self.profile:
                torch.cuda.synchronize()
                self.prof["bwd_s"] = self.prof.get("bwd_s", 0.0) + (time.perf_counter() - t)
            grad = h_in.grad
            h_in.grad = None
            h_in.requires_grad_(False)
            if update:
                named = [(f"L{i}.{n}", p) for n, p in layer.named_parameters()]
                if self.use_master:
                    if self.clip and self.clip > 0:
                        ps = [p for _, p in named if p.grad is not None]
                        if ps:
                            torch.nn.utils.clip_grad_norm_(ps, self.clip)
                    grads = []
                    tg = time.perf_counter()
                    for n, p in named:
                        g = p.grad
                        grads.append((n, None if g is None else g.detach().to("cpu")))
                        p.grad = None
                    if self.profile:
                        self.prof["grad_s"] = self.prof.get("grad_s", 0.0) + (time.perf_counter() - tg)
                    self._to(layer, "cpu")
                    t = time.perf_counter()
                    self._master_step(grads, lr, t_step)
                    for n, p in named:
                        self._refresh(p, n)
                    if self.profile:
                        torch.cuda.synchronize()
                        self.prof["upd_s"] = self.prof.get("upd_s", 0.0) + (time.perf_counter() - t)
                else:
                    t = time.perf_counter()
                    self._update(named)
                    if self.profile:
                        torch.cuda.synchronize()
                        self.prof["upd_s"] = self.prof.get("upd_s", 0.0) + (time.perf_counter() - t)
                    self._unload(layer, writeback=True)
            else:
                self._unload(layer, writeback=False)
            if self.verbose and (len(self.layers) - i) % 8 == 0:
                print(f"    [bwd] {len(self.layers) - i}/{len(self.layers)} "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

        # input-embedding gradient (forward was no_grad): recompute lookup
        if grad is not None:
            h0 = F.embedding(xs, self.embed.weight)
            g0, = torch.autograd.grad(h0, self.embed.weight, grad_outputs=grad)
            if self.embed.weight.grad is None:
                self.embed.weight.grad = g0
            else:
                self.embed.weight.grad.add_(g0)
            del h0, g0

        # final updates (embedding/head, norm)
        if update:
            if self.use_master:
                def _grab(p):
                    g = p.grad
                    p.grad = None
                    return None if g is None else g.detach().to("cpu")

                for key, p in (("embed.weight", self.embed.weight),
                               ("norm.weight", self.norm.weight)):
                    self._master_step([(key, _grab(p))], lr, t_step)
                    self._refresh(p, key)
                if not self.tied:
                    self._master_step([("head.weight", _grab(self.lm_head.weight))],
                                      lr, t_step)
                    self._refresh(self.lm_head.weight, "head.weight")
            else:
                self._update([("embed.weight", self.embed.weight)])
                if not self.tied:
                    self._update([("head.weight", self.lm_head.weight)])
                self._update([("norm.weight", self.norm.weight)])

        # free cached activations
        cache.clear()
        del h_blocks, h_b, z_g, z, grad
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() / 1e9
        if update:
            self.steps_done += 1
        return float(loss), time.time() - t0, peak
