"""EXP61: train the router where content actually matters (needle retrieval).

Why: on WikiText the dense attention mass is essentially sink+recent, so a distilled
router correctly learns a position prior and never looks far (LCGA_FINDINGS.md).
On a needle/retrieval context the dense model *does* attend to the planted fact
(exp60: dense CE 0.16 vs window 3.08 at 8k), so the same distillation signal now
points at content -- but only at the handful of answer positions. Averaging the loss
over all tokens dilutes that signal ~1/256, which is why the plain recipe cannot
learn it.

This script trains the v1 block-mass objective (KL(teacher block mass || router
probs)) but **only at the query rows that predict the answer tokens**. The
hidden-state MSE term (v1) is optional: the exact sparse output is computed under
no_grad, so it only carries router gradient through the coarse branch; the selection
itself is trained purely by the KL term.

Diagnostics printed every log interval are the actual learning signal:
  t_mass = teacher (dense) block mass on the fact block at answer rows
  p_fact = router probability on the fact block at answer rows
  rec@1  = fraction of answer rows whose top-1 block is the fact block
  rec@m  = fraction whose top-m selection contains the fact block

Usage:
  python experiments/exp61_train_router_needle.py --steps 800 --seq 1024 --tag needle_r
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.attention import repeat_kv
from lcsa.data import load_tokens
from lcsa.harness import load_model
from lcsa.metrics import append_jsonl, now
from lcsa.model import (begin_capture, end_capture, patch_model, set_capture_qk,
                        set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"
QUESTION = " Question: What is the secret code? Answer:"
# Paraphrase augmentation: matched (fact sentence, question) pairs. The exp62
# "unseen" probe uses a fourth phrasing that is NOT in this list, so it stays a
# held-out test when --paraphrase is on.
PARAPHRASES = [
    (" The secret code is {c}.", QUESTION),
    (" The backup key is {c}.", " Question: What is the backup key? Answer:"),
    (" The old password is {c}.", " Question: What was the old password? Answer:"),
]


class NeedleSampler:
    """Builds fixed-length needle examples: filler + fact + question + answer.

    Returns ids (len seq), the answer start index, and the block holding the fact.
    Depth of the fact is uniform over the filler, as in exp60.
    """

    def __init__(self, tok, filler, seq, block=64, seed=0, paraphrase=False,
                 n_facts=1):
        self.tok, self.filler, self.seq, self.block = tok, filler, seq, block
        self.rng = random.Random(seed)
        self.paraphrase = paraphrase
        self.n_facts = max(1, int(n_facts))
        # all answers must have the same token length so one set of answer rows
        # applies to the whole batch
        self.ans_len = len(tok(" 1234", add_special_tokens=False).input_ids)

    def _code(self):
        for _ in range(50):
            code = str(self.rng.randrange(1000, 9999))
            ans = self.tok(f" {code}", add_special_tokens=False).input_ids
            if len(ans) == self.ans_len:
                return code, ans
        raise RuntimeError("could not draw a code with the reference answer length")

    def one(self):
        code, ans = self._code()
        # n_facts facts; the LAST one is the queried one. With more than one fact the
        # phrasings must differ, otherwise the question cannot disambiguate them.
        pairs = list(PARAPHRASES)
        if self.paraphrase or self.n_facts > 1:
            self.rng.shuffle(pairs)
        pairs = pairs[: self.n_facts]
        if self.n_facts == 1 and not self.paraphrase:
            pairs = [PARAPHRASES[0]]
        facts = []
        for k, (tmpl, _q) in enumerate(pairs):
            c = code if k == len(pairs) - 1 else self._code()[0]
            facts.append(self.tok(tmpl.format(c=c), add_special_tokens=False).input_ids)
        q = self.tok(pairs[-1][1], add_special_tokens=False).input_ids

        body_total = self.seq - len(q) - len(ans)
        total_fact = sum(len(f) for f in facts)
        if len(self.filler) < body_total + 1:
            raise RuntimeError("filler too short")
        span = max(1, body_total - total_fact - 16)
        start = self.rng.randrange(0, max(1, len(self.filler) - body_total - 1))
        slot = max(1, span // len(facts))
        pos = [self.rng.randrange(i * slot, (i + 1) * slot) for i in range(len(facts))]
        out, cur, fact_block = [], 0, 0
        for i, (f, p) in enumerate(zip(facts, pos)):
            out.extend(self.filler[start + cur:start + p])
            out.extend(f)
            cur = p
            if i == len(facts) - 1:
                fact_block = min((len(out) - 2) // self.block,
                                 (body_total + len(q) - 1) // self.block)
        out.extend(self.filler[start + cur:start + body_total])
        body = out[:body_total]
        ids = body + q + ans
        assert len(ids) == self.seq, (len(ids), self.seq)
        ans_start = len(ids) - len(ans)
        return ids, ans_start, fact_block

    def batch(self, n):
        xs, starts, facts = [], [], []
        for _ in range(n):
            ids, a, f = self.one()
            assert len(ids) == self.seq, (len(ids), self.seq)
            xs.append(ids)
            starts.append(a)
            facts.append(f)
        return (torch.tensor(xs, dtype=torch.long),
                torch.tensor(starts, dtype=torch.long),
                torch.tensor(facts, dtype=torch.long))


def save_ckpt(model, path, meta):
    torch.save({"state": [dict(layer.self_attn.named_parameters()) for
                          layer in model.model.layers], "meta": meta}, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--lam_kl", type=float, default=1.0)
    ap.add_argument("--lam_mse", type=float, default=0.0)
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--n_concepts", type=int, default=64)
    ap.add_argument("--graph_rank", type=int, default=0)
    ap.add_argument("--coarse", type=int, default=0,
                    help="1 = enable the coarse block-value branch (v1 default); "
                         "needed only if lam_mse > 0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=40)
    ap.add_argument("--loss", default="residual",
                    choices=("kl", "residual", "retrieval", "top1content"),
                    help="kl = v1 KL on the whole teacher distribution; residual = KL "
                         "only over blocks outside {sink, self, recent window} with layers "
                         "whose teacher actually has content mass; top1content = "
                         "cross-entropy on the teacher's argmax content block (no fact "
                         "label: the target is read off the dense attention); retrieval = "
                         "labelled -log p(fact block) at answer rows (upper bound)")
    ap.add_argument("--min_content", type=float, default=0.03,
                    help="residual mode: skip a layer whose teacher content mass is "
                         "below this")
    ap.add_argument("--content_rank", type=int, default=0,
                    help="bilinear content shortcut width; the concept-graph path is "
                         "content-blind at init and its gradient is ~1/K^2, so a direct "
                         "content path is the fair test of learnability")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--teacher", default="token", choices=("token", "block"),
                    help="token = attention of the answer tokens aggregated to blocks "
                         "(concentrated, the real retrieval signal); block = the v1 "
                         "block mass, diluted 1/block by averaging over the answer block")
    ap.add_argument("--rows", default="answer", choices=("answer", "all"),
                    help="answer = only the query rows that predict the answer tokens "
                         "(needs the token teacher); all = every query row with the v1 "
                         "block-mass teacher (ablation of loss placement)")
    ap.add_argument("--lam_window", type=float, default=0.0,
                    help="extra KL term on the NON-content part of the teacher (sink, self "
                         "and the recent window), so the position prior is not forgotten "
                         "while content selection is learned. 0 = off")
    ap.add_argument("--shuffle_label", type=int, default=0,
                    help="control: 1 = point the retrieval loss at a random block instead "
                         "of the true fact block (same compute, wrong label)")
    ap.add_argument("--paraphrase", type=int, default=0,
                    help="1 = draw the fact sentence + question from three matched "
                         "paraphrases (the exp62 'unseen' phrasing stays held out)")
    ap.add_argument("--n_facts", type=int, default=1,
                    help=">1 plants several facts; the question asks about the last one, "
                         "so the router must select the *queried* block")
    ap.add_argument("--tag", default="needle_r")
    ap.add_argument("--model", default="models/Qwen2.5-0.5B",
                    help="base model directory (relative to the repo root)")
    ap.add_argument("--lm_every", type=int, default=0,
                    help="mixed objective: every (K+1)-th step is a needle step and the "
                         "other K steps are plain-LM steps (full-KL v1 objective on "
                         "WikiText). 0 = needle only (default)")
    ap.add_argument("--q_pool", default="mean", choices=("mean", "last"),
                    help="router query pooling over a block's 64 positions; 'last' uses "
                         "the only position that can attend to the whole block")
    ap.add_argument("--init_ckpt", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model_dir = os.path.join(ROOT, args.model) if not os.path.isabs(args.model) else args.model
    torch.manual_seed(args.seed)
    model = load_model(model_dir, device=DEV)
    tok = AutoTokenizer.from_pretrained(model_dir)
    wrappers = patch_model(model, budget=args.budget, block=args.block,
                           n_concepts=args.n_concepts, graph_rank=args.graph_rank,
                           dynamic=False, content_rank=args.content_rank, rel_bias=True,
                           capture_dense=True, use_coarse=bool(args.coarse),
                           null_init=-1e9, q_pool=args.q_pool)
    if args.init_ckpt:
        ck = torch.load(os.path.join(CKPT_DIR, args.init_ckpt), map_location="cpu",
                        weights_only=False)
        for i, layer in enumerate(model.model.layers):
            own = dict(layer.self_attn.named_parameters())
            for n, t in ck["state"][i].items():
                if n in own:
                    own[n].data.copy_(t.to(own[n].device))
        print(f"[init] from {args.init_ckpt}", flush=True)

    params = [p for w in wrappers.values() for p in w.router.parameters()
              if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    print(f"[model] patched {len(wrappers)} layers, router params: {n_params/1e6:.3f}M, "
          f"coarse={bool(args.coarse)}", flush=True)

    filler = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))[:60000].tolist()
    sampler = NeedleSampler(tok, filler, args.seq, block=args.block, seed=args.seed,
                            paraphrase=bool(args.paraphrase),
                            n_facts=args.n_facts)
    lm_sampler = None
    if args.lm_every > 0:
        from lcsa.data import TrainSampler
        tokens = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))
        lm_sampler = TrainSampler(tokens, args.seq)
        print(f"[mix] one needle step per {args.lm_every + 1} steps; the rest are "
              f"plain-LM full-KL steps", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    model.train()
    run = {"kl": 0.0, "mse": 0.0, "t_mass": 0.0, "p_fact": 0.0,
           "rec1": 0.0, "recm": 0.0, "used": 0.0, "n": 0, "t": time.time()}
    t0 = time.time()
    n_blocks = args.seq // args.block
    m_sel = max(1, int(round(args.budget * n_blocks)))

    for step in range(1, args.steps + 1):
        use_lm = lm_sampler is not None and (step % (args.lm_every + 1) != 0)
        if use_lm:
            x, _y = lm_sampler.batch(args.batch)
            x = x.to(DEV)
            starts = facts = None
        else:
            x, starts, facts = sampler.batch(args.batch)
            x = x.to(DEV)
            starts = starts.to(DEV)
            facts = facts.to(DEV)
            if args.shuffle_label:
                # control: the loss points at a random block -> if the eval still
                # improves, the gain was not retrieval (label) but something else
                facts = torch.randint(1, (args.seq // args.block) - 1, (facts.shape[0],),
                                      device=DEV)
        set_sparse_enabled(True)
        set_capture_qk(args.teacher == "token" and not use_lm)
        begin_capture()
        model.model(input_ids=x, use_cache=False)
        captured = end_capture()

        loss = torch.zeros((), device=DEV)
        kl_acc = mse_acc = 0.0
        t_mass_acc = p_fact_acc = rec1_acc = recm_acc = 0.0
        nb = n_used = 0
        keys_all = torch.arange(args.seq, device=DEV)
        rows_all = use_lm or args.rows == "all"
        for i, cap in enumerate(captured):
            if "mass" not in cap:
                continue
            B, T, _d = cap["student"].shape
            ar = list(range(T - sampler.ans_len, T))
            if rows_all:
                # loss on every query row, using the v1 block-mass teacher: isolates
                # whether restricting the loss to the answer rows is what matters
                # (also the plain-LM half of a mixed run)
                all_rows = list(range(T // args.block))
                teacher = cap["mass"][:, :, all_rows, :]
                probs_r = cap["probs"][:, all_rows, :]
                rows = all_rows
            else:
                # every answer token is scored; group them by their query block
                rows = sorted({t // args.block for t in ar})
                if args.teacher == "token":
                    q = cap["q"]
                    k = cap["k"]
                    H = q.shape[1]
                    if k.shape[1] != H:
                        k = repeat_kv(k, H // k.shape[1])
                    qs = q[:, :, ar, :].float()
                    logits = torch.einsum("bhrd,bhtd->bhrt", qs, k.float()) * wrappers[i].scale
                    mask = keys_all.view(1, 1, 1, T) > torch.tensor(ar, device=DEV).view(1, 1, -1, 1)
                    att = torch.softmax(logits.masked_fill(mask, float("-inf")), dim=-1)
                    att_b = att.view(B, H, len(ar), args.seq // args.block, args.block).sum(-1)
                    teacher = torch.stack([att_b[:, :, [j for j, t in enumerate(ar)
                                                        if t // args.block == r], :].sum(2)
                                           for r in rows], dim=2)      # (B,H,R,NB)
                else:
                    teacher = cap["mass"][:, :, rows, :]
                probs_r = cap["probs"][:, rows, :]                     # (B,R,NB)

            # blocks that the mandatory slots / recency prior already cover:
            # 0 (sink), the query block itself, and the (m_sel-1) most recent blocks
            rr = torch.tensor(rows, device=DEV).view(1, -1, 1)
            jj = torch.arange(args.seq // args.block, device=DEV).view(1, 1, -1)
            noncontent = (jj == 0) | (jj == rr) | (jj > rr - m_sel)
            cmask = (~noncontent).float()                          # (1,R,NB)

            if use_lm or args.loss == "kl":
                kl = wrappers[i].router.kl_mass_loss(teacher, probs_r)
            elif args.loss == "residual":
                with torch.no_grad():
                    tn = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-9)
                    content_mass = (tn * cmask).sum(-1).mean().item()
                if content_mass < args.min_content:
                    nb += 1
                    continue
                t_c = teacher * cmask
                p_c = probs_r * cmask
                p_c = p_c / p_c.sum(-1, keepdim=True).clamp_min(1e-9)
                kl = wrappers[i].router.kl_mass_loss(t_c, p_c)
            elif args.loss == "top1content":
                # label-free: distil the teacher's *content* argmax (the block dense
                # attention picks outside the budget the prior already covers)
                with torch.no_grad():
                    tm = teacher.mean(1)                              # (B,R,NB)
                    tm = tm / tm.sum(-1, keepdim=True).clamp_min(1e-9)
                    tgt_block = tm.masked_fill(cmask == 0, -1.0).argmax(-1)   # (B,R)
                    content_mass = (tm * cmask).sum(-1).mean().item()
                if content_mass < args.min_content:
                    nb += 1
                    continue
                p_t = probs_r.gather(-1, tgt_block.unsqueeze(-1)).squeeze(-1)
                kl = -torch.log(p_t.clamp_min(1e-9)).mean()
            else:  # retrieval: labelled upper bound, needs no teacher
                p_f = probs_r.gather(-1, facts.view(-1, 1, 1).expand(-1, len(rows), 1)
                                     ).squeeze(-1)
                kl = -torch.log(p_f.clamp_min(1e-9)).mean()
            if args.lam_window > 0 and not use_lm:
                w = 1.0 - cmask                                   # (1,R,NB)
                t_w = teacher * w
                p_w = probs_r * w
                p_w = p_w / p_w.sum(-1, keepdim=True).clamp_min(1e-9)
                kl = kl + args.lam_window * wrappers[i].router.kl_mass_loss(t_w, p_w)
            loss = loss + args.lam_kl * kl
            kl_acc += kl.item()
            if args.lam_mse > 0 and starts is not None:
                tok_rows = torch.cat([torch.arange(max(0, int(a) - 1), min(T, int(a) + 4),
                                                   device=DEV) for a in starts.tolist()])
                tok_rows = tok_rows.unique()
                mse = F.mse_loss(cap["student"][:, tok_rows], cap["dense"][:, tok_rows])
                loss = loss + args.lam_mse * mse
                mse_acc += mse.item()
            if facts is not None:
                with torch.no_grad():
                    tm = teacher.sum(2)                            # (B,H,NB)
                    tm = tm / tm.sum(-1, keepdim=True).clamp_min(1e-9)
                    pr = probs_r                                 # (B,R,NB)
                    tf = tm.gather(-1, facts.view(-1, 1, 1).expand(-1, tm.shape[1], 1)).squeeze(-1)
                    pf = pr.gather(-1, facts.view(-1, 1, 1).expand(-1, len(rows), 1)).squeeze(-1)
                    top1 = pr.argmax(-1) == facts.view(-1, 1)
                    topm = pr.topk(min(m_sel, pr.shape[-1]), dim=-1).indices == \
                        facts.view(-1, 1, 1)
                    t_mass_acc += tf.mean().item()
                    p_fact_acc += pf.mean().item()
                    rec1_acc += top1.float().mean().item()
                    recm_acc += topm.any(-1).float().mean().item()
            nb += 1
            n_used += 1
        if n_used == 0:
            raise RuntimeError(f"no layer contributed (loss={args.loss}, "
                               f"min_content={args.min_content})")
        loss = loss / n_used
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        del captured

        run["kl"] += kl_acc / nb
        run["mse"] += mse_acc / nb
        run["t_mass"] += t_mass_acc / nb
        run["p_fact"] += p_fact_acc / nb
        run["rec1"] += rec1_acc / nb
        run["recm"] += recm_acc / nb
        run["used"] += n_used
        run["n"] += 1
        if step % args.log_every == 0:
            dt = time.time() - run["t"]
            n = run["n"]
            msg = (f"[step {step}/{args.steps}] loss={loss.item():.4f} "
                   f"kl={run['kl']/n:.4f} t_mass={run['t_mass']/n:.3f} "
                   f"p_fact={run['p_fact']/n:.3f} rec@1={run['rec1']/n:.3f} "
                   f"rec@{m_sel}={run['recm']/n:.3f} used={run['used']/n:.1f} "
                   f"gnorm={float(gnorm):.2f} "
                   f"lr={sched.get_last_lr()[0]:.2e} {dt/n*1000:.0f}ms/step "
                   f"vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            print(msg, flush=True)
            append_jsonl(RESULTS, {"exp": "exp61_needle_router", "tag": args.tag,
                                   "step": step, "loss": loss.item(),
                                   "kl": run["kl"]/n, "t_mass": run["t_mass"]/n,
                                   "p_fact": run["p_fact"]/n, "rec1": run["rec1"]/n,
                                   "recm": run["recm"]/n, "gnorm": float(gnorm),
                                   "lr": sched.get_last_lr()[0], "ts": now()})
            run = {k: 0.0 for k in run if k != "t"}
            run["t"] = time.time()

    out = os.path.join(CKPT_DIR, f"{args.tag}_final.pt")
    save_ckpt(model, out, {"args": vars(args), "router_params": n_params,
                           "steps": args.steps})
    print(f"[done] {time.time()-t0:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
