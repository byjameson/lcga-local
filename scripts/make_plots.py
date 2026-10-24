"""Make PNG plots from results/ JSON files (PPL/C speed/quality curves)."""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")


def load_jsonl():
    rows = []
    p = os.path.join(RES, "results.jsonl")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    os.makedirs(PLOTS, exist_ok=True)
    rows = load_jsonl()

    # ---- 1. speed vs T (dense vs sparse from the SAME (latest clean) run) -----
    exp4rows = [r for r in rows if r.get("exp") == "exp4_speed"
                and any(isinstance(v, dict) and "dense_ms" in v
                        for v in r.get("prefill", {}).values())]
    if exp4rows:
        row = exp4rows[-1]
        Ts, dense, sp = [], [], []
        for k, v in row.get("prefill", {}).items():
            if isinstance(v, dict) and "dense_ms" in v and "sparse_ms" in v:
                Ts.append(int(k[1:]))
                dense.append(v["dense_ms"])
                sp.append(v["sparse_ms"])
        order = sorted(range(len(Ts)), key=lambda i: Ts[i])
        Ts = [Ts[i] for i in order]
        dense = [dense[i] for i in order]
        sp = [sp[i] for i in order]
        plt.figure(figsize=(6, 4.2))
        plt.plot(Ts, dense, "o-", label="dense (fused SDPA, B=1)")
        plt.plot(Ts, sp, "s-", label="LCGA sparse (B=1, rho=0.25)")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.xlabel("sequence length T")
        plt.ylabel("prefill forward ms (logits_to_keep=1)")
        plt.title(f"Qwen2.5-0.5B prefill speed (RTX 3050 6GB) [{row.get('tag', '?')}]")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS, "prefill_speed.png"), dpi=130)
        plt.close()
        print("wrote plots/prefill_speed.png")

    # ---- 2. budget sweep (exp3) ----------------------------------------------
    for f in glob.glob(os.path.join(RES, "exp3_*.json")):
        d = json.load(open(f, encoding="utf-8"))
        sw = d.get("budget_sweep") or {}
        rhos, deltas, recs = [], [], []
        for k, v in sw.items():
            if isinstance(v, dict):
                rhos.append(float(k.split("_")[1]))
                deltas.append(v["delta"])
                recs.append(v.get("recall", float("nan")))
        if not rhos:
            continue
        order = sorted(range(len(rhos)), key=lambda i: rhos[i])
        rhos = [rhos[i] for i in order]
        deltas = [deltas[i] for i in order]
        recs = [recs[i] for i in order]
        tag = d.get("tag", "eval")
        fig, ax1 = plt.subplots(figsize=(6, 4.2))
        ax1.plot(rhos, deltas, "o-", color="tab:red")
        ax1.set_xlabel("budget rho (fraction of blocks)")
        ax1.set_ylabel("sparse PPL - dense PPL", color="tab:red")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(rhos, recs, "s--", color="tab:blue")
        ax2.set_ylabel("attention mass recall", color="tab:blue")
        plt.title(f"LCGA quality vs budget ({tag})")
        fig.tight_layout()
        out = os.path.join(PLOTS, f"budget_sweep_{tag}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("wrote", out)

    # ---- 3. training curves (exp2) -------------------------------------------
    ev = [r for r in rows if r.get("exp") == "exp2_eval"]
    tr = [r for r in rows if r.get("exp") == "exp2_train"]
    if ev:
        by_tag = {}
        for r in ev:
            by_tag.setdefault(r.get("tag", "v1"), []).append(r)
        for tag, rr in by_tag.items():
            rr.sort(key=lambda r: r["step"])
            steps = [r["step"] for r in rr]
            dense = [r["dense_ppl"] for r in rr]
            sparse = [r["sparse_ppl"] for r in rr]
            gates = [sum(r["gate"]) / len(r["gate"]) for r in rr]
            plt.figure(figsize=(6.4, 4.2))
            plt.plot(steps, dense, "o-", label="dense PPL")
            plt.plot(steps, sparse, "s-", label="sparse PPL")
            plt.plot(steps, gates, "^--", label="mean sigmoid(gate)")
            plt.xlabel("training step")
            plt.ylabel("PPL (32 dev chunks)")
            plt.title(f"LCGA router training [{tag}]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out = os.path.join(PLOTS, f"train_curve_{tag}.png")
            plt.savefig(out, dpi=130)
            plt.close()
            print("wrote", out)
        if tr:
            by_tag2 = {}
            for r in tr:
                by_tag2.setdefault(r.get("tag", "v1"), []).append(r)
            for tag, rr in by_tag2.items():
                rr.sort(key=lambda r: r["step"])
                steps = [r["step"] for r in rr]
                plt.figure(figsize=(6.4, 4.2))
                plt.plot(steps, [r["mse"] for r in rr], "o-", label="MSE")
                plt.plot(steps, [r["kl"] for r in rr], "s-", label="KL")
                plt.xlabel("training step")
                plt.title(f"training losses [{tag}]")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                out = os.path.join(PLOTS, f"train_losses_{tag}.png")
                plt.savefig(out, dpi=130)
                plt.close()
                print("wrote", out)

    # ---- 4. stage-2 adaptation curves (exp7_eval) -----------------------------
    ev7 = [r for r in rows if r.get("exp") == "exp7_eval"]
    if ev7:
        by_tag = {}
        for r in ev7:
            by_tag.setdefault(r.get("tag", "v2"), []).append(r)
        for tag, rr in by_tag.items():
            if all(r.get("step") is not None for r in rr):
                rr.sort(key=lambda r: r["step"])
                steps = [r["step"] for r in rr]
            else:
                steps = list(range(len(rr)))   # rows lacking step: use insertion order
            dense = [r.get("dense_ppl") for r in rr]
            sparse = [r.get("sparse_ppl") for r in rr]
            rec = [r.get("recall_mean") for r in rr]
            plt.figure(figsize=(6.4, 4.2))
            plt.plot(steps, dense, "o-", label="dense PPL (sparse off)")
            plt.plot(steps, sparse, "s-", label="sparse PPL (clean)")
            if any(v is not None for v in rec):
                plt.plot(steps, rec, "^--", label="recall (mean)")
            plt.xlabel("stage-2 step")
            plt.ylabel("PPL / recall (16 dev chunks)")
            plt.title(f"LCGA stage-2 adaptation [{tag}]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out = os.path.join(PLOTS, f"stage2_curve_{tag}.png")
            plt.savefig(out, dpi=130)
            plt.close()
            print("wrote", out)

    # ---- 5. stage-2 losses (exp7_train) ---------------------------------------
    tr7 = [r for r in rows if r.get("exp") == "exp7_train"]
    if tr7:
        by_tag = {}
        for r in tr7:
            by_tag.setdefault(r.get("tag", "v2"), []).append(r)
        for tag, rr in by_tag.items():
            rr.sort(key=lambda r: r["step"])
            steps = [r["step"] for r in rr]
            plt.figure(figsize=(6.4, 4.2))
            plt.plot(steps, [r["ce"] for r in rr], "o-", label="CE")
            plt.plot(steps, [r["kl"] for r in rr], "s-", label="KL")
            plt.plot(steps, [100 * r["mse"] for r in rr], "^--", label="MSE x100")
            plt.xlabel("stage-2 step")
            plt.title(f"stage-2 losses [{tag}]")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            out = os.path.join(PLOTS, f"stage2_losses_{tag}.png")
            plt.savefig(out, dpi=130)
            plt.close()
            print("wrote", out)


    # ---- 6. P2 byte-level valid bpb curves (exp10) -----------------------------
    p2rows = [r for r in rows if r.get("exp") == "exp10_byte" and r.get("tag") == "p2"
              and r.get("evals")]
    if p2rows:
        plt.figure(figsize=(6.4, 4.2))
        for r in p2rows:
            steps = [e["step"] for e in r["evals"]]
            bpbv = [e["valid_bpb"] for e in r["evals"]]
            label = f"{r['model']} (test {r.get('test_bpb')})"
            plt.plot(steps, bpbv, "o-", label=label)
        plt.xlabel("step")
        plt.ylabel("valid bpb (bits/byte)")
        plt.title("P2 byte-level training [p2]")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(PLOTS, "p2_valid_bpb.png")
        plt.savefig(out, dpi=130)
        plt.close()
        print("wrote", out)


if __name__ == "__main__":
    sys.exit(main())
