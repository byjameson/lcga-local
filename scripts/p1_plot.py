"""P1 (exp8) summary figure: peak VRAM and step time, naive vs stream.

Reads results/exp8_*.json (skips smokes), renders plots/p1_streaming.png:
  left  panel: peak reserved VRAM per step (naive vs stream), 6GB line, OOM marks
  right panel: seconds per step (avg)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")


def main():
    os.makedirs(PLOTS, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(RESULTS, "exp8_*.json"))):
        with open(f, encoding="utf-8") as fh:
            r = json.load(fh)
        if "smoke" in r.get("tag", ""):
            continue
        rows.append(r)
    if not rows:
        print("no exp8 results found")
        return
    rows.sort(key=lambda r: (r.get("model", ""), r.get("cap_fraction") or 0, r.get("tag", "")))

    labels = []
    for r in rows:
        m = r["model"].split("/")[-1].replace("Qwen2.5-", "")
        cap = r.get("cap_fraction")
        labels.append(m + (f"\ncap {cap*100:.0f}%" if cap else ""))

    def peak_of(r, mode):
        e = r.get(mode) or {}
        pk = e.get("peak_gb")
        if isinstance(pk, list):
            return max(pk) if pk else 0.0
        return pk or 0.0

    def step_of(r, mode):
        e = r.get(mode) or {}
        ts = e.get("times_s")
        if isinstance(ts, list) and ts:
            return sum(ts) / len(ts)
        return e.get("s_per_step") or 0.0

    n_peak = [peak_of(r, "naive") for r in rows]
    s_peak = [peak_of(r, "stream") for r in rows]
    n_t = [step_of(r, "naive") for r in rows]
    s_t = [step_of(r, "stream") for r in rows]

    x = list(range(len(rows)))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.bar([i - w / 2 for i in x], n_peak, w, label="naive (full GPU graph)")
    ax.bar([i + w / 2 for i in x], s_peak, w, label="stream (block offload, CPU base)")
    for i, r in enumerate(rows):
        if (r.get("naive") or {}).get("oom"):
            ax.annotate("OOM", (i - w / 2, 0), xytext=(0, 6), textcoords="offset points",
                        ha="center", color="tab:red", fontsize=8)
    ax.axhline(6.0, color="gray", ls="--", lw=1)
    ax.text(len(rows) - 0.55, 6.05, "6GB VRAM", ha="right", color="gray", fontsize=8)
    ax.set_ylabel("peak reserved (GB)")
    ax.set_title("Peak VRAM per training step (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar([i - w / 2 for i in x], n_t, w, label="naive")
    ax.bar([i + w / 2 for i in x], s_t, w, label="stream")
    for i, r in enumerate(rows):
        if (r.get("naive") or {}).get("oom"):
            ax.annotate("OOM", (i - w / 2, 0), xytext=(0, 6), textcoords="offset points",
                        ha="center", color="tab:red", fontsize=8)
    ax.set_ylabel("s / step (avg)")
    ax.set_title("Step time (T=1024, LoRA r=16)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("P1: block-streaming LoRA training on 1x RTX 3050 6GB", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(PLOTS, "p1_streaming.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
