"""exp36: learn a quality router from measured per-prompt outcomes.

Training data: exp30_60 (60 prompts, exp26 set).
Independent test: exp34 holdout (35 prompts, not used for fitting).

The label is "3B is uniquely better on this prompt" (1) versus "chat is equal
or better" (0). That is deliberately cost-sensitive: ties go to 1.5B because it
is ~2.7x faster. Features are hashed word/char n-grams plus a few transparent
task indicators. A tiny L2 logistic regression is trained from scratch with
NumPy; no sklearn/scipy dependency.

Hyperparameters and the probability threshold are chosen by deterministic
5-fold cross-validation on the 60 training prompts only. The 35-prompt holdout
is touched once for the final report. If the learned policy does not beat the
existing rule router, it is a negative result and is not integrated.

Usage:
  python experiments/exp36_learned_router.py
  python experiments/exp36_learned_router.py --train results/exp30_60_chat3b_quality.json \
      --test results/exp34_holdout_chat3b_quality.json --tag exp36
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp31_router_eval import quality  # noqa: E402

DIM = 768
TOKEN_RE = re.compile(r"[a-z0-9çğıöşüâîû]+")
MATH_RE = re.compile(r"\d\s*[\+\-\*/^]\s*\d")

LIST_CUES = ("listele", "list three", "list ", "maddede", "liste")
SOCIAL_CUES = ("tesekkur", "teşekkür", "thank")
FACT_CUES = ("baskenti", "başkenti", "capital", "en buyuk", "en büyük",
             "largest", "sembol", "symbol", "kac", "kaç", "neresi",
             "how many", "what is", "which")
INSTRUCTION_CUES = ("sadece", "just", "reply with", "answer with", "write",
                    "yaz", "translate", "cevir", "çevir", "count down",
                    "sentence", "cumle", "cümle", "kelime", "word")
TR_WORDS = (" bir ", " ve ", " kac ", " kaç ", " neresi", " nedir", " nasil",
            " nasıl ", " mi ", " mu ", " icin ", " için ")


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def has_any(q: str, cues: tuple[str, ...]) -> bool:
    return any(c in q for c in cues)


def features(prompt: str) -> np.ndarray:
    """Hashed sparse features plus 12 normalized handcrafted indicators."""
    q = normalize(prompt)
    toks = TOKEN_RE.findall(q)
    x = np.zeros(DIM + 12, dtype=np.float32)

    def add(key: str) -> None:
        h = zlib.crc32(key.encode("utf-8"))
        idx = h % DIM
        sign = 1.0 if (h >> 15) & 1 else -1.0
        x[idx] += sign

    for t in toks:
        add("w:" + t)
        if 2 < len(t) <= 12:
            for n in (3, 4, 5):
                for i in range(len(t) - n + 1):
                    add(f"c{n}:" + t[i:i + n])
    for a, b in zip(toks, toks[1:]):
        add("b:" + a + "_" + b)

    tr = (any(ch in q for ch in "çğıöşü") or
          any(w in f" {q} " for w in TR_WORDS))
    x[DIM + 0] = 1.0 if tr else 0.0
    x[DIM + 1] = 1.0 if has_any(q, LIST_CUES) else 0.0
    x[DIM + 2] = 1.0 if has_any(q, SOCIAL_CUES) else 0.0
    x[DIM + 3] = 1.0 if has_any(q, FACT_CUES) else 0.0
    x[DIM + 4] = 1.0 if has_any(q, INSTRUCTION_CUES) else 0.0
    x[DIM + 5] = 1.0 if MATH_RE.search(q) else 0.0
    x[DIM + 6] = 1.0 if "?" in q else 0.0
    x[DIM + 7] = min(len(q) / 100.0, 2.0)
    x[DIM + 8] = min(len(toks) / 15.0, 2.0)
    x[DIM + 9] = min(len(re.findall(r"\d+", q)) / 4.0, 1.0)
    x[DIM + 10] = 1.0 if q.endswith(".") else 0.0
    x[DIM + 11] = 1.0 if any(w in q for w in ("code", "kod", "python",
                                               "debug", "hata")) else 0.0

    # L2 normalize string features and handcrafted features separately.
    a = np.linalg.norm(x[:DIM])
    if a:
        x[:DIM] /= a
    b = np.linalg.norm(x[DIM:])
    if b:
        x[DIM:] /= b
    return x


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    variants = {v["label"]: v["rows"] for v in data["variants"]}
    if "chat" not in variants or "chat3b" not in variants:
        raise SystemExit(f"{path} needs chat and chat3b variants")
    crows, brows = variants["chat"], variants["chat3b"]
    if len(crows) != len(brows):
        raise SystemExit("variant row counts differ")
    rows = []
    for c, b in zip(crows, brows):
        if c["prompt"] != b["prompt"]:
            raise SystemExit("variant prompt order differs")
        qc, qb = quality(c), quality(b)
        rows.append({
            "prompt": c["prompt"], "category": c["category"],
            "chat_gen_s": float(c["gen_s"]), "b_gen_s": float(b["gen_s"]),
            "chat_q": qc, "b_q": qb,
            "label": int(qb > qc),
        })
    return rows


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float,
               pos_weight: float, epochs: int = 1400,
               lr: float = 0.20) -> tuple[np.ndarray, float]:
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    sw = np.where(y > 0, pos_weight, 1.0)
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = (p - y) * sw
        grad = X.T @ err / len(y) + l2 * w
        gb = float(err.mean())
        w -= lr * grad
        b -= lr * gb
    return w, b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(X @ w + b)


def stratified_folds(y: np.ndarray, k: int = 5,
                     seed: int = 42) -> list[np.ndarray]:
    pos = list(np.where(y == 1)[0])
    neg = list(np.where(y == 0)[0])
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(k)]
    for arr in (pos, neg):
        for i, idx in enumerate(arr):
            folds[i % k].append(int(idx))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def choose_use_3b(prompt: str, prob: float, threshold: float,
                  safety: bool) -> bool:
    q = normalize(prompt)
    if safety and (has_any(q, LIST_CUES) or has_any(q, SOCIAL_CUES)):
        return False
    return prob >= threshold


def eval_policy(rows: list[dict], probs: np.ndarray, threshold: float,
                safety: bool) -> dict:
    q = lat = 0.0
    units = {"chat": 0, "chat3b": 0}
    unique_3b = 0
    unique_chat = 0
    wrong = []
    for r, p in zip(rows, probs):
        use_b = choose_use_3b(r["prompt"], float(p), threshold, safety)
        if use_b:
            q += r["b_q"]
            lat += r["b_gen_s"]
            units["chat3b"] += 1
        else:
            q += r["chat_q"]
            lat += r["chat_gen_s"]
            units["chat"] += 1
        if r["b_q"] > r["chat_q"]:
            if use_b:
                unique_3b += 1
            else:
                wrong.append({"prompt": r["prompt"], "miss": "3b_better"})
        elif r["chat_q"] > r["b_q"]:
            if not use_b:
                unique_chat += 1
            else:
                wrong.append({"prompt": r["prompt"], "miss": "chat_better"})
    n = max(1, len(rows))
    return {
        "n": len(rows), "quality": round(q / n, 4),
        "avg_gen_s": round(lat / n, 3), "units": units,
        "unique_3b_kept": unique_3b, "unique_chat_kept": unique_chat,
        "wrong": wrong,
    }


def baseline(rows: list[dict], unit: str) -> dict:
    if unit == "chat":
        q = sum(r["chat_q"] for r in rows) / max(1, len(rows))
        lat = sum(r["chat_gen_s"] for r in rows) / max(1, len(rows))
    else:
        q = sum(r["b_q"] for r in rows) / max(1, len(rows))
        lat = sum(r["b_gen_s"] for r in rows) / max(1, len(rows))
    return {"quality": round(q, 4), "avg_gen_s": round(lat, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="results/exp30_60_chat3b_quality.json")
    ap.add_argument("--test", default="results/exp34_holdout_chat3b_quality.json")
    ap.add_argument("--tag", default="exp36")
    args = ap.parse_args()

    train_path = ROOT / args.train
    test_path = ROOT / args.test
    train = load_rows(train_path)
    test = load_rows(test_path)
    Xtr = np.stack([features(r["prompt"]) for r in train])
    ytr = np.array([r["label"] for r in train], dtype=np.float64)
    Xte = np.stack([features(r["prompt"]) for r in test])

    # Deterministic CV; all hyperparameter/threshold choices use train only.
    folds = stratified_folds(ytr)
    l2_grid = (0.3, 1.0, 3.0)
    pw_grid = (1.0, 2.0, 4.0)
    th_grid = tuple(round(0.20 + 0.05 * i, 2) for i in range(13))
    cv_candidates = []
    for l2 in l2_grid:
        for pw in pw_grid:
            oof = np.zeros(len(train), dtype=np.float64)
            for val_idx in folds:
                tr_idx = np.setdiff1d(np.arange(len(train)), val_idx)
                w, b = fit_logreg(Xtr[tr_idx], ytr[tr_idx], l2, pw)
                oof[val_idx] = predict(Xtr[val_idx], w, b)
            for safety in (False, True):
                for th in th_grid:
                    m = eval_policy(train, oof, th, safety)
                    cv_candidates.append({
                        "l2": l2, "pos_weight": pw, "threshold": th,
                        "safety": safety, **m,
                    })
    # Quality first, then lower latency, then fewer 3B calls.
    best = max(cv_candidates,
               key=lambda m: (m["quality"], -m["avg_gen_s"],
                              -m["units"]["chat3b"]))
    w, b = fit_logreg(Xtr, ytr, best["l2"], best["pos_weight"])
    tr_probs = predict(Xtr, w, b)
    te_probs = predict(Xte, w, b)
    train_eval = eval_policy(train, tr_probs, best["threshold"],
                             best["safety"])
    test_eval = eval_policy(test, te_probs, best["threshold"],
                            best["safety"])

    # Existing rule router, evaluated on the same rows.
    sys.path.insert(0, str(ROOT))
    from lcsa.service import route  # noqa: E402

    def rule_eval(rows: list[dict]) -> dict:
        q = lat = 0.0
        units = {"chat": 0, "chat3b": 0}
        for r in rows:
            keys, _ = route(r["prompt"], "quality", chat=True, chat3b=True)
            key = keys[0] if keys and keys[0] in units else "chat"
            if key == "chat3b":
                q += r["b_q"]
                lat += r["b_gen_s"]
            else:
                q += r["chat_q"]
                lat += r["chat_gen_s"]
            units[key] += 1
        n = max(1, len(rows))
        return {"quality": round(q / n, 4), "avg_gen_s": round(lat / n, 3),
                "units": units}

    test_curve = []
    for th in th_grid:
        m = eval_policy(test, te_probs, th, best["safety"])
        test_curve.append({"threshold": th, "quality": m["quality"],
                           "avg_gen_s": m["avg_gen_s"],
                           "units": m["units"]})

    summary = {
        "exp": "exp36_learned_router",
        "tag": f"{args.tag}_summary",
        "train": args.train, "test": args.test,
        "train_n": len(train), "test_n": len(test),
        "label": "1 only when chat3b is uniquely better; ties -> chat",
        "selected": {k: best[k] for k in ("l2", "pos_weight", "threshold",
                                           "safety")},
        "cv": {"quality": best["quality"], "avg_gen_s": best["avg_gen_s"],
               "units": best["units"]},
        "train_eval": train_eval,
        "test_eval": test_eval,
        "test_rule_router": rule_eval(test),
        "baselines_test": {
            "always_chat": baseline(test, "chat"),
            "always_chat3b": baseline(test, "chat3b"),
        },
        "test_curve": test_curve,
        "model": {
            "dim": DIM, "weights": [round(float(v), 8) for v in w],
            "bias": round(float(b), 8),
        },
        "note": ("Small n; threshold/hyperparameters chosen by 5-fold CV on "
                 "60 prompts, holdout touched once for final report."),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_learned_router.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    printable = {k: summary[k] for k in
                 ("selected", "cv", "train_eval", "test_eval",
                  "test_rule_router", "baselines_test")}
    print(json.dumps(printable, ensure_ascii=True, indent=2), flush=True)
    print(f"[exp36] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
