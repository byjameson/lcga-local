"""Print a compact human-readable summary of results/results.jsonl."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "results", "results.jsonl")


def main():
    if not os.path.exists(P):
        print("no results yet:", P)
        return
    rows = []
    with open(P, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    for r in rows:
        exp = r.get("exp", "?")
        if exp == "exp1_baseline":
            print("== exp1 dense baseline ==")
            for k, v in r.get("dense", {}).items():
                print(f"  {k:28s} ppl={v['ppl']:8.3f} ce={v['ce']:.4f} ({v['chunks']} chunks)")
            for k, v in r.get("dense_speed", {}).items():
                ms = v["median"] if isinstance(v, dict) else v
                print(f"  {k:28s} {ms}")
        elif exp == "exp2_train":
            print(f"== exp2 train [{r.get('tag')}] step {r.get('step')}: "
                  f"loss={r.get('loss', 0):.4f} mse={r.get('mse', 0):.5f} "
                  f"kl={r.get('kl', 0):.4f} gnorm={r.get('gnorm', 0):.2f}")
        elif exp == "exp2_eval":
            g = r.get("gate") or []
            gm = sum(g) / len(g) if g else float("nan")
            rec = r.get("recall") or []
            print(f"== exp2 eval [{r.get('tag')}] step {r.get('step')}: "
                  f"dense={r.get('dense_ppl', 0):.3f} sparse={r.get('sparse_ppl', 0):.3f} "
                  f"delta={r.get('sparse_ppl', 0) - r.get('dense_ppl', 0):+.3f} "
                  f"recall0={rec[0] if rec else float('nan'):.3f} gate={gm:.3f}")
        elif exp == "exp3_eval":
            print(f"== exp3 eval [{r.get('tag')}] budget={r.get('budget')} ==")
            for name, s in (r.get("sets") or {}).items():
                print(f"  {name:28s} teacher={s['ppl_teacher']:.3f} student={s['ppl_student']:.3f} "
                      f"kl={s['kl_teacher_student']:.4f} top1={s['top1_agreement']:.3f}")
            sw = r.get("budget_sweep") or {}
            for k, v in sw.items():
                if isinstance(v, dict):
                    print(f"  sweep {k:10s} ppl={v['sparse_ppl']:.3f} delta={v['delta']:+.3f} "
                          f"recall={v['recall']:.3f}")
                else:
                    print(f"  sweep {k:10s} {v}")
        elif exp == "exp4_speed":
            print(f"== exp4 speed [{r.get('tag')}] ==")
            for k, v in (r.get("prefill") or {}).items():
                if isinstance(v, dict):
                    print(f"  {k:8s} dense={v['dense_ms']:.1f}ms sparse={v['sparse_ms']:.1f}ms "
                          f"speedup={v['speedup']}x breakdown={v.get('breakdown_ms_per_fwd')}")
                else:
                    print(f"  {k:8s} {v}")
            for k, v in (r.get("decode") or {}).items():
                if isinstance(v, dict):
                    print(f"  {k:8s} dense={v['dense_ms_per_step']:.2f}ms/tok "
                          f"sparse={v['sparse_ms_per_step']:.2f}ms/tok speedup={v['speedup']}x")
                else:
                    print(f"  {k:8s} {v}")
        elif exp == "exp7_train":
            print(f"== exp7 train [{r.get('tag')}] step {r.get('step')}: "
                  f"ce={r.get('ce', 0):.4f} kl={r.get('kl', 0):.4f} "
                  f"mse={r.get('mse', 0):.5f} gnorm={r.get('gnorm', 0):.2f}")
        elif exp == "exp7_eval":
            d, s = r.get("dense_ppl"), r.get("sparse_ppl")
            print(f"== exp7 eval [{r.get('tag')}] step {r.get('step')}: "
                  f"dense={d:.3f} sparse={s:.3f} delta={s - d:+.3f} "
                  f"recall={r.get('recall_mean', float('nan')):.3f} "
                  f"gain={r.get('gain_mean', float('nan')):.3f} "
                  f"alpha={r.get('alpha_mean', float('nan')):.3f}")
        elif exp == "exp8_stream":
            print(f"== exp8 stream [{r.get('tag')}] {r.get('model')} seq={r.get('seq')} "
                  f"cap={r.get('cap_fraction')} ==")
            for m in ("naive", "stream"):
                e = r.get(m) or {}
                if not e:
                    continue
                if e.get("oom"):
                    print(f"  {m:7s} OOM")
                else:
                    print(f"  {m:7s} loss_last={e['loss_last']:.4f} "
                          f"{e['s_per_step']:.2f}s/step peak={e['peak_gb']:.2f}GB "
                          f"lora={e['n_lora']:,}")
            if "loss_absdiff_max" in r:
                print(f"  parity max|diff| = {r['loss_absdiff_max']:.2e}")
        elif exp == "exp9_decode":
            print(f"== exp9 decode [{r.get('tag')}] steps={r.get('steps')} "
                  f"budget={r.get('budget')} gqa={r.get('gqa_sdpa')} ==")
            for ctx, rec in (r.get("contexts") or {}).items():
                if not rec:
                    continue
                if rec.get("error"):
                    print(f"  {ctx}: ERROR {rec['error']}")
                    continue
                print(f"  {ctx}: dense n={rec.get('dense_naive_ms')} gqa={rec.get('dense_fast_gqa_ms')} "
                      f"exp={rec.get('dense_fast_expanded_ms')} | sparse n={rec.get('sparse_naive_ms')} "
                      f"f={rec.get('sparse_fast_ms')} | graph d={rec.get('dense_graph_ms')} "
                      f"s={rec.get('sparse_graph_ms')}")
        elif exp == "exp10_byte":
            print(f"== exp10 byte [{r.get('tag')}] {r.get('model')} "
                  f"params={r.get('params', 0)/1e6:.2f}M steps={r.get('steps')} "
                  f"W={r.get('W')} P={r.get('P')} ==")
            print(f"  best_valid_bpb={r.get('best_valid_bpb')} test_bpb={r.get('test_bpb')} "
                  f"step_med={r.get('step_med_ms')}ms bytes/s={r.get('bytes_per_s')} "
                  f"train_min={r.get('train_min')}")
            evs = r.get("evals") or []
            if evs:
                print("  evals: " + " ".join(f"{e['step']}:{e['valid_bpb']}" for e in evs))
        elif exp == "exp10_byte_verify":
            f = r.get("flat", {})
            p = r.get("patch", {})
            print(f"== exp10 verify [{r.get('tag')}] synced timing + quality re-check ==")
            print(f"  flat : {f.get('ms_sync')}ms (nosync {f.get('ms_nosync')}) "
                  f"valid={f.get('valid_bpb')} test={f.get('test_bpb')}")
            print(f"  patch: {p.get('ms_sync')}ms (nosync {p.get('ms_nosync')}) "
                  f"valid={p.get('valid_bpb')} test={p.get('test_bpb')}")


if __name__ == "__main__":
    sys.exit(main())
