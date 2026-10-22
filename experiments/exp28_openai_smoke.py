"""exp28: live smoke test for the OpenAI-compatible endpoints.

Requires serve.py already running on 127.0.0.1:8000. This is an integration
check, not a quality benchmark: it verifies response shape, SSE framing,
[DONE], and end-to-end latency with the real chat unit.

Usage:
  python experiments/exp28_openai_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = "http://127.0.0.1:8000"


def post(path: str, obj: dict):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=120)


def main() -> int:
    models = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=10).read())
    print("[models]", [m["id"] for m in models["data"]], flush=True)

    t0 = time.perf_counter()
    r = post("/v1/chat/completions", {
        "model": "lcga-chat",
        "messages": [{"role": "user",
                      "content": "Merhaba, kendini bir cumlede tanit."}],
        "max_tokens": 24,
    })
    nonstream_s = time.perf_counter() - t0
    body = json.loads(r.read())
    content = body["choices"][0]["message"]["content"]
    print("[nonstream] %.3fs %r" % (nonstream_s,
          content[:120].encode("ascii", "backslashreplace").decode()), flush=True)

    t0 = time.perf_counter()
    r = post("/v1/chat/completions", {
        "model": "lcga-chat",
        "messages": [{"role": "user", "content": "Uc kisa cumlede kahvenin faydalarini anlat."}],
        "max_tokens": 32, "stream": True,
    })
    chunks = []
    done = False
    ttft = None
    for raw_line in r:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            done = True
            break
        ev = json.loads(payload)
        ch = ev.get("choices") or []
        if ch and ch[0].get("delta", {}).get("content"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(ch[0]["delta"]["content"])
    r.close()
    total_s = time.perf_counter() - t0
    text = "".join(chunks)
    print("[stream] total=%.3fs ttft=%s done=%s chunks=%d %r" % (
        total_s, ("%.3fs" % ttft) if ttft else None, done, len(chunks),
        text[:120].encode("ascii", "backslashreplace").decode()), flush=True)
    if not (body.get("object") == "chat.completion" and content and done and text):
        print("[exp28] FAIL", flush=True)
        return 1

    out = {"exp": "exp28_openai_smoke", "tag": "exp28_openai_smoke",
           "nonstream_s": round(nonstream_s, 3),
           "stream_total_s": round(total_s, 3),
           "stream_ttft_s": round(ttft, 3) if ttft else None,
           "stream_chunks": len(chunks), "done": done,
           "content_chars": len(content), "stream_chars": len(text),
           "measurement": "ttft measured on first incremental SSE content line",
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "results" / "exp28_openai_smoke.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print("[exp28] PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
