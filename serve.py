"""HTTP shell for the LCGA service (Python stdlib only, no extra deps).

Run:
  python serve.py --port 8000
Open http://127.0.0.1:8000/ for the demo UI.

API:
  GET  /health
  GET  /units
  GET  /stats
  GET  /metrics        # Prometheus text
  POST /v1/answer   {"query": "...", "mode": "auto", "max_new_tokens": 24,
                     "merge": "best", "budget_gb": 5.0, "units": ["fast"]}
  POST /v1/stream   same body; SSE-like `data: {"event":...}` events
  POST /v1/preload  {"units": ["deep"]}   # warm a unit before the first query
  GET  /v1/models   # OpenAI-compatible model list
  POST /v1/chat/completions  # OpenAI-compatible chat (stream true/false)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lcsa.service import Service  # noqa: E402

INDEX = r"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>LCGA - 6GB'de &gt;VRAM servis</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root { color-scheme: dark; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #101418; color: #e6edf3; }
  header { padding: 14px 20px; border-bottom: 1px solid #26303a; display: flex; gap: 16px; align-items: baseline; }
  header b { font-size: 16px; }
  header span { color: #8b98a5; font-size: 13px; }
  main { max-width: 900px; margin: 0 auto; padding: 20px; display: grid; gap: 14px; }
  textarea { width: 100%; min-height: 90px; padding: 10px; border-radius: 8px;
             border: 1px solid #2d3a46; background: #161c22; color: inherit; font: inherit; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  select, button { padding: 8px 12px; border-radius: 8px; border: 1px solid #2d3a46;
                   background: #1b232b; color: inherit; font: inherit; }
  button { background: #2563eb; border-color: #2563eb; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: wait; }
  .card { border: 1px solid #26303a; border-radius: 10px; padding: 14px; background: #141a20; }
  .answer { white-space: pre-wrap; font-size: 15px; }
  .meta { color: #8b98a5; font-size: 12.5px; white-space: pre-wrap; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px;
          background: #1f2a35; border: 1px solid #2d3a46; margin-right: 4px; }
  button.ghost { background: #1b232b; border-color: #2d3a46; font-weight: 500; }
  .ex { font-size: 12.5px; padding: 5px 9px; }
  .chatrow { padding: 6px 0; border-bottom: 1px solid #202a33; white-space: pre-wrap; }
  .chatrow:last-child { border-bottom: 0; }
  .chatrow b { color: #8b98a5; font-weight: 600; }
</style></head>
<body>
<header><b>LCGA</b><span>6GB VRAM'de VRAM'e sigmayan modeller: route &rarr; plan &rarr; calistir &rarr; birlestir</span></header>
<main>
  <div class="card">
    <textarea id="q">Fransa'nin baskenti neresi? Kisa acikla.</textarea>
    <div class="row" style="margin-top:10px">
      <select id="mode">
        <option value="auto">auto (sorguya gore)</option>
        <option value="fast">fast - 0.5B</option>
        <option value="balanced">balanced - 1.5B</option>
        <option value="chat">chat - 1.5B Instruct</option>
        <option value="chat3b" id="chat3bopt" disabled>chat3b - 3B Instruct paged (model yok)</option>
        <option value="quality" id="qualityopt" disabled>quality - secici (TR liste 3B, EN liste 1.5B)</option>
        <option value="alt" id="altopt" disabled>alt - SmolLM2 1.7B (Qwen disi)</option>
        <option value="deep">deep - 3B int8 paged</option>
        <option value="ensemble">ensemble - 0.5B + 1.5B</option>
      </select>
      <select id="merge">
        <option value="best">merge: best (guven)</option>
        <option value="concat">merge: concat</option>
      </select>
      <label class="meta">token <input id="n" type="number" value="24" min="4" max="64" style="width:60px"></label>
      <label class="meta"><input id="stream" type="checkbox"> stream</label>
      <label class="meta" title="Deneysel: son 8 mesaj + 5 basit olgu icin sablon bellek. Genel hafiza sinirli (exp25c 2/5)."><input id="hist" type="checkbox"> gecmis (deneysel)</label>
      <button id="go">Gonder</button>
      <button id="pre" class="ghost" title="3B paged birimini simdi yukle">deep on-yukle</button>
      <span id="status" class="meta">hazir</span>
    </div>
    <div class="row" style="margin-top:8px">
      <span class="meta">ornek:</span>
      <button class="ghost ex" data-q="Fransa'nin baskenti neresi? Kisa acikla.">bilgi</button>
      <button class="ghost ex" data-q="Bu Python kodundaki hatayi acikla:&#10;&#10;def add(a, b):&#10;    return a - b">kod</button>
      <button class="ghost ex" data-q="6GB VRAM'de 3B modeli nasil calistiririz? Adim adim acikla.">deep</button>
    </div>
  </div>
  <div class="card" id="chatcard" style="display:none">
    <div id="chatlog"></div>
    <div class="row" style="margin-top:8px">
      <button id="clear" class="ghost">gecmisi temizle</button>
      <span class="meta">deneysel: son 8 mesaj + 5 basit olgu sablonu; genel hafiza sinirli</span>
    </div>
  </div>
  <div class="card">
    <div class="answer" id="answer">-</div>
  </div>
  <div class="card meta" id="meta">-</div>
</main>
<script>
const $ = (id) => document.getElementById(id);
let history = [];
function renderHistory() {
  $('chatcard').style.display = $('hist').checked ? 'block' : 'none';
  const box = $('chatlog'); box.replaceChildren();
  history.forEach((m) => {
    const d = document.createElement('div'); d.className = 'chatrow';
    const b = document.createElement('b');
    b.textContent = (m.role === 'user' ? 'Sen' : 'Asistan') + ': ';
    d.appendChild(b); d.appendChild(document.createTextNode(m.content));
    box.appendChild(d);
  });
}
function pushTurn(role, content) {
  history.push({role, content});
  if (history.length > 8) history = history.slice(-8);
  renderHistory();
}
function memoryFromHistory() {
  const facts = [];
  let name = null;
  for (const m of history) {
    if (m.role !== 'user') continue;
    const t = m.content;
    let x;
    if ((x = t.match(/benim ad[iı]m\s+(?!ne\b|nedir\b|kim\b)([^\s.,!?]+)/i))) {
      name = x[1]; facts.push(`Kullanicinin adi ${name}.`);
    } else if ((x = t.match(/kedimin ad[iı]\s+(?!ne\b|nedir\b)([^\s.,!?]+)/i))) {
      facts.push(`Kedinin adi ${x[1]}.`);
    } else if ((x = t.match(/en sevdi[gğ]im renk\s+(?!ne\b|nedir\b|hangisi\b)([^\s.,!?]+)/i))) {
      facts.push(`Kullanicinin en sevdigi renk ${x[1]}.`);
    } else if ((x = t.match(/bug[uü]n hava\s+(?!nasil\b|ne\b)(.+?)[.!]?$/i))) {
      facts.push(`Bugun hava ${x[1]}.`);
    } else if (/yar[iı]n (sabah )?toplant[iı]m var/i.test(t)) {
      facts.push('Yarin sabah toplanti var.');
    }
  }
  return {facts, name};
}
function augmentQuestion(q) {
  const mem = memoryFromHistory();
  if (!mem.facts.length) return q;
  if (mem.name && /(benim )?ad[iı]m ne/i.test(q)) {
    return `Kullanicinin adi ne? Bilinen bilgi: kullanicinin adi ${mem.name}.`;
  }
  return `[Bilinen bilgiler: ${mem.facts.join(' ')}]\n${q}`;
}
async function refresh() {
  try {
    const r = await fetch('/health'); const j = await r.json();
    const ur = await fetch('/units'); const uj = await ur.json();
    const has3b = (uj.units || []).some(u => u.key === 'chat3b');
    const hasAlt = (uj.units || []).some(u => u.key === 'alt');
    const opt = $('chat3bopt');
    if (opt) {
      opt.disabled = !has3b;
      opt.textContent = has3b
        ? 'chat3b - 3B Instruct paged (yavas, kaliteli)'
        : 'chat3b - 3B Instruct paged (model yok)';
    }
    const qopt = $('qualityopt');
    if (qopt) {
      qopt.disabled = !has3b;
      qopt.textContent = has3b
        ? 'quality - secici (TR liste 3B, EN liste 1.5B)'
        : 'quality - secici (3B model yok)';
    }
    const aopt = $('altopt');
    if (aopt) {
      aopt.disabled = !hasAlt;
      aopt.textContent = hasAlt
        ? 'alt - SmolLM2 1.7B (Qwen disi)'
        : 'alt - SmolLM2 (model yok)';
    }
    $('status').textContent = `budget ${j.budget_gb}GB | bos VRAM ${j.free_vram_gb}GB | ` +
      `ckpt ${j.deep_ckpt || 'base'} | chat3b ${has3b ? 'var' : 'yok'} | ` +
      `alt ${hasAlt ? 'var' : 'yok'} | ` +
      `istek ${j.counters.requests} | cache ${j.cache.hits}`;
  } catch (e) { $('status').textContent = 'servis yok'; }
}
async function preload() {
  $('pre').disabled = true; $('status').textContent = 'deep yukleniyor...';
  try {
    const r = await fetch('/v1/preload', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({units: ['deep']})});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.statusText);
    $('status').textContent = `deep hazir (${j.elapsed_s}s, bos VRAM ${j.free_vram_gb}GB)`;
  } catch (e) {
    $('status').textContent = 'preload hatasi: ' + e.message;
  } finally { $('pre').disabled = false; }
}
function showMeta(j) {
  const plan = j.plan.map(g => g.units.join(' + ') + ' [' + g.kind + ', ' + g.wall_s + 's]').join('  ->  ');
  const units = Object.entries(j.per_unit).map(([k, v]) =>
    `${k}: ${v.gen_s}s` + (v.score !== null ? `, skor ${v.score}` : '')).join(' | ');
  $('meta').textContent =
    `rota: ${j.route.reason}\nplan (${j.n_groups} grup${j.parallel ? ', paralel' : ''}): ${plan}\n` +
    `birimler: ${units}\ntoplam ${j.elapsed_s}s (yukleme ${j.timings.load_s}s, ` +
    `kuyruk ${j.timings.queue_s || 0}s, uretim ${j.timings.gen_s}s) | ` +
    `peak VRAM ${j.peak_vram_mb}MB | merge=${j.merge} | istek #${j.request_id}` +
    (j.cached ? ` (cache; ilk ${j.cached_orig_elapsed_s}s)` : '') +
    (j.warnings && j.warnings.length ? `\nikaz: ${j.warnings.join(' | ')}` : '');
}
async function ask() {
  const q = $('q').value.trim(); if (!q) return;
  const useHistory = $('hist').checked;
  const body = {merge: $('merge').value, max_new_tokens: Number($('n').value)};
  if (useHistory) {
    body.mode = ['chat', 'chat3b', 'alt'].includes($('mode').value)
      ? $('mode').value : 'chat';
    body.messages = history.concat([{role: 'user', content: augmentQuestion(q)}]);
  } else {
    body.mode = $('mode').value;
    body.query = q;
  }
  $('go').disabled = true; $('status').textContent = 'calisiyor...';
  $('answer').textContent = '...'; $('meta').textContent = '';
  try {
    if (!$('stream').checked) {
      const r = await fetch('/v1/answer', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.statusText);
      $('answer').textContent = j.answer || '(bos)';
      if (useHistory) {
        pushTurn('user', q);
        pushTurn('assistant', j.answer || '(bos)');
      }
      showMeta(j);
    } else {
      const r = await fetch('/v1/stream', {method: 'POST',
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      if (!r.ok || !(r.headers.get('content-type') || '').includes('text/event-stream')) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j.error || r.statusText);
      }
      const reader = r.body.getReader(); const dec = new TextDecoder();
      let buf = '', text = '', final = null;
      $('answer').textContent = '';
      while (true) {
        const {value, done} = await reader.read(); if (done) break;
        buf += dec.decode(value, {stream: true});
        let idx;
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const block = buf.slice(0, idx); buf = buf.slice(idx + 2);
          if (!block.startsWith('data: ')) continue;
          const ev = JSON.parse(block.slice(6));
          if (ev.event === 'token') {
            text += ev.text; $('answer').textContent = text;
          } else if (ev.event === 'final') {
            final = ev.response;
          } else if (ev.event === 'error') {
            throw new Error(ev.error);
          }
        }
      }
      if (final) {
        $('answer').textContent = final.answer || text;
        if (useHistory) {
          pushTurn('user', q);
          pushTurn('assistant', final.answer || text || '(bos)');
        }
        showMeta(final);
      } else {
        $('answer').textContent = text || '(bos)';
      }
    }
  } catch (e) {
    $('answer').textContent = 'Hata: ' + e.message;
  } finally { $('go').disabled = false; refresh(); }
}
$('go').onclick = ask;
$('pre').onclick = preload;
$('hist').onchange = renderHistory;
$('clear').onclick = () => {
  history = []; renderHistory();
  $('answer').textContent = '-'; $('meta').textContent = '-';
};
document.querySelectorAll('.ex').forEach((b) => {
  b.onclick = () => { $('q').value = b.dataset.q; $('q').focus(); };
});
$('q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask(); });
renderHistory();
refresh();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    service: Service = None
    server_version = "LCGA/1.0"

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, INDEX.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._json(200, {"ok": True, "ts": round(time.time(), 3),
                             **self.service.stats()})
        elif self.path == "/units":
            self._json(200, {"units": self.service.units_info()})
        elif self.path == "/v1/models":
            models = [{
                "id": "lcga-chat", "object": "model",
                "created": 0, "owned_by": "lcga",
                "description": "Local Qwen2.5-1.5B-Instruct chat unit",
            }]
            if "chat3b" in self.service.specs:
                models.append({
                    "id": "lcga-chat3b", "object": "model",
                    "created": 0, "owned_by": "lcga",
                    "description": "Local Qwen2.5-3B-Instruct int8 paged quality unit",
                })
            if "chat4b" in self.service.specs:
                models.append({
                    "id": "lcga-chat4b", "object": "model",
                    "created": 0, "owned_by": "lcga",
                    "description": "Local Qwen3-4B-Instruct-2507 int8 GPU-resident quality unit",
                })
            if "chat3b" in self.service.specs or "chat4b" in self.service.specs:
                models.append({
                    "id": "lcga-quality", "object": "model",
                    "created": 0, "owned_by": "lcga",
                    "description": ("Task router: 4B for general, 1.5B for social"
                                    if "chat4b" in self.service.specs else
                                    "Task router: 3B for Turkish lists/general, 1.5B for English lists/social"),
                })
            if "alt" in self.service.specs:
                models.append({
                    "id": "lcga-alt", "object": "model",
                    "created": 0, "owned_by": "lcga",
                    "description": "Optional non-Qwen SmolLM2-1.7B-Instruct adapter test",
                })
            self._json(200, {"object": "list", "data": models})
        elif self.path == "/stats":
            self._json(200, self.service.stats())
        elif self.path == "/metrics":
            s = self.service.stats()
            c = s["counters"]
            lines = [
                "# HELP lcga_requests_total Requests answered (including cache hits).",
                "# TYPE lcga_requests_total counter",
                f"lcga_requests_total {c['requests']}",
                "# TYPE lcga_errors_total counter",
                f"lcga_errors_total {c['errors']}",
                "# TYPE lcga_cache_hits_total counter",
                f"lcga_cache_hits_total {c['cache_hits']}",
                "# TYPE lcga_cache_misses_total counter",
                f"lcga_cache_misses_total {c['cache_misses']}",
                "# TYPE lcga_paged_loads_total counter",
                f"lcga_paged_loads_total {c['paged_loads']}",
                "# TYPE lcga_avg_elapsed_seconds gauge",
                f"lcga_avg_elapsed_seconds {s['avg_elapsed_s']}",
                "# TYPE lcga_avg_gen_seconds gauge",
                f"lcga_avg_gen_seconds {s['avg_gen_s']}",
                "# TYPE lcga_queue_wait_seconds_total counter",
                f"lcga_queue_wait_seconds_total {c['queue_s']}",
                "# TYPE lcga_free_vram_gb gauge",
                f"lcga_free_vram_gb {s['free_vram_gb'] or 0}",
            ]
            self._send(200, ("\n".join(lines) + "\n").encode("utf-8"),
                       "text/plain; version=0.0.4; charset=utf-8")
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/v1/answer", "/v1/preload", "/v1/stream",
                             "/v1/chat/completions"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0 or n > 1_000_000:
                return self._json(400, {"error": "bad body"})
            req = json.loads(self.rfile.read(n).decode("utf-8"))
            if self.path == "/v1/chat/completions":
                return self._openai_chat(req)
            if self.path == "/v1/preload":
                units = req.get("units")
                if not isinstance(units, list) or not units:
                    raise ValueError("units list required")
                return self._json(200, self.service.preload([str(u) for u in units]))
            q = str(req.get("query", "")).strip()
            messages = req.get("messages")
            if messages is not None and not isinstance(messages, list):
                raise ValueError("messages must be a list")
            if not q and messages is None:
                return self._json(400, {"error": "query or messages required"})
            units = req.get("units")
            if units is not None and not isinstance(units, list):
                raise ValueError("units must be a list")
            unit_keys = [str(u) for u in units] if units else None
            mode = str(req.get("mode", "auto"))
            if mode not in ("auto", "chat", "chat3b", "chat4b", "alt", "quality",
                            "fast", "balanced", "deep", "ensemble"):
                raise ValueError(f"unknown mode: {mode}")
            if mode == "chat" and "chat" not in self.service.specs:
                raise ValueError(
                    "chat unit is not available (model missing or --no-chat)")
            if mode == "chat3b" and "chat3b" not in self.service.specs:
                raise ValueError("chat3b unit is not available (3B instruct model missing)")
            if mode == "chat4b" and "chat4b" not in self.service.specs:
                raise ValueError("chat4b unit is not available (Qwen3-4B model missing)")
            if mode == "alt" and "alt" not in self.service.specs:
                raise ValueError("alt unit is not available (SmolLM model missing)")
            if self.path == "/v1/stream":
                # stream_answer is a generator: validate before sending 200 headers
                if unit_keys:
                    bad = [k for k in unit_keys if k not in self.service.specs]
                    if bad:
                        raise ValueError(f"unknown unit(s): {bad}")
                if messages is not None:
                    self.service._clean_messages(messages)
                    if unit_keys:
                        raise ValueError("messages and units are mutually exclusive")
                    if mode not in ("auto", "chat", "chat3b", "chat4b", "alt",
                                    "quality"):
                        raise ValueError(
                            "messages are only supported with mode auto/chat/chat3b/chat4b/alt/quality")
            gen = self.service.stream_answer(
                q,
                mode=mode,
                max_new_tokens=int(req.get("max_new_tokens", 24)),
                merge=str(req.get("merge", "best")),
                budget_gb=req.get("budget_gb"),
                units=unit_keys,
                messages=messages,
            ) if self.path == "/v1/stream" else None
            if gen is not None:
                return self._stream(gen)
            out = self.service.answer(
                q,
                mode=mode,
                max_new_tokens=int(req.get("max_new_tokens", 24)),
                merge=str(req.get("merge", "best")),
                budget_gb=req.get("budget_gb"),
                units=unit_keys,
                messages=messages,
            )
            self._json(200, out)
        except ValueError as e:
            if self.path == "/v1/chat/completions":
                return self._openai_error(400, str(e))
            self._json(400, {"error": str(e)})
        except Exception as e:  # keep the server alive; report honestly
            self.service.counters["errors"] += 1
            if self.path == "/v1/chat/completions":
                return self._openai_error(500, f"{type(e).__name__}: {e}",
                                          "server_error")
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def _openai_error(self, code: int, message: str,
                      err_type: str = "invalid_request_error") -> None:
        self._json(code, {"error": {"message": message, "type": err_type,
                                    "param": None, "code": None}})

    def _openai_chat(self, req: dict) -> None:
        """Minimal OpenAI-compatible chat endpoint.

        The local service is deterministic. `model` is echoed back for client
        compatibility; `lcga-chat4b` selects the Qwen3-4B int8 GPU-resident
        unit, `lcga-chat3b` the optional 3B paged unit, `lcga-quality` the
        measured 4B/3B + 1.5B task router, `lcga-alt` the optional non-Qwen
        SmolLM adapter test, and every other model id uses the fast 1.5B chat
        unit. Unsupported OpenAI features are rejected explicitly instead of
        being silently ignored.
        """
        if not isinstance(req, dict):
            raise ValueError("JSON object required")
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        clean = self.service._clean_messages(messages)
        if any(k in req for k in ("tools", "functions", "tool_choice")):
            raise ValueError("tools/functions are not supported")
        try:
            n = int(req.get("n", 1) or 1)
        except (TypeError, ValueError):
            raise ValueError("n must be an integer")
        if n != 1:
            raise ValueError("only n=1 is supported")
        raw_max = req.get("max_tokens", req.get("max_completion_tokens", 64))
        try:
            max_tokens = int(raw_max)
        except (TypeError, ValueError):
            raise ValueError("max_tokens must be an integer")
        if not 1 <= max_tokens <= 256:
            raise ValueError("max_tokens must be between 1 and 256")
        model = str(req.get("model") or "lcga-chat").strip()
        if model.lower() in ("lcga-quality", "quality"):
            mode = "quality"
        elif model.lower() in ("lcga-chat4b", "chat4b",
                               "qwen3-4b-instruct-2507"):
            mode = "chat4b"
        elif model.lower() in ("lcga-alt", "alt", "smollm2-1.7b-instruct"):
            mode = "alt"
        elif model.lower() in ("lcga-chat3b", "chat3b",
                               "qwen2.5-3b-instruct",
                               "qwen/qwen2.5-3b-instruct"):
            mode = "chat3b"
        else:
            mode = "chat"
        if mode == "chat4b":
            if "chat4b" not in self.service.specs:
                raise ValueError(
                    "chat4b unit is not available (Qwen3-4B model missing)")
        elif mode == "chat3b":
            if "chat3b" not in self.service.specs:
                raise ValueError(
                    "chat3b unit is not available (3B instruct model missing)")
        elif mode == "alt":
            if "alt" not in self.service.specs:
                raise ValueError(
                    "alt unit is not available (SmolLM model missing)")
        elif mode == "quality":
            if not any(k in self.service.specs
                       for k in ("chat", "chat3b", "chat4b")):
                raise ValueError("no chat unit available for quality mode")
        elif "chat" not in self.service.specs:
            raise ValueError(
                "chat unit is not available (model missing or --no-chat)")
        cid = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        if req.get("stream") is True:
            gen = self.service.stream_answer("", mode=mode,
                                             max_new_tokens=max_tokens,
                                             messages=clean)
            return self._openai_stream(gen, cid, created, model)
        out = self.service.answer("", mode=mode, max_new_tokens=max_tokens,
                                  messages=clean)
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": str(out.get("answer", ""))},
                "finish_reason": "stop",
            }],
            "system_fingerprint": "lcga",
        })

    def _openai_stream(self, gen, cid: str, created: int, model: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        def write(obj) -> None:
            payload = "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"
            self.wfile.write(payload.encode("utf-8"))
            self.wfile.flush()

        def done() -> None:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def chunk(delta: dict, finish=None) -> dict:
            return {"id": cid, "object": "chat.completion.chunk",
                    "created": created, "model": model,
                    "choices": [{"index": 0, "delta": delta,
                                 "finish_reason": finish}]}

        sent = ""
        finished = False
        try:
            write(chunk({"role": "assistant", "content": ""}))
            for ev in gen:
                kind = ev.get("event")
                if kind == "token":
                    text = str(ev.get("text", ""))
                    if text:
                        sent += text
                        write(chunk({"content": text}))
                elif kind == "final":
                    if not sent:
                        text = str((ev.get("response") or {}).get("answer", ""))
                        if text:
                            sent = text
                            write(chunk({"content": text}))
                    write(chunk({}, "stop"))
                    finished = True
                elif kind == "error":
                    write({"error": {"message": str(ev.get("error", "error")),
                                     "type": "server_error"}})
            if not finished:
                write(chunk({}, "stop"))
            done()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                write({"error": {"message": f"{type(e).__name__}: {e}",
                                 "type": "server_error"}})
                done()
            except Exception:
                pass
        finally:
            gen.close()

    def _stream(self, gen) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        try:
            for ev in gen:
                line = "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                payload = {"event": "error",
                           "error": f"{type(e).__name__}: {e}"}
                self.wfile.write(("data: " + json.dumps(payload) +
                                  "\n\n").encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass
        finally:
            gen.close()

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--budget", type=float, default=5.0, help="VRAM budget in GB")
    ap.add_argument("--deep-ckpt", default="auto",
                    help="state_dict for the deep (3B paged) unit; 'auto' uses "
                         "checkpoints/qwen3b_full_adaf_t300_best.pt when present, "
                         "'none' serves the base model")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-log", action="store_true",
                    help="do not append requests to results/service_requests.jsonl")
    ap.add_argument("--cache", type=int, default=32,
                    help="LRU response cache size (0 disables)")
    ap.add_argument("--no-chat", action="store_true",
                    help="do not register the instruct chat unit")
    args = ap.parse_args()

    ckpt = args.deep_ckpt
    if ckpt.lower() == "auto":
        cand = ROOT / "checkpoints/qwen3b_full_adaf_t300_best.pt"
        ckpt = str(cand) if cand.exists() else None
    elif ckpt.lower() in ("", "none", "base"):
        ckpt = None
    svc = Service(budget_gb=args.budget, verbose=args.verbose, deep_ckpt=ckpt,
                  log_requests=not args.no_log, cache_size=args.cache,
                  enable_chat=not args.no_chat)
    Handler.service = svc
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    ck = f" deep_ckpt={Path(ckpt).name}" if ckpt else " deep_ckpt=base"
    chat = f" chat={Path(svc.chat_model).name}" if svc.chat_model else ""
    print(f"[serve] http://{args.host}:{args.port} budget={args.budget}GB{ck}{chat} "
          f"(Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[serve] stopped", flush=True)
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
