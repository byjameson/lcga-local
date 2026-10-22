"""CPU-only HTTP integration tests for serve.py.

Starts the real stdlib HTTP handler with a fake Service, so no model or GPU is
loaded. Covers request validation, JSON responses, SSE formatting, and the
pre-stream validation path.

Run:
  python -m unittest discover -s tests -p "test_http_service.py" -v
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
from lcsa.service import Service  # noqa: E402


class FakeService:
    specs = {"fast": object(), "balanced": object(), "deep": object(),
             "chat": object(), "chat3b": object(), "chat4b": object(),
             "alt": object()}
    counters = {"requests": 0, "errors": 0, "paged_loads": 0,
                "cache_hits": 0, "cache_misses": 0, "queue_s": 0.0}
    captured: list[dict] = []

    def stats(self) -> dict:
        return {
            "counters": dict(self.counters),
            "avg_elapsed_s": 0.0, "avg_gen_s": 0.0,
            "cache": {"size": 0, "capacity": 32, "hits": 0, "misses": 0},
            "queue": {"in_flight": False, "wait_s_total": 0.0},
            "deep_ckpt": None, "chat_model": "fake-instruct",
            "budget_gb": 5.0, "free_vram_gb": 5.0,
        }

    def units_info(self) -> list[dict]:
        return [{"key": "chat", "kind": "dense", "est_gb": 3.3}]

    def preload(self, keys: list[str]) -> dict:
        self.captured.append({"kind": "preload", "units": list(keys)})
        return {"loaded": list(keys), "elapsed_s": 0.0}

    def answer(self, query, mode="auto", max_new_tokens=24, merge="best",
               budget_gb=None, units=None, messages=None) -> dict:
        self.captured.append({"kind": "answer", "query": query, "mode": mode,
                              "messages": messages, "units": units})
        return {"query": query, "answer": "fake answer",
                "history_len": len(messages or []), "route": {"units": ["chat"]}}

    def stream_answer(self, query, mode="auto", max_new_tokens=24, merge="best",
                      budget_gb=None, units=None, messages=None):
        self.captured.append({"kind": "stream", "query": query, "mode": mode,
                              "messages": messages, "units": units})
        yield {"event": "start", "unit": "chat"}
        yield {"event": "token", "text": "fake "}
        yield {"event": "token", "text": "stream"}
        yield {"event": "final", "response": {"answer": "fake stream",
                                              "history_len": len(messages or [])}}


FakeService._clean_messages = staticmethod(Service._clean_messages)


class TestHTTPService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = FakeService()
        serve.Handler.service = cls.svc
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        self.svc.captured.clear()

    def _post(self, path: str, obj: dict):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        return urllib.request.urlopen(req, timeout=10)

    def _get(self, path: str):
        return urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10)

    def test_health_and_units(self):
        h = json.loads(self._get("/health").read())
        self.assertTrue(h["ok"])
        self.assertEqual(h["chat_model"], "fake-instruct")
        u = json.loads(self._get("/units").read())
        self.assertEqual(u["units"][0]["key"], "chat")

    def test_metrics_text(self):
        body = self._get("/metrics").read().decode("utf-8")
        self.assertIn("lcga_requests_total", body)
        self.assertIn("lcga_queue_wait_seconds_total", body)

    def test_answer_query(self):
        r = self._post("/v1/answer", {"query": "merhaba", "mode": "chat",
                                      "max_new_tokens": 4})
        self.assertEqual(r.status, 200)
        body = json.loads(r.read())
        self.assertEqual(body["answer"], "fake answer")
        call = self.svc.captured[-1]
        self.assertEqual(call["query"], "merhaba")
        self.assertEqual(call["mode"], "chat")
        self.assertIsNone(call["messages"])

    def test_answer_messages(self):
        msgs = [{"role": "user", "content": "merhaba"}]
        r = self._post("/v1/answer", {"messages": msgs, "mode": "auto"})
        self.assertEqual(r.status, 200)
        call = self.svc.captured[-1]
        self.assertEqual(call["messages"], msgs)
        self.assertEqual(json.loads(r.read())["history_len"], 1)

    def test_missing_query_or_messages(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/v1/answer", {"mode": "auto"})
        self.assertEqual(cm.exception.code, 400)

    def test_stream_events(self):
        r = self._post("/v1/stream", {"query": "merhaba", "mode": "chat",
                                      "max_new_tokens": 4})
        self.assertEqual(r.status, 200)
        body = r.read().decode("utf-8")
        events = [json.loads(block[6:]) for block in body.split("\n\n")
                  if block.startswith("data: ")]
        self.assertEqual([e["event"] for e in events],
                         ["start", "token", "token", "final"])
        self.assertEqual(events[-1]["response"]["answer"], "fake stream")

    def test_stream_bad_messages_is_400(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/v1/stream", {"messages": [{"role": "assistant",
                                                    "content": "x"}]})
        self.assertEqual(cm.exception.code, 400)

    def test_stream_unknown_mode_is_400(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._post("/v1/stream", {"query": "x", "mode": "turbo"})
        self.assertEqual(cm.exception.code, 400)

    def test_preload(self):
        r = self._post("/v1/preload", {"units": ["deep"]})
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["loaded"], ["deep"])
        self.assertEqual(self.svc.captured[-1]["kind"], "preload")

    def test_openai_models(self):
        body = json.loads(self._get("/v1/models").read())
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], "lcga-chat")
        self.assertEqual([m["id"] for m in body["data"]],
                         ["lcga-chat", "lcga-chat3b", "lcga-chat4b",
                          "lcga-quality", "lcga-alt"])

    def test_openai_chat(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-chat",
            "messages": [{"role": "user", "content": "merhaba"}],
            "max_tokens": 4,
        })
        self.assertEqual(r.status, 200)
        body = json.loads(r.read())
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "fake answer")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.svc.captured[-1]["mode"], "chat")
        self.assertEqual(self.svc.captured[-1]["messages"],
                         [{"role": "user", "content": "merhaba"}])

    def test_openai_chat3b_model(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-chat3b",
            "messages": [{"role": "user", "content": "merhaba"}],
            "max_tokens": 4,
        })
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["model"], "lcga-chat3b")
        self.assertEqual(self.svc.captured[-1]["mode"], "chat3b")

    def test_openai_chat4b_model(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-chat4b",
            "messages": [{"role": "user", "content": "merhaba"}],
            "max_tokens": 4,
        })
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["model"], "lcga-chat4b")
        self.assertEqual(self.svc.captured[-1]["mode"], "chat4b")

    def test_openai_quality_model(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-quality",
            "messages": [{"role": "user", "content": "merhaba"}],
            "max_tokens": 4,
        })
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["model"], "lcga-quality")
        self.assertEqual(self.svc.captured[-1]["mode"], "quality")

    def test_openai_alt_model(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-alt",
            "messages": [{"role": "user", "content": "merhaba"}],
            "max_tokens": 4,
        })
        self.assertEqual(r.status, 200)
        self.assertEqual(json.loads(r.read())["model"], "lcga-alt")
        self.assertEqual(self.svc.captured[-1]["mode"], "alt")

    def test_quality_answer_mode(self):
        r = self._post("/v1/answer", {"query": "merhaba", "mode": "quality",
                                      "max_new_tokens": 4})
        self.assertEqual(r.status, 200)
        self.assertEqual(self.svc.captured[-1]["mode"], "quality")

    def test_openai_chat_system_message(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-chat",
            "messages": [
                {"role": "system", "content": "Kisa cevap ver."},
                {"role": "user", "content": "merhaba"},
            ],
            "max_tokens": 4,
        })
        self.assertEqual(r.status, 200)
        body = json.loads(r.read())
        self.assertEqual(body["choices"][0]["message"]["content"], "fake answer")
        self.assertEqual(self.svc.captured[-1]["messages"][0]["role"], "system")

    def test_openai_chat_stream(self):
        r = self._post("/v1/chat/completions", {
            "model": "lcga-chat",
            "messages": [{"role": "user", "content": "merhaba"}],
            "max_tokens": 4, "stream": True,
        })
        self.assertEqual(r.status, 200)
        body = r.read().decode("utf-8")
        events = []
        for block in body.split("\n\n"):
            if not block.startswith("data: "):
                continue
            payload = block[6:].strip()
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        text = "".join(e["choices"][0]["delta"].get("content", "")
                       for e in events if e.get("choices"))
        self.assertEqual(text, "fake stream")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "stop")
        self.assertIn("data: [DONE]", body)

    def test_openai_validation(self):
        cases = [
            {"messages": [{"role": "user", "content": "x"}], "n": 2},
            {"messages": [{"role": "user", "content": "x"}], "tools": []},
            {"messages": [{"role": "assistant", "content": "x"}]},
        ]
        for body in cases:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self._post("/v1/chat/completions", body)
            self.assertEqual(cm.exception.code, 400)
            err = json.loads(cm.exception.read())["error"]
            self.assertIn("message", err)


if __name__ == "__main__":
    unittest.main()
