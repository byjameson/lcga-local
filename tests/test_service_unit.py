"""CPU-only unit tests for routing, planning and merging.

Run:
  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.scheduler import Unit, merge_outputs, plan  # noqa: E402
from lcsa.service import Service, repair_list_text, route  # noqa: E402


class TestRoute(unittest.TestCase):
    def test_greeting_goes_fast(self):
        keys, reason = route("merhaba, nasilsin?")
        self.assertEqual(keys, ["fast"])
        self.assertIn("selamlama", reason)

    def test_code_cue_goes_deep(self):
        keys, reason = route("Bu Python kodundaki hatayi acikla ve optimize et")
        self.assertEqual(keys, ["deep"])
        self.assertIn("3B", reason)

    def test_short_query_goes_fast(self):
        keys, _ = route("Fransa'nin baskenti neresi?")
        self.assertEqual(keys, ["fast"])

    def test_long_generic_is_parallel_pair(self):
        q = ("Bu paragrafi Turkce olarak ozetle ve ana fikri birkac cumleyle "
             "acikla, ardindan kisa bir sonuc yaz.")
        keys, _ = route(q)
        self.assertEqual(keys, ["fast", "balanced"])

    def test_explicit_modes(self):
        for mode in ("fast", "balanced", "deep"):
            self.assertEqual(route("herhangi bir sey", mode)[0], [mode])
        self.assertEqual(route("herhangi bir sey", "ensemble")[0],
                         ["fast", "balanced"])

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            route("x", "turbo")

    def test_route_keys_exist(self):
        from lcsa.service import UNIT_SPECS
        known = {s.key for s in UNIT_SPECS} | {"chat"}
        for q in ("merhaba", "kod yaz", "uzun bir metni ozetle " * 5):
            keys, _ = route(q)
            self.assertTrue(set(keys) <= known)

    def test_chat_route_when_available(self):
        self.assertEqual(route("Merhaba, nasilsin?", "auto", chat=True)[0],
                         ["chat"])
        self.assertEqual(route("Fransa'nin baskenti neresi?", "auto",
                               chat=True)[0], ["chat"])
        self.assertEqual(route("Fransa'nin baskenti neresi?", "auto")[0],
                         ["fast"])

    def test_chat_mode_requires_unit(self):
        with self.assertRaises(ValueError):
            route("x", "chat", chat=False)

    def test_chat3b_mode_requires_unit(self):
        with self.assertRaises(ValueError):
            route("x", "chat3b", chat3b=False)
        self.assertEqual(route("x", "chat3b", chat3b=True)[0], ["chat3b"])
        self.assertEqual(route("x", "auto", chat3b=True)[0], ["fast"])

    def test_alt_mode_requires_unit(self):
        with self.assertRaises(ValueError):
            route("x", "alt", alt=False)
        self.assertEqual(route("x", "alt", alt=True)[0], ["alt"])

    def test_chat4b_mode_requires_unit(self):
        with self.assertRaises(ValueError):
            route("x", "chat4b", chat4b=False)
        self.assertEqual(route("x", "chat4b", chat4b=True)[0], ["chat4b"])
        self.assertEqual(route("x", "auto", chat4b=True)[0], ["fast"])

    def test_quality_prefers_chat4b_when_available(self):
        kw = dict(chat=True, chat3b=True, chat4b=True)
        self.assertEqual(route("Fransa'nin baskenti neresi?", "quality",
                               **kw)[0], ["chat4b"])
        self.assertEqual(route("8 * 9 = ?", "quality", **kw)[0], ["chat4b"])
        self.assertEqual(route("Uc meyve listele.", "quality", **kw)[0],
                         ["chat4b"])
        self.assertEqual(route("List three fruits.", "quality", **kw)[0],
                         ["chat4b"])
        self.assertEqual(route("Bu Python kodundaki hatayi acikla", "quality",
                               **kw)[0], ["chat4b"])
        self.assertEqual(route("Tesekkur eden bir cumle yaz.", "quality",
                               **kw)[0], ["chat"])

    def test_quality_route_uses_language_aware_list_units(self):
        kw = dict(chat=True, chat3b=True)
        self.assertEqual(route("Fransa'nin baskenti neresi?", "quality",
                               **kw)[0], ["chat3b"])
        self.assertEqual(route("8 * 9 = ?", "quality", **kw)[0], ["chat3b"])
        self.assertEqual(route("Uc meyve listele.", "quality", **kw)[0],
                         ["chat3b"])
        self.assertEqual(route("Uc maddede su icmenin faydalarini yaz.",
                               "quality", **kw)[0], ["chat3b"])
        self.assertEqual(route("List three fruits.", "quality", **kw)[0],
                         ["chat"])
        self.assertEqual(route("Tesekkur eden bir cumle yaz.", "quality",
                               **kw)[0], ["chat"])
        self.assertEqual(route("Listen to music", "quality", **kw)[0],
                         ["chat3b"])
        self.assertEqual(route("Fransa'nin baskenti neresi?", "quality",
                               chat=True, chat3b=False)[0], ["chat"])

    def test_quality_route_ignores_alt_in_quality_mode(self):
        kw = dict(chat=True, chat3b=True, alt=True)
        self.assertEqual(route("Uc meyve listele.", "quality", **kw)[0],
                         ["chat3b"])
        self.assertEqual(route("List three fruits.", "quality", **kw)[0],
                         ["chat"])
        self.assertEqual(route("Tesekkur eden bir cumle yaz.", "quality",
                               **kw)[0], ["chat"])
        self.assertEqual(route("Fransa'nin baskenti neresi?", "quality",
                               **kw)[0], ["chat3b"])

    def test_repair_list_text_preserves_content(self):
        self.assertEqual(
            repair_list_text("Red, blue, and yellow are the primary colors."),
            "1. Red\n2. blue\n3. yellow")
        self.assertEqual(repair_list_text("Monday, Tuesday, Wednesday"),
                         "1. Monday\n2. Tuesday\n3. Wednesday")
        already = "1. Dog\n2. Cat\n3. Bird"
        self.assertEqual(repair_list_text(already), already)
        self.assertEqual(repair_list_text("not a list"), "not a list")


class TestPlan(unittest.TestCase):
    @staticmethod
    def unit(name: str, gb: float) -> Unit:
        return Unit(name, model_path=f"models/{name}", est_gb=gb)

    def test_parallel_when_fits(self):
        units = [self.unit("a", 1.1), self.unit("b", 3.3)]
        self.assertEqual(len(plan(units, 5.0)), 1)

    def test_sequential_when_overflow(self):
        units = [self.unit("a", 3.3), self.unit("b", 3.3)]
        self.assertEqual(len(plan(units, 5.0)), 2)

    def test_empty(self):
        self.assertEqual(plan([], 5.0), [])

    def test_oversized_unit_runs_alone(self):
        units = [self.unit("a", 0.5), self.unit("b", 9.0)]
        groups = plan(units, 5.0)
        self.assertEqual([[u.name for u in g] for g in groups],
                         [["a"], ["b"]])


class TestMerge(unittest.TestCase):
    @staticmethod
    def unit(name: str, text: str, score: float | None = None) -> Unit:
        return Unit(name, model_path="x", est_gb=1.0, text=text, score=score)

    def test_best_picks_highest_score(self):
        units = [self.unit("fast", "kisa", -0.5), self.unit("balanced", "uzun", -0.2)]
        self.assertEqual(merge_outputs(units, "best"), "uzun")

    def test_best_falls_back_to_concat_without_scores(self):
        units = [self.unit("fast", " bir"), self.unit("balanced", " iki")]
        self.assertEqual(merge_outputs(units, "best"), "[fast] bir | [balanced] iki")

    def test_concat_labels_every_unit(self):
        units = [self.unit("a", "x", -0.1), self.unit("b", "y", -0.2)]
        self.assertEqual(merge_outputs(units, "concat"), "[a] x | [b] y")

    def test_single_unit_has_no_label(self):
        self.assertEqual(merge_outputs([self.unit("deep", " cevap")], "concat"),
                         "cevap")


class TestServiceValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = Service(budget_gb=5.0, deep_ckpt=None, log_requests=False)

    def test_unknown_unit_rejected(self):
        with self.assertRaises(ValueError):
            self.svc.answer("merhaba", units=["nope"])

    def test_cache_key_normalizes_whitespace(self):
        a = self.svc._cache_key("  hello   world ", "auto", 24, "best", 5.0,
                                ["fast"])
        b = self.svc._cache_key("hello world", "auto", 24, "best", 5.0,
                                ["fast"])
        self.assertEqual(a, b)

    def test_cache_key_distinguishes_units_and_tokens(self):
        a = self.svc._cache_key("q", "auto", 24, "best", 5.0, ["fast"])
        b = self.svc._cache_key("q", "auto", 24, "best", 5.0, ["balanced"])
        c = self.svc._cache_key("q", "auto", 8, "best", 5.0, ["fast"])
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_units_info_has_core_units(self):
        keys = [u["key"] for u in self.svc.units_info()]
        self.assertTrue({"fast", "balanced", "deep"} <= set(keys))
        self.assertLessEqual(set(keys),
                             {"fast", "balanced", "deep", "chat", "chat3b",
                              "chat4b", "alt"})

    def test_clean_messages_trims_history(self):
        msgs = []
        for i in range(10):
            msgs.append({"role": "user" if i % 2 == 0 else "assistant",
                         "content": f"m{i}"})
        msgs.append({"role": "user", "content": "son"})
        out = self.svc._clean_messages(msgs)
        self.assertEqual(len(out), 8)
        self.assertEqual(out[-1], {"role": "user", "content": "son"})
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in out))
        long = self.svc._clean_messages([{"role": "user", "content": "x" * 3000}])
        self.assertEqual(len(long[0]["content"]), 2000)

    def test_clean_messages_rejects_bad_input(self):
        bad = [
            [],
            "merhaba",
            [{"role": "system", "content": "x"}],
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
            [{"role": "user", "content": "   "}],
            [{"role": "user"}],
        ]
        for msgs in bad:
            with self.assertRaises(ValueError):
                self.svc._clean_messages(msgs)

    def test_messages_require_chat_unit(self):
        no_chat = Service(budget_gb=5.0, deep_ckpt=None, log_requests=False,
                          enable_chat=False)
        with self.assertRaises(ValueError):
            no_chat.answer("", messages=[{"role": "user", "content": "merhaba"}])
        with self.assertRaises(ValueError):
            no_chat.answer("", mode="chat3b",
                           messages=[{"role": "user", "content": "merhaba"}])

    def test_fit_prompt_keeps_recent_tokens(self):
        from types import SimpleNamespace
        import torch
        eng = SimpleNamespace(max_seq=10)
        ids = torch.arange(20).reshape(1, -1)
        out = Service._fit_prompt(eng, ids, 3)
        self.assertEqual(tuple(out.shape), (1, 6))
        self.assertEqual(int(out[0, -1]), 19)

    def test_messages_reject_units_and_non_chat_mode(self):
        msgs = [{"role": "user", "content": "merhaba"}]
        with self.assertRaises(ValueError):
            self.svc.answer("", units=["fast"], messages=msgs)
        with self.assertRaises(ValueError):
            self.svc.answer("", mode="deep", messages=msgs)

    def test_cache_key_distinguishes_history(self):
        a = self.svc._cache_key("q", "chat", 24, "best", 5.0, ["chat"],
                                [{"role": "user", "content": "a"}])
        b = self.svc._cache_key("q", "chat", 24, "best", 5.0, ["chat"],
                                [{"role": "user", "content": "b"}])
        c = self.svc._cache_key("q", "chat", 24, "best", 5.0, ["chat"],
                                [{"role": "user", "content": "a"}])
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)


if __name__ == "__main__":
    unittest.main()
