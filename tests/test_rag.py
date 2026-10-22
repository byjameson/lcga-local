"""CPU-only tests for the local BM25 retrieval layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.rag import RagStore, stem, tokenize  # noqa: E402


DOCS = [
    {
        "id": "tr:kahire",
        "title": "Kahire",
        "lang": "tr",
        "text": ("Kahire, Mısır'ın başkenti ve en büyük şehridir. Nil Nehri "
                 "kıyısında yer alır. Şehirde Gize piramitleri bulunur."),
        "source": "https://example.invalid/kahire",
    },
    {
        "id": "tr:ankara",
        "title": "Ankara",
        "lang": "tr",
        "text": ("Ankara, Türkiye'nin başkentidir. İç Anadolu Bölgesi'nde "
                 "yer alan bir şehirdir."),
        "source": "https://example.invalid/ankara",
    },
    {
        "id": "en:water",
        "title": "Water",
        "lang": "en",
        "text": ("Water is a chemical compound with the formula H2O. At "
                 "standard pressure water boils at 100 degrees Celsius and "
                 "freezes at 0 degrees Celsius."),
        "source": "https://example.invalid/water",
    },
]


class TestRagStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p = Path(cls.tmp.name) / "docs.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for d in DOCS:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        cls.store = RagStore(p, min_coverage=0.5, min_terms=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_tokenize_is_turkish_aware(self):
        self.assertEqual(tokenize("Mısır'ın BAŞKENTİ neresi?"),
                         ["misir", "baskent"])
        self.assertEqual(tokenize("Narnia'nın başkenti neresi?"),
                         ["narnia", "baskent"])
        # "su" is content, not a question word.
        self.assertEqual(tokenize("Su kaç derecede kaynar?"),
                         ["su", "derece", "kaynar"])

    def test_light_stemmer_keeps_roots(self):
        self.assertEqual(stem("boils"), "boil")
        self.assertEqual(stem("baskenti"), "baskent")
        self.assertEqual(stem("okyanus"), "okyanus")
        self.assertEqual(stem("okyanusu"), "okyanus")
        self.assertEqual(stem("sembolu"), "sembol")
        self.assertEqual(stem("gumusun"), "gumus")
        self.assertEqual(stem("altin"), "altin")

    def test_search_ranks_relevant_doc(self):
        hits = self.store.search("Mısır'ın başkenti neresi?")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc.title, "Kahire")
        self.assertGreaterEqual(hits[0].coverage, 0.5)

    def test_search_rejects_unrelated_query(self):
        self.assertEqual(self.store.search("merhaba nasilsin?"), [])
        self.assertEqual(self.store.search("bana bir saka yap"), [])

    def test_search_rejects_single_term_overlap(self):
        # Both terms are required by min_terms=2; a query with only one
        # matching content term must not inject a source.
        hits = self.store.search("Mısır'ın hava durumu")
        self.assertEqual(hits, [])

    def test_context_has_numbered_source_and_passage(self):
        ctx = self.store.context("Mısır'ın başkenti neresi?")
        self.assertIsNotNone(ctx)
        self.assertIn("[1] Kahire", ctx.system)
        self.assertIn("Mısır'ın başkenti", ctx.system)
        self.assertIn("example.invalid/kahire", ctx.sources[0]["url"])
        self.assertTrue(ctx.signature)

    def test_english_query_hits_english_doc(self):
        hits = self.store.search("At what temperature does water boil?")
        self.assertTrue(hits)
        self.assertEqual(hits[0].doc.title, "Water")

    def test_missing_corpus_is_unavailable(self):
        s = RagStore(Path(self.tmp.name) / "missing.jsonl")
        self.assertFalse(s.available)
        self.assertEqual(s.search("anything"), [])
        self.assertIsNone(s.context("anything"))

    def test_empty_corpus_is_available_false(self):
        p = Path(self.tmp.name) / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        s = RagStore(p)
        self.assertFalse(s.available)


class TestServiceRagGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lcsa.service import Service
        cls.svc = Service(budget_gb=5.0, log_requests=False, enable_chat=False)
        cls.svc.rag_enabled = True
        cls.tmp = tempfile.TemporaryDirectory()
        p = Path(cls.tmp.name) / "docs.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for d in DOCS:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        cls.svc.rag_store = RagStore(p, min_coverage=0.5, min_terms=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_factual_question_retrieves(self):
        ctx = self.svc._rag_for("Mısır'ın başkenti neresi?", "quality",
                                ["chat4b"], True)
        self.assertIsNotNone(ctx)

    def test_greeting_and_list_do_not_retrieve(self):
        self.assertIsNone(self.svc._rag_for("merhaba, nasilsin?", "quality",
                                            ["chat4b"], True))
        self.assertIsNone(self.svc._rag_for(
            "Mısır hakkında üç madde listele", "quality", ["chat4b"], True))

    def test_code_query_does_not_retrieve(self):
        self.assertIsNone(self.svc._rag_for(
            "Python kodundaki hatayı açıkla", "quality", ["chat4b"], True))

    def test_exact_output_and_creative_do_not_retrieve(self):
        self.assertIsNone(self.svc._rag_for(
            "Sadece 'mavi' yaz.", "quality", ["chat4b"], True))
        self.assertIsNone(self.svc._rag_for(
            "Bana kısa bir şaka yap.", "quality", ["chat4b"], True))

    def test_math_query_does_not_retrieve(self):
        self.assertIsNone(self.svc._rag_for("2 + 2 kaç eder?", "quality",
                                            ["chat4b"], True))

    def test_message_writing_does_not_retrieve(self):
        self.assertIsNone(self.svc._rag_for(
            "Bir arkadaşıma doğum günü mesajı yaz.", "quality",
            ["chat4b"], True))

    def test_explicit_off_disables(self):
        self.assertIsNone(self.svc._rag_for("Mısır'ın başkenti neresi?",
                                            "quality", ["chat4b"], False))


RANK_DOCS = [
    {
        "id": "tr:hollanda",
        "title": "Hollanda",
        "lang": "tr",
        "text": ("Kuzey Hollanda (Baskent: Haarlem) ve Guney Hollanda "
                 "(Baskent Lahey: Den Haag) iki ayri eyalettir. Bu bolge "
                 "hakkinda tarihsel bir aciklama daha vardir."),
        "source": "https://example.invalid/hollanda",
    },
    {
        "id": "tr:amsterdam",
        "title": "Amsterdam",
        "lang": "tr",
        "text": ("Amsterdam, Hollanda'nin baskenti ve en buyuk sehridir. "
                 "Sehir kanallari ve tarihi evleriyle bilinir."),
        "source": "https://example.invalid/amsterdam",
    },
    {
        "id": "tr:asya",
        "title": "Asya",
        "lang": "tr",
        "text": ("Asya, Dunya'nin en buyuk kitasidir. Nufus bakimindan da "
                 "en kalabalik kitadir. Sinirlari tarih boyunca degismistir."),
        "source": "https://example.invalid/asya",
    },
    {
        "id": "tr:antarktika",
        "title": "Antarktika",
        "lang": "tr",
        "text": ("Antarktika, Guney Kutbu'nda bulunan bir kitadir. "
                 "Kesifler gercekten buyuk bir kita bulundugunu gostermistir."),
        "source": "https://example.invalid/antarktika",
    },
    {
        "id": "tr:nil",
        "title": "Nil",
        "lang": "tr",
        "text": ("Nil Nehri dunyanin en uzun nehridir. Afrika'nin dogusunda "
                 "baslar ve Akdeniz'e dokulur."),
        "source": "https://example.invalid/nil",
    },
    {
        "id": "tr:amazon",
        "title": "Amazon Nehri",
        "lang": "tr",
        "text": ("Amazon Nehri, Afrika'daki Nil Nehri'nden sonra dunyanin "
                 "en uzun ikinci nehridir. Gunes Amerika'da yer alir."),
        "source": "https://example.invalid/amazon",
    },
    {
        "id": "tr:mavi-balina",
        "title": "Mavi balina",
        "lang": "tr",
        "text": ("Mavi balina, Dunya'nin yasayan en buyuk hayvani olarak "
                 "bilinir. Boyu otuz metreyi ve agirligi yuz tonu asabilir."),
        "source": "https://example.invalid/mavi-balina",
    },
    {
        "id": "tr:kopek",
        "title": "Kopek",
        "lang": "tr",
        "text": ("Kopekler, kedilerle birlikte dunyanin en genis cografyaya "
                 "yayilan ve en cok beslenen iki evcil hayvanindan biridir."),
        "source": "https://example.invalid/kopek",
    },
    {
        "id": "en:poland",
        "title": "Poland",
        "lang": "en",
        "text": ("The capital of Poland is Warsaw. The former capital was "
                 "Krakow. Poland is a country in Central Europe."),
        "source": "https://example.invalid/poland",
    },
]


class TestRagRanking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        p = Path(cls.tmp.name) / "rank_docs.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for d in RANK_DOCS:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        cls.store = RagStore(p, min_coverage=0.5, min_terms=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_precise_questions_select_one_source(self):
        for query in (
                "Hollanda'nin baskenti neresi?",
                "Dunyanin en buyuk kitasi hangisi?",
                "Dunyanin en uzun nehri hangisi?",
                "Dunyanin en buyuk hayvani hangisi?"):
            with self.subTest(query=query):
                ctx = self.store.context(query)
                self.assertIsNotNone(ctx)
                self.assertEqual(len(ctx.sources), 1, ctx.system)

    def test_capital_direct_phrase_beats_province_list(self):
        ctx = self.store.context("Hollanda'nin baskenti neresi?")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.sources[0]["title"], "Amsterdam")
        self.assertIn("Amsterdam", ctx.system)

    def test_superlative_target_and_ordinal_noise(self):
        asia = self.store.context("Dunyanin en buyuk kitasi hangisi?")
        nile = self.store.context("Dunyanin en uzun nehri hangisi?")
        whale = self.store.context("Dunyanin en buyuk hayvani hangisi?")
        self.assertIsNotNone(asia)
        self.assertIsNotNone(nile)
        self.assertIsNotNone(whale)
        self.assertEqual(asia.sources[0]["title"], "Asya")
        self.assertEqual(nile.sources[0]["title"], "Nil")
        self.assertEqual(whale.sources[0]["title"], "Mavi balina")

    def test_historical_capital_is_not_selected(self):
        ctx = self.store.context("What is the capital of Poland?")
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.sources[0]["title"], "Poland")
        self.assertIn("Warsaw", ctx.system)
        self.assertNotIn("Krakow", ctx.system)

    def test_broad_question_never_exceeds_two_sources(self):
        ctx = self.store.context(
            "Hollanda ve Polonya hakkinda genel bilgi ver")
        if ctx is not None:
            self.assertLessEqual(len(ctx.sources), 2)
            self.assertEqual(len({s["title"] for s in ctx.sources}),
                             len(ctx.sources))


if __name__ == "__main__":
    unittest.main()
