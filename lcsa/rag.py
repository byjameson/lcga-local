"""Small CPU-only retrieval layer for grounded local answers.

This is deliberately boring: a JSONL corpus, Unicode-aware tokenisation and
BM25. It needs no embedding model, no GPU memory and no server. Retrieval is
only used when a query has enough lexical overlap with a source; otherwise the
normal chat path is left alone.

The corpus is built from Wikipedia plain-text extracts by
``scripts/build_rag_corpus.py``. Each document keeps its source URL and the
retrieved text is quoted back to the model with a source number. The retrieval
layer does not claim to be a general search engine: a small corpus only covers
the pages it contains.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORPUS = "data/rag/world_facts.jsonl"
RAG_SYSTEM = (
    "Sen yerel bir bilgi asistanisin. Kullanicinin dilinde, asagidaki "
    "kaynaklari kullanarak sorusunu kisa ve dogru yanitla. Kaynaklarda "
    "olmayan bir bilgiyi uydurma. Cevabin sonunda kullandigin kaynak "
    "numarasini [1] seklinde belirt. Kaynaklarda yeterli bilgi yoksa "
    "'Bunu yerel kaynaklarda bulamadim' de."
)

TR_FOLD = str.maketrans(
    "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
    "cgiosuCGIOSU")

# Question words and function words are removed for retrieval only. Content
# words such as "capital", "water", "baskent" and "okyanus" are kept.
STOPWORDS = frozenset("""
a an and are as at be been being by can could did do does for from had has
have he her here his how i if in into is it its me my no not of on or our
she should so than that the their them there these they this those to was we
were what when where which who why will with would you your
bir bu o da de mi mu mı mü ne nedir neresi kac kac? hangi hangisi nasil neden
icin ile daha cok var yok olan olarak gibi ama fakat ancak ki ise
nin nun
eder eden etti etmek ediyor olur oldu olmus yapar yapmak
ben sen biz siz onlar benim senin bizim sizin onlarin
""".split())


def fold(text: str) -> str:
    """Case/diacritic-fold text without destroying Turkish word boundaries."""
    text = str(text or "").translate(TR_FOLD).casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def stem(token: str) -> str:
    """Very light, intentionally conservative suffix stripping.

    It is not a linguistic stemmer. It only removes common English plurals and
    Turkish plural/possessive/case endings so that "boils"/"boil" and
    "başkenti"/"başkent" can meet. Short roots are left alone to avoid
    collapsing unrelated words.
    """
    t = token
    if len(t) <= 3:
        return t
    # English plurals; keep -us/-is/-ss words (e.g. "okyanus", "analysis").
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    # Turkish endings, longest first. Require a root of at least 4 chars for
    # the short case endings (so "altin" does not turn into "alt").
    for suffix in (
            "lerinden", "larindan", "lerinin", "larinin", "lerine", "larina",
            "lerini", "larini", "lerin", "larin", "leri", "lari", "ler",
            "lar", "lerde", "larda", "lerden", "lardan",
            "den", "dan", "de", "da", "in", "ın", "un", "ün",
            "nin", "nın", "nun", "nün",
            "si", "sı", "i", "ı", "u", "ü"):
        if t.endswith(suffix):
            root = t[:-len(suffix)]
            need = 4 if suffix in ("in", "ın", "un", "ün", "i", "ı", "u",
                                   "ü", "de", "da", "si", "sı") else 3
            if len(root) >= need:
                return root
    return t


def _stem_fully(token: str) -> str:
    """Apply the light stemmer more than once for stacked Turkish suffixes."""
    for _ in range(4):
        shorter = stem(token)
        if shorter == token:
            break
        token = shorter
    return token


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    text = re.sub(r"[^a-z0-9]+", " ", fold(text))
    toks = text.split()
    if drop_stop:
        toks = [t for t in toks if t not in STOPWORDS and len(t) > 0]
    return [_stem_fully(t) for t in toks]


@dataclass(frozen=True)
class RagDoc:
    doc_id: str
    title: str
    text: str
    source: str
    lang: str


@dataclass(frozen=True)
class SearchHit:
    doc: RagDoc
    score: float
    coverage: float
    matched: tuple[str, ...]


@dataclass(frozen=True)
class RagContext:
    system: str
    sources: tuple[dict, ...]
    hits: tuple[SearchHit, ...]
    signature: str


def _passages(text: str) -> list[str]:
    """Split an article into sentence/paragraph-sized candidate passages."""
    text = re.sub(r"\r", " ", str(text or ""))
    raw = re.split(r"\n{2,}|(?<=[.!?])\s+", text)
    out = []
    for p in raw:
        p = " ".join(p.split())
        if 25 <= len(p):
            out.append(p)
    return out


_NUMBER_WORDS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
    r"iki|uc|dort|bes|alti|yedi|sekiz|dokuz|on)\b")
_ORDINAL_SUPERLATIVE = re.compile(
    r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th|"
    r"ikinci|ucuncu|dorduncu|besinci)\b[\s-]*(?:en\s+)?"
    r"(?:largest|biggest|buyuk)")


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text)) or bool(_NUMBER_WORDS.search(fold(text)))


def _has_degree(text: str) -> bool:
    return bool(re.search(
        r"(°\s*[cf]|\bdegrees?\b|\bderece\b|\bcelsius\b|\bfahrenheit\b|"
        r"\bsicaklik\b)", fold(text)))


def _has_boiling(text: str) -> bool:
    return bool(re.search(r"\b(boil\w*|kaynar\w*|kaynama\w*)", fold(text)))


def _has_speed_measurement(text: str) -> bool:
    return bool(re.search(
        r"\d[\d.,]*\s*(?:m/s|km/s|mph|km/h|m/sn|metre/saniye|"
        r"meter/second|kilometre/saat|kilometer/hour)", fold(text)))


def _ordinal_superlative(text: str) -> bool:
    return bool(_ORDINAL_SUPERLATIVE.search(fold(text)))


_SUPERLATIVE_TERMS = frozenset("""
largest biggest longest highest deepest buyuk uzun yuksek derin
""".split())
_TURKISH_SUPERLATIVES = frozenset({"buyuk", "uzun", "yuksek", "derin"})
_OTHER_SUPERLATIVES = frozenset({"genis", "kalabalik", "kucuk", "eski", "yeni"})
_ORDINAL_TERMS = frozenset("""
second third fourth fifth 2nd 3rd 4th 5th ikinci ucuncu dorduncu besinci
ikinc ucunc dordunc besinc birinci birinc
""".split())
_OPPOSITE_SUPERLATIVE = {
    "largest": ("smallest",),
    "biggest": ("smallest",),
    "buyuk": ("kucuk",),
    "longest": ("shortest",),
    "uzun": ("kisa",),
    "highest": ("lowest",),
    "yuksek": ("alcak",),
    "deepest": ("shallowest",),
    "derin": ("sig",),
}
_CAPITAL_TERMS = frozenset({"baskent", "capital"})
_SYMBOL_TERMS = frozenset({"sembol", "symbol", "simge"})
_ANSWER_TYPE_TERMS = frozenset("""
baskent capital sembol symbol simge kimyasal chemical
kac how many much old temperature derece sicaklik far long hangi yil
kaynar kaynama boil boiling odacik chamber vardir
""".split())
_HISTORICAL_CAPITAL = re.compile(
    r"\b(?:eski|onceki|former|old|ancient)\b(?:\s+\w+){0,4}\s+"
    r"(?:baskent\w*|capital\w*)\b|"
    r"\b(?:baskent\w*|capital\w*)\b(?:\s+\w+){0,4}\s+"
    r"(?:eski|onceki|former|old|ancient)\b")


@dataclass(frozen=True)
class _QueryProfile:
    folded: str
    terms: frozenset[str]
    subject_terms: frozenset[str]
    object_terms: frozenset[str]
    capital: bool
    symbol: bool
    numeric: bool
    temperature: bool
    boiling: bool
    year: bool
    numeral: str | None
    superlative: str | None
    ordinal: str | None


def _query_profile(query: str) -> _QueryProfile:
    qterms = frozenset(tokenize(query))
    qfold = fold(query)
    capital = bool(qterms & _CAPITAL_TERMS)
    symbol = bool(qterms & _SYMBOL_TERMS)
    year = any(c in qfold for c in (
        "hangi yil", "hangi yıl", "what year", "which year", "hangi yill",
        "hangi yıll", "when did", "ne zaman"))
    numeric = any(c in qfold for c in (
        "kac", "how many", "how much", "how old", "what temperature",
        "ne kadar", "derece", "temperature", "sicaklik", "how far",
        "how long", "hangi yil", "hangi yıl", "what year", "which year",
        "speed", "hiz", "hız", "velocity", "surat", "sürat")) or year
    temperature = any(c in qfold for c in (
        "derece", "temperature", "sicaklik", "°c", "°f"))
    boiling = any(c in qfold for c in ("boil", "kaynar", "kaynama"))
    numeral_match = re.search(r"(?<![a-z])([ivx]+)\.", qfold)
    numeral = numeral_match.group(1) if numeral_match else None
    superlative = next(
        (t for t in ("largest", "biggest", "longest", "highest", "deepest",
                     "buyuk", "uzun", "yuksek", "derin") if t in qterms),
        None)
    ordinal = next((t for t in _ORDINAL_TERMS if t in qterms), None)
    removed = (_ANSWER_TYPE_TERMS | _SUPERLATIVE_TERMS | _ORDINAL_TERMS |
               {"en"})
    subject = qterms - removed
    objects = subject - {"dunyan", "world", "earth", "dunya"}
    return _QueryProfile(
        folded=qfold,
        terms=qterms,
        subject_terms=subject,
        object_terms=objects,
        capital=capital,
        symbol=symbol,
        numeric=numeric,
        temperature=temperature,
        boiling=boiling,
        year=year,
        numeral=numeral,
        superlative=superlative,
        ordinal=ordinal,
    )


def _ordinal_near(text: str, wanted: str | None = None) -> bool:
    """Whether an ordinal is close to a superlative in *text*."""
    toks = tokenize(text, drop_stop=False)
    for i, token in enumerate(toks):
        if token not in _ORDINAL_TERMS:
            continue
        if wanted is not None and token != wanted:
            continue
        window = set(toks[max(0, i - 4):i + 5])
        if window & _SUPERLATIVE_TERMS:
            return True
    return False


def _opposite_superlative(text: str, profile: _QueryProfile) -> bool:
    """Whether a page says the opposite/different rank from the question."""
    if not profile.superlative:
        return False
    toks = set(tokenize(text))
    opposites = _OPPOSITE_SUPERLATIVE.get(profile.superlative, ())
    if not (toks & set(opposites)):
        return False
    return not profile.object_terms or bool(profile.object_terms & toks)


def _superlative_relation(text: str, profile: _QueryProfile) -> bool:
    """Whether the superlative is near the object being asked about."""
    if not profile.superlative:
        return False
    toks = tokenize(text, drop_stop=False)
    for i, token in enumerate(toks):
        if token != profile.superlative:
            continue
        if (profile.superlative in _TURKISH_SUPERLATIVES and
                (i == 0 or toks[i - 1] != "en")):
            continue
        window = set(toks[max(0, i - 7):i + 8])
        if profile.object_terms & window:
            return True
    return False


def _subject_capital_near(text: str, profile: _QueryProfile) -> bool:
    if not profile.subject_terms:
        return False
    toks = tokenize(text, drop_stop=False)
    for i, token in enumerate(toks):
        if token not in _CAPITAL_TERMS:
            continue
        window = set(toks[max(0, i - 6):i + 7])
        if profile.subject_terms & window:
            return True
    return False


def _capital_direct(text: str, profile: _QueryProfile) -> bool:
    """Whether the sentence has a subject -> capital / capital of subject link."""
    p = re.sub(r"[^a-z0-9]+", " ", fold(text))
    q = re.sub(r"[^a-z0-9]+", " ", profile.folded)
    qwords = q.split()
    if any(w.startswith("baskent") for w in qwords):
        for i in range(1, len(qwords)):
            if qwords[i] not in {"in", "nin", "un", "nun"}:
                continue
            subject = qwords[i - 1]
            if len(subject) < 3:
                continue
            pat = (r"\b" + re.escape(subject) + r"\w*\s+" + qwords[i] +
                   r"\b(?:\s+\w+){0,3}\s+(?:baskent|capital)\b")
            if re.search(pat, p):
                return True
    if "capital" in qwords:
        for subject in qwords:
            if subject in {"what", "which", "the", "is", "of", "capital",
                           "a", "an"} or len(subject) < 3:
                continue
            if re.search(r"\bcapital\b(?:\s+\w+){0,5}\s+of\s+" +
                         re.escape(subject) + r"\b", p):
                return True
            if re.search(r"\b" + re.escape(subject) +
                         r"\b(?:\s+\w+){0,4}\s+(?:is\s+)?(?:the\s+)?"
                         r"capital\b", p):
                return True
    return False


_COMPETING_OBJECTS = frozenset("""
okyanus deniz kita nehr gol dag hayvan bitki memeli gezegen uydu sehir
ulke bolge kitap film genis dar kalabalik
ocean sea river lake mountain animal species mammal planet moon city country
region continent wide narrow
""".split())


def _object_match(token: str, objects: frozenset[str]) -> bool:
    return any(token.startswith(o) or o.startswith(token) for o in objects)


def _superlative_target_match(text: str,
                               profile: _QueryProfile) -> bool | None:
    """Compare the noun after the superlative with the asked-about noun."""
    if not profile.superlative or not profile.object_terms:
        return None
    toks = tokenize(text, drop_stop=False)
    for i, token in enumerate(toks):
        if token != profile.superlative:
            continue
        if (profile.superlative in _TURKISH_SUPERLATIVES and
                (i == 0 or toks[i - 1] != "en")):
            continue
        for candidate in toks[i + 1:i + 6]:
            if (candidate in _SUPERLATIVE_TERMS or
                    candidate in _ORDINAL_TERMS or
                    candidate in STOPWORDS or candidate == "en"):
                continue
            if _object_match(candidate, profile.object_terms):
                return True
            if candidate in _COMPETING_OBJECTS:
                return False
            return None
    return None


def _title_numeral(title: str) -> str | None:
    m = re.match(r"\s*([ivx]+)\.", fold(title))
    return m.group(1) if m else None


def _is_specific_question(profile: _QueryProfile) -> bool:
    """Short fact questions need one precise source, not a noisy pair."""
    return bool(profile.capital or profile.symbol or profile.numeric or
                profile.temperature or profile.boiling or profile.superlative or
                profile.ordinal or
                (profile.subject_terms & {"odacik", "chamber"}))


class RagStore:
    """BM25 index over a local JSONL corpus."""

    def __init__(self, path: str | Path = DEFAULT_CORPUS,
                 min_coverage: float = 0.5, min_terms: int = 2,
                 top_k: int = 3):
        self.path = Path(path)
        self.min_coverage = float(min_coverage)
        self.min_terms = int(min_terms)
        self.top_k = int(top_k)
        self.docs: list[RagDoc] = []
        self._tf: list[Counter] = []
        self._doc_len: list[int] = []
        self._idf: dict[str, float] = {}
        self._avgdl = 1.0
        self._load()

    @property
    def available(self) -> bool:
        return bool(self.docs)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                title = str(row.get("title", "")).strip()
                text = str(row.get("text", "")).strip()
                source = str(row.get("source", "")).strip()
                lang = str(row.get("lang", "")).strip()
                if not title or len(text) < 80:
                    continue
                doc_id = str(row.get("id") or f"{lang}:{title}")
                self.docs.append(RagDoc(doc_id, title, text, source, lang))

        for d in self.docs:
            # Repeating title terms gives titles a small, useful boost.
            toks = tokenize(d.title) * 3 + tokenize(d.text)
            tf = Counter(toks)
            self._tf.append(tf)
            self._doc_len.append(max(1, len(toks)))
        n = len(self.docs)
        if not n:
            return
        self._avgdl = sum(self._doc_len) / n
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {
            t: math.log(1.0 + (n - c + 0.5) / (c + 0.5))
            for t, c in df.items()
        }

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        qterms = list(dict.fromkeys(tokenize(query)))
        if not qterms or not self.docs:
            return []
        k1, b = 1.5, 0.75
        qset = set(qterms)
        hits: list[SearchHit] = []
        for i, tf in enumerate(self._tf):
            matched = tuple(t for t in qterms if t in tf)
            if not matched:
                continue
            coverage = len(matched) / len(qterms)
            if coverage + 1e-9 < self.min_coverage:
                continue
            if len(matched) < min(self.min_terms, len(qterms)):
                continue
            dl = self._doc_len[i]
            score = 0.0
            for t in matched:
                freq = tf[t]
                score += self._idf.get(t, 0.0) * (
                    freq * (k1 + 1.0) /
                    (freq + k1 * (1.0 - b + b * dl / self._avgdl)))
            # A query word in the title is a much stronger subject signal than
            # another incidental mention in a long article. This is especially
            # important for short titles such as "Su" or "Fransa".
            title_terms = set(tokenize(self.docs[i].title))
            title_hits = qset & title_terms
            if title_hits:
                score += 8.0 * len(title_hits)
            hits.append(SearchHit(self.docs[i], score, coverage, matched))
        hits.sort(key=lambda h: (h.score, h.coverage), reverse=True)
        return hits[:int(top_k or self.top_k)]

    def context(self, query: str) -> RagContext | None:
        # Search a wider pool than the final sources: a long, generic page can
        # win BM25 while the answer-bearing passage is a little lower.
        hits = self.search(query, top_k=12)
        if not hits:
            return None
        profile = _query_profile(query)
        qterms = set(profile.terms)
        candidates: list[tuple[float, int, SearchHit, str]] = []
        for hit in hits:
            passages = _passages(hit.doc.text)
            if not passages:
                continue
            title_terms = set(tokenize(hit.doc.title))
            title_hits = qterms & title_terms
            lead_text = " ".join(hit.doc.text.split())[:1200]
            lead_tokens = set(tokenize(lead_text))
            best = None
            best_overlap = -1
            best_score = -1.0
            for idx, p in enumerate(passages):
                pt = set(tokenize(p))
                ov = len(qterms & pt)
                # Prefer passages with query terms, then comprehensive ones.
                s = ov * 4.0 + min(len(p), 300) / 2000.0
                s += max(0.0, 100 - idx) * 0.02
                if idx == 0:
                    s += 0.3
                if title_hits:
                    s += 3.0 * len(title_hits)
                if profile.numeral:
                    title_numeral = _title_numeral(hit.doc.title)
                    if title_numeral == profile.numeral:
                        s += 8.0
                        if title_hits <= pt:
                            s += 6.0
                    elif title_numeral:
                        s -= 8.0

                # The question type should decide which sentence in a page is
                # the answer, not merely which page has the most overlapping
                # words. These cues are deliberately small and local.
                if profile.capital and (_CAPITAL_TERMS & pt):
                    s += 3.0
                    if _HISTORICAL_CAPITAL.search(fold(p)):
                        s -= 7.0
                    if _subject_capital_near(p, profile):
                        s += 4.0
                    if _capital_direct(p, profile):
                        s += 8.0
                    if (fold(p).count("baskent") >= 2 or
                            re.search(r"baskent\s*:", fold(p))):
                        s -= 4.0
                if profile.symbol and (_SYMBOL_TERMS & pt):
                    s += 3.0
                    if qterms & {"kimyasal", "chemical"}:
                        if pt & {"kimyasal", "chemical"}:
                            s += 2.0
                if profile.numeric and _has_number(p):
                    if profile.temperature:
                        if _has_degree(p):
                            s += 8.0
                    elif profile.year:
                        s += 12.0
                    else:
                        s += 2.5
                if profile.numeric and _has_speed_measurement(p):
                    s += 4.5
                if profile.boiling and _has_boiling(p):
                    s += 4.0
                type_hits = _ANSWER_TYPE_TERMS & pt
                if type_hits:
                    s += 3.0 * len(type_hits)
                if profile.subject_terms:
                    subject_hits = profile.subject_terms & pt
                    if subject_hits:
                        s += 1.5 * len(subject_hits)
                    elif not (profile.subject_terms & title_hits):
                        # A number without the queried subject is usually a
                        # different fact (e.g. lead's boiling point for water).
                        if _is_specific_question(profile):
                            s -= 4.0

                if profile.superlative:
                    passage_folded = fold(p)
                    if (_OTHER_SUPERLATIVES & pt and
                            not _superlative_relation(p, profile)):
                        s -= 4.0
                    if _opposite_superlative(passage_folded, profile):
                        s -= 5.0
                    if _opposite_superlative(lead_text, profile):
                        s -= 6.0
                    if _ordinal_near(p) and not profile.ordinal:
                        s -= 10.0
                    if profile.ordinal:
                        if _ordinal_near(p, profile.ordinal):
                            s += 3.0
                        elif _ordinal_near(p):
                            s -= 2.0
                        if _ordinal_near(lead_text, profile.ordinal):
                            s += 2.0
                    if _superlative_relation(p, profile):
                        s += 3.0
                    if _superlative_relation(lead_text, profile):
                        s += 3.0
                    target = _superlative_target_match(p, profile)
                    if target is True:
                        s += 4.0
                    elif target is False:
                        s -= 4.0
                    lead_target = _superlative_target_match(lead_text, profile)
                    if lead_target is True:
                        s += 2.0
                    elif lead_target is False:
                        s -= 3.0
                    if (profile.object_terms & title_terms and
                            profile.superlative in lead_tokens):
                        s += 2.0

                if s > best_score:
                    best, best_overlap, best_score = p, ov, s
            if best is None:
                continue
            # Keep the prompt within the 512-token engine budget.
            best = best[:520].strip()
            if best:
                candidates.append((best_score, best_overlap, hit, best))
        # Passage score already contains overlap and answer-type cues. Keeping
        # it first avoids a generic passage with two incidental query words
        # outranking the title page's actual answer sentence.
        candidates.sort(key=lambda x: (x[0], x[1], x[2].score), reverse=True)
        if not candidates:
            return None

        # A precise short-answer question rarely needs a second page. Adding
        # one was the main source of unrelated context (for example, a
        # potassium page next to a gold question). Broad questions may still
        # use a second source when it independently repeats the subject.
        chosen = [(candidates[0][2], candidates[0][3])]
        if not _is_specific_question(profile):
            top_score = candidates[0][0]
            for score, _ov, hit, passage in candidates[1:]:
                if score < top_score - 2.0:
                    break
                if profile.subject_terms and not (
                        profile.subject_terms & set(tokenize(passage)) or
                        profile.subject_terms & set(tokenize(hit.doc.title))):
                    continue
                if (profile.superlative and
                        _opposite_superlative(fold(passage), profile)):
                    continue
                chosen.append((hit, passage))
                break

        sources = []
        blocks = []
        for n, (hit, passage) in enumerate(chosen, 1):
            src = {
                "id": n,
                "title": hit.doc.title,
                "url": hit.doc.source,
                "lang": hit.doc.lang,
                "score": round(hit.score, 3),
                "coverage": round(hit.coverage, 3),
            }
            sources.append(src)
            blocks.append(f"[{n}] {hit.doc.title}\n{passage}")
        system = RAG_SYSTEM + "\n\nKAYNAKLAR:\n" + "\n\n".join(blocks)
        sig = ",".join(h.doc.doc_id for h, _ in chosen)
        return RagContext(system, tuple(sources), tuple(hits), sig)
