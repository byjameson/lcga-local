"""Query-level serving service: route -> plan -> execute -> merge.

This is the product shell around the paging/pool/scheduler work. It accepts a
user query, picks which model units should answer it, packs the units into
VRAM-sized groups (parallel when they fit, sequential otherwise), runs them,
and merges the outputs.

Units are heterogeneous:

  fast      Qwen2.5-0.5B   dense bf16, ~1.1GB
  balanced  Qwen2.5-1.5B   dense bf16, ~3.3GB
  deep      Qwen2.5-3B     int8 paged, ~4.7GB peak (does not fit VRAM whole)
  chat      Qwen2.5-1.5B-Instruct  dense bf16, ~3.3GB (if the model is present)
  chat3b    Qwen2.5-3B-Instruct    int8 paged, ~4.7GB
  chat4b    Qwen3-4B-Instruct-2507 int8 GPU-resident, ~4.9GB (best quality, if present)

Honest limits: the router is rule-based (keywords/length), not learned; the
VRAM estimates are declared, not measured per request; requests are serialized
behind one GPU lock (single 6GB card).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

from lcsa.pool import ModelPool
from lcsa.rag import DEFAULT_CORPUS, RagStore, tokenize
from lcsa.scheduler import Unit, merge_outputs, plan

ROOT = Path(__file__).resolve().parents[1]

DEEP_CUES = (
    "kod", "code", "python", "algoritma", "algorithm", "debug", "hata",
    "adım adım", "step by step", "kanıt", "proof", "türet", "derive",
    "neden", "why", "karşılaştır", "compare", "optimize", "matematik",
    "hesapla", "calculate", "mantık", "reason", "açıkla", "explain",
)
GREETING_CUES = ("merhaba", "selam", "hello", "hi ", "hey", "nasılsın")
QUESTION_CUES = ("?", "nedir", "nasıl", "neden", "kim", "kaç", "neresi",
                 "what", "why", "how", "who", "when", "where", "which",
                 "anlat", "açıkla", "yaz", "listele", "özetle")
# Quality mode is deliberately simple. Turkish list requests go to the 3B
# model with an explicit numbered-line instruction: its list content measured
# better, and the instruction fixes its usual inline formatting. English list
# requests stay on the 1.5B model, where a cheap deterministic formatter can
# turn an inline answer into three numbered lines without changing its content.
QUALITY_LIST_CUES = ("listele", "list three", "list ")
QUALITY_CHAT_CUES = ("tesekkur", "teşekkür")
LIST_FORMAT_INSTRUCTION = (
    "\n\nCevabi tam olarak 3 satir halinde ver. Her satir '1. ', '2. ', "
    "'3. ' ile baslasin. Baska aciklama ekleme.")
CHAT_MODEL = "models/Qwen2.5-1.5B-Instruct"
CHAT3B_MODEL = "models/Qwen2.5-3B-Instruct"
CHAT4B_MODEL = "models/Qwen3-4B-Instruct-2507"
# The 4B int8 payload is deterministic; cache it so only the first process
# start pays the ~17s quantization cost. Set LCGA_NO_QUANT_CACHE=1 to opt out.
CHAT4B_QUANT_CACHE = "cache/qwen3_4b_instruct_2507_int8_resident.pt"


@dataclass(frozen=True)
class UnitSpec:
    key: str
    model_path: str
    est_gb: float
    kind: str          # "dense" | "paged"
    label: str
    description: str


UNIT_SPECS = (
    UnitSpec("fast", "models/Qwen2.5-0.5B", 1.1, "dense",
             "Qwen2.5-0.5B", "hizli, kisa/genel sorgular"),
    UnitSpec("balanced", "models/Qwen2.5-1.5B", 3.3, "dense",
             "Qwen2.5-1.5B", "dengeli kalite, orta uzunluk"),
    UnitSpec("deep", "models/Qwen2.5-3B", 4.7, "paged",
             "Qwen2.5-3B (int8 paged)", "en yuksek kalite; VRAM'e sigmaz, paging"),
)

CHAT_SPEC = UnitSpec("chat", CHAT_MODEL, 3.3, "dense",
                     "Qwen2.5-1.5B-Instruct", "instruct sohbet; soru-cevap")
CHAT3B_SPEC = UnitSpec("chat3b", CHAT3B_MODEL, 4.7, "paged",
                       "Qwen2.5-3B-Instruct (int8 paged)",
                       "yavas ama daha kaliteli sohbet; 6GB'de paged")
# Qwen3-4B-Instruct-2507 is the strongest model that fits the 6GB budget with
# int8 paging (~4.8GB peak). It is the default quality unit when present.
CHAT4B_SPEC = UnitSpec("chat4b", CHAT4B_MODEL, 4.9, "paged",
                       "Qwen3-4B-Instruct-2507 (int8, GPU)",
                       "en yuksek kalite; 4B int8 GPU-resident, 6GB'de")
# Optional non-Qwen dense model. Its purpose is to prove that the serving
# layer is model-agnostic; it is never selected by `auto` or `quality`.
ALT_MODEL = "models/SmolLM2-1.7B-Instruct"
ALT_SPEC = UnitSpec("alt", ALT_MODEL, 3.4, "dense",
                    "SmolLM2-1.7B-Instruct",
                    "Qwen disi adapter testi; deneysel, kalite iddiasi degil")

CHAT_SYSTEM = ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve "
               "dogru cevap ver.")
MAX_HISTORY = 8          # messages kept, including the current user turn
MAX_MESSAGE_CHARS = 2000  # per-message guard against oversized contexts


def chat_available() -> bool:
    """True only when the instruct model looks fully downloaded (~3.1GB)."""
    d = ROOT / CHAT_MODEL
    if not (d / "config.json").exists() or not (d / "tokenizer.json").exists():
        return False
    total = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    return total > 2_500_000_000


def chat3b_available() -> bool:
    """True only when Qwen2.5-3B-Instruct is fully downloaded (~6.2GB)."""
    d = ROOT / CHAT3B_MODEL
    if not (d / "config.json").exists() or not (d / "tokenizer.json").exists():
        return False
    total = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    return total > 6_000_000_000


def chat4b_available() -> bool:
    """True only when Qwen3-4B-Instruct-2507 is fully downloaded (~8.0GB)."""
    d = ROOT / CHAT4B_MODEL
    if not (d / "config.json").exists() or not (d / "tokenizer.json").exists():
        return False
    total = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    return total > 7_400_000_000


def alt_available() -> bool:
    """True when the optional non-Qwen instruct model is complete (~3.4GB)."""
    d = ROOT / ALT_MODEL
    if not (d / "config.json").exists() or not (d / "tokenizer.json").exists():
        return False
    total = sum(p.stat().st_size for p in d.glob("*.safetensors"))
    return total > 3_000_000_000


def list_kind(query: str) -> str | None:
    """Return 'tr' for Turkish list requests, 'other' for English/other cues."""
    q = " ".join(query.strip().lower().split())
    if "listele" in q:
        return "tr"
    if any(c in q for c in QUALITY_LIST_CUES):
        return "other"
    return None


def repair_list_text(text: str) -> str:
    """Make a short inline list into three numbered lines without rewriting it.

    This is intentionally conservative: already-numbered answers are returned
    unchanged, and no content is invented. It only helps the measured English
    list prompts whose chat answer was correct but inline.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    numbered = sum(
        1 for x in lines
        if re.match(r"^\s*(?:[0-9]+[.)]|[-*])\s+\S", x))
    if numbered >= 3:
        return "\n".join(lines)
    joined = " ".join(lines)
    joined = re.sub(r"\s+(?:are|is)\s+the\s+.*$", "", joined,
                    flags=re.IGNORECASE)
    joined = re.sub(r"^(?:here are|iste|işte)\s+.*?:", "", joined,
                    flags=re.IGNORECASE)
    parts = re.split(r"\s*(?:,|;|\n|\band\b|\bve\b)\s*", joined)
    parts = [re.sub(r"^\s*[0-9]+[.)]\s*", "", p).strip(" .;:")
             for p in parts]
    parts = [p for p in parts if p and p.lower() not in ("and", "ve")]
    if len(parts) < 3:
        return raw
    return "\n".join(f"{i}. {p}" for i, p in enumerate(parts[:3], 1))


def route(query: str, mode: str = "auto",
          chat: bool = False, chat3b: bool = False,
          alt: bool = False, chat4b: bool = False) -> tuple[list[str], str]:
    """Pick unit keys for a query. Returns (keys, human-readable reason)."""
    q = " ".join(query.strip().lower().split())
    if mode in ("fast", "balanced", "deep", "chat", "chat3b", "chat4b", "alt"):
        if mode == "chat" and not chat:
            raise ValueError("chat unit is not available")
        if mode == "chat3b" and not chat3b:
            raise ValueError("chat3b unit is not available (3B instruct model missing)")
        if mode == "chat4b" and not chat4b:
            raise ValueError("chat4b unit is not available (Qwen3-4B model missing)")
        if mode == "alt" and not alt:
            raise ValueError("alt unit is not available (SmolLM model missing)")
        return [mode], f"mode={mode} (kullanici secti)"
    if mode == "ensemble":
        return ["fast", "balanced"], "mode=ensemble: 0.5B + 1.5B birlikte"
    if mode == "quality":
        if chat4b:
            hits = [c for c in DEEP_CUES if c in q]
            if hits:
                return ["chat4b"], ("quality: kod/akil yurutme (" +
                                    ", ".join(hits[:3]) + ") -> 4B Instruct")
            if list_kind(q):
                return ["chat4b"], "quality: liste -> 4B Instruct"
            if chat and any(c in q for c in QUALITY_CHAT_CUES):
                return ["chat"], "quality: kisa nezaket cumlesi -> 1.5B Instruct"
            return ["chat4b"], "quality: genel -> 4B Instruct"
        if not chat3b:
            if chat:
                return ["chat"], "quality: 3B yok, 1.5B Instruct'a dusuldu"
            return ["fast"], "quality: instruct yok, 0.5B'ye dusuldu"
        hits = [c for c in DEEP_CUES if c in q]
        if hits:
            return ["deep"], ("quality: kod/akil yurutme (" +
                              ", ".join(hits[:3]) + ") -> 3B paged")
        kind = list_kind(q)
        if kind == "tr":
            return ["chat3b"], "quality: TR liste -> 3B + numarali format"
        if kind == "other" and chat:
            return ["chat"], "quality: EN liste -> 1.5B + format duzeltme"
        if kind == "other":
            return ["chat3b"], "quality: liste, 1.5B yok -> 3B Instruct"
        if chat and any(c in q for c in QUALITY_CHAT_CUES):
            return ["chat"], "quality: kisa nezaket cumlesi -> 1.5B Instruct"
        return ["chat3b"], "quality: genel -> 3B Instruct"
    if mode != "auto":
        raise ValueError(f"unknown mode: {mode}")

    if any(c in q for c in GREETING_CUES) and len(q) < 40:
        if chat:
            return ["chat"], "auto: kisa selamlama -> 1.5B Instruct"
        return ["fast"], "auto: kisa selamlama -> yalniz 0.5B"
    hits = [c for c in DEEP_CUES if c in q]
    if hits:
        return ["deep"], ("auto: karmasik sorgu (" + ", ".join(hits[:3]) +
                          ") -> 3B int8 paged")
    if chat and (len(q) < 40 or any(c in q for c in QUESTION_CUES)):
        return ["chat"], "auto: soru/sohbet -> 1.5B Instruct"
    if len(q) < 40:
        return ["fast"], "auto: kisa sorgu -> yalniz 0.5B"
    return (["fast", "balanced"],
            "auto: genel sorgu -> 0.5B + 1.5B paralel, en guvenilir cevap secilir")


class Service:
    def __init__(self, budget_gb: float = 5.0, device: str = "cuda",
                 verbose: bool = False, deep_ckpt: str | None = None,
                 log_requests: bool = True, cache_size: int = 32,
                 enable_chat: bool = True, rag: bool = False,
                 rag_path: str | None = None):
        self.budget_gb = float(budget_gb)
        self.device = device
        self.verbose = verbose
        self.pool = ModelPool(self.budget_gb, device=device, verbose=verbose)
        self._lock = threading.Lock()
        self._paged: dict = {}
        self._tok: dict = {}
        self.specs = {s.key: s for s in UNIT_SPECS}
        self.chat_model = None
        self._chat_tok = None
        if enable_chat and chat_available():
            from transformers import AutoTokenizer
            try:
                tok = AutoTokenizer.from_pretrained(
                    str(ROOT / CHAT_MODEL), trust_remote_code=True)
            except Exception:
                tok = None
            if tok is not None:
                self.chat_model = str(ROOT / CHAT_MODEL)
                self._chat_tok = tok
                self.specs["chat"] = CHAT_SPEC
        self.chat3b_model = None
        if enable_chat and chat3b_available():
            self.chat3b_model = str(ROOT / CHAT3B_MODEL)
            self.specs["chat3b"] = CHAT3B_SPEC
        self.chat4b_model = None
        if enable_chat and chat4b_available():
            self.chat4b_model = str(ROOT / CHAT4B_MODEL)
            self.specs["chat4b"] = CHAT4B_SPEC
        self.alt_model = None
        if enable_chat and alt_available():
            from transformers import AutoTokenizer
            try:
                tok = AutoTokenizer.from_pretrained(
                    str(ROOT / ALT_MODEL), trust_remote_code=True)
            except Exception:
                tok = None
            if tok is not None:
                self.alt_model = str(ROOT / ALT_MODEL)
                self._tok[self.alt_model] = tok
                self.specs["alt"] = ALT_SPEC
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.deep_ckpt = None
        if deep_ckpt:
            p = Path(deep_ckpt)
            p = p if p.is_absolute() else ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"deep checkpoint not found: {p}")
            self.deep_ckpt = str(p)
        self.log_path = (ROOT / "results" / "service_requests.jsonl"
                         if log_requests else None)
        self.counters = {"requests": 0, "errors": 0, "paged_loads": 0,
                         "cache_hits": 0, "cache_misses": 0,
                         "total_s": 0.0, "gen_s": 0.0, "queue_s": 0.0}
        # Retrieval is opt-in for the service/API. The product CLI enables it;
        # tests and HTTP clients keep the old behaviour unless they ask for it.
        self.rag_enabled = bool(rag)
        self.rag_path = (ROOT / (rag_path or DEFAULT_CORPUS))
        self.rag_store = None
        if self.rag_enabled:
            try:
                self.rag_store = RagStore(self.rag_path)
            except (OSError, ValueError):
                self.rag_store = None

    # ------------------------------------------------------------- model caches
    def _abs(self, p: str) -> str:
        path = Path(p)
        return str(path if path.is_absolute() else ROOT / path)

    def _get_tok(self, model_path: str):
        tok = self._tok.get(model_path)
        if tok is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._tok[model_path] = tok
        return tok

    def _get_paged(self, model_path: str):
        from lcsa.paged_qwen import PagedQwen
        eng = self._paged.get(model_path)
        if eng is None:
            sd = None
            is_deep = (Path(model_path).resolve() ==
                       (ROOT / self.specs["deep"].model_path).resolve())
            if self.deep_ckpt and is_deep:
                t0 = time.perf_counter()
                sd = torch.load(self.deep_ckpt, map_location="cpu", mmap=True,
                                weights_only=True)
                if self.verbose:
                    print(f"[service] deep checkpoint {Path(self.deep_ckpt).name} "
                          f"({len(sd)} tensors, mmap, "
                          f"{time.perf_counter()-t0:.2f}s)", flush=True)
            t0 = time.perf_counter()
            is_chat4b = ("chat4b" in self.specs and
                         Path(model_path).resolve() ==
                         (ROOT / self.specs["chat4b"].model_path).resolve())
            quant_cache = None
            if is_chat4b and os.environ.get("LCGA_NO_QUANT_CACHE", "") not in ("1", "true", "TRUE"):
                quant_cache = ROOT / CHAT4B_QUANT_CACHE
            eng = PagedQwen(model_path, device=self.device, dtype=torch.bfloat16,
                            resident_k=None, max_seq=512, pin=False, quant="int8",
                            state_dict=sd, resident_quant=is_chat4b,
                            quant_cache=quant_cache, verbose=False)
            del sd
            self._paged[model_path] = eng
            self.counters["paged_loads"] += 1
            if self.verbose:
                ck = f" ckpt={Path(self.deep_ckpt).name}" if self.deep_ckpt and is_deep else ""
                print(f"[service] paged load {model_path}{ck} "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)
        return eng

    # ------------------------------------------------------------------- route
    def units_info(self) -> list[dict]:
        out = []
        for s in self.specs.values():
            d = {"key": s.key, "model_path": s.model_path, "est_gb": s.est_gb,
                 "kind": s.kind, "label": s.label, "description": s.description}
            if s.key == "deep" and self.deep_ckpt:
                d["checkpoint"] = Path(self.deep_ckpt).name
            out.append(d)
        return out

    # ----------------------------------------------------------------- execute
    def preload(self, keys: list[str]) -> dict:
        """Load units now so the first real request skips the cold start."""
        keys = list(dict.fromkeys(keys))
        bad = [k for k in keys if k not in self.specs]
        if bad:
            raise ValueError(f"unknown unit(s): {bad}")
        if not keys:
            raise ValueError("units is empty")
        t0 = time.perf_counter()
        with self._lock:
            for k in keys:
                spec = self.specs[k]
                path = self._abs(spec.model_path)
                if spec.kind == "dense":
                    self.pool.get(path, spec.est_gb)
                else:
                    self._release_other_paged(path)
                    self._get_paged(path)
            return {
                "loaded": keys,
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "pool": self.pool.stats(),
                "paged": [Path(p).name for p in self._paged],
                "free_vram_gb": (round(torch.cuda.mem_get_info()[0] / 2**30, 2)
                                 if torch.cuda.is_available() else None),
            }

    def _rag_for(self, query: str, mode: str, keys: list[str],
                 use_rag: bool):
        """Return a retrieved context only for a plausible factual query.

        The lexical retriever already returns nothing for greetings and most
        instructions. This extra gate keeps list/code requests on the model's
        normal path instead of forcing unrelated encyclopedia text into them.
        """
        if not use_rag or self.rag_store is None or not self.rag_store.available:
            return None
        if mode not in ("quality", "chat4b", "chat3b", "chat"):
            return None
        if not any(k in ("chat", "chat3b", "chat4b") for k in keys):
            return None
        q = " ".join(query.strip().lower().split())
        if len(q) < 8 or list_kind(q):
            return None
        # Arithmetic/single-term requests are not encyclopedia lookups.
        if re.fullmatch(r"[\d\s+\-*/().,=]+", q):
            return None
        if len(set(tokenize(query))) < 2:
            return None
        if any(c in q for c in GREETING_CUES) and len(q) < 40:
            return None
        if any(c in q for c in ("kod", "code", "python", "algoritma",
                                "algorithm", "debug", "hata ayikla")):
            return None
        # Creative and exact-output requests do not benefit from forced
        # encyclopedia context; keep them on the normal model path.
        if any(c in q for c in ("saka", "şaka", "hikaye", "hikâye", "siir",
                                "şiir", "masal", "joke", "story", "poem",
                                "motivasyon", "mesaj", "doğum günü",
                                "dogum gunu", "tebrik", "kutlama",
                                "e-posta", "eposta", "email", "taslak",
                                "dilekçe", "dilekce")):
            return None
        if (re.search(r"\b(sadece|yalniz|yalnız|only|exactly)\b", q)
                and re.search(r"\b(yaz|write|soyle|söyle|cevapla)\b", q)):
            return None
        return self.rag_store.context(query)

    def _rag_payload(self, ctx) -> dict:
        available = bool(self.rag_store is not None and self.rag_store.available)
        if ctx is None:
            return {"enabled": bool(self.rag_enabled), "available": available,
                    "used": False, "sources": []}
        return {
            "enabled": True,
            "available": available,
            "used": True,
            "sources": [dict(s) for s in ctx.sources],
            "signature": ctx.signature,
        }

    def _chat_prompt(self, query: str,
                     messages: list[dict] | None = None,
                     model_path: str | None = None,
                     extra_system: str | None = None) -> str:
        system_parts = []
        convo = []
        for m in (messages or []):
            if m.get("role") == "system":
                system_parts.append(str(m.get("content", "")).strip())
            else:
                convo.append(m)
        if extra_system:
            system_parts.append(str(extra_system).strip())
        system = " ".join(x for x in system_parts if x) or CHAT_SYSTEM
        msgs = [{"role": "system", "content": system}] + convo
        if not convo:
            msgs.append({"role": "user", "content": query})
        tok = self._chat_tok if model_path is None else self._get_tok(model_path)
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        # Conservative fallback for a tokenizer without a chat template. It is
        # intentionally simple; a model-specific template is always preferred.
        lines = [f"System: {system}"]
        for m in msgs[1:]:
            role = "User" if m.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {m.get('content', '')}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _prompt_for(self, key: str, query: str,
                    history: list[dict] | None,
                    list_format: bool = False,
                    extra_system: str | None = None) -> str:
        if list_format and key == "chat3b":
            query = query.rstrip() + LIST_FORMAT_INSTRUCTION
        if key in ("chat", "alt", "chat4b"):
            path = None if key == "chat" else self._abs(self.specs[key].model_path)
            return self._chat_prompt(query, history, path, extra_system)
        if key == "chat3b":
            return self._chat_prompt(
                query, history, self._abs(self.specs["chat3b"].model_path),
                extra_system)
        return query

    @staticmethod
    def _fit_prompt(eng, ids, max_new_tokens: int):
        """Keep the most recent tokens so prompt + generation fits max_seq."""
        limit = max(1, int(eng.max_seq) - int(max_new_tokens) - 1)
        if ids.shape[1] > limit:
            ids = ids[:, -limit:]
        return ids

    @staticmethod
    def _clean_messages(messages) -> list[dict]:
        """Validate a chat history; keep the last MAX_HISTORY messages.

        System messages are accepted (OpenAI clients commonly send one) and are
        merged into the fixed system prompt by _chat_prompt. The final message
        must still be from the user.
        """
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        if not messages:
            raise ValueError("messages is empty")
        out: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                raise ValueError("each message must be an object")
            role = str(m.get("role", "")).strip()
            content = m.get("content")
            if role not in ("system", "user", "assistant"):
                raise ValueError("message role must be system, user or assistant")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("message content must be a non-empty string")
            out.append({"role": role,
                        "content": content.strip()[:MAX_MESSAGE_CHARS]})
        out = out[-MAX_HISTORY:]
        if out[-1]["role"] != "user":
            raise ValueError("last message must be from the user")
        return out

    def _release_other_paged(self, keep_path: str) -> None:
        """Drop other paged engines before loading a different 3B model."""
        keep = Path(keep_path).resolve()
        for p, eng in list(self._paged.items()):
            if Path(p).resolve() == keep:
                continue
            try:
                eng.release_gpu()
            except Exception:
                pass
            self._paged.pop(p, None)

    def _run_paged(self, u: Unit) -> None:
        self._release_other_paged(u.model_path)
        t_load = time.perf_counter()
        eng = self._get_paged(u.model_path)
        u.load_s = time.perf_counter() - t_load
        tok = self._get_tok(u.model_path)
        ids = tok(u.prompt, return_tensors="pt").input_ids
        ids = self._fit_prompt(eng, ids, u.max_new_tokens)
        t0 = time.perf_counter()
        out, _stats = eng.generate(ids, max_new_tokens=u.max_new_tokens,
                                   temperature=0.0, seed=1234,
                                   eos_id=tok.eos_token_id)
        u.gen_s = time.perf_counter() - t0
        u.text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        u.score = None  # paged engine does not expose token logprobs

    def answer(self, query: str, mode: str = "auto", max_new_tokens: int = 24,
               merge: str = "best", budget_gb: float | None = None,
               units: list[str] | None = None,
               messages: list[dict] | None = None,
               rag: bool | None = None) -> dict:
        t_all = time.perf_counter()
        history = self._clean_messages(messages) if messages is not None else None
        if history is not None:
            if units:
                raise ValueError("messages and units are mutually exclusive")
            if mode not in ("auto", "chat", "chat3b", "chat4b", "alt", "quality"):
                raise ValueError(
                    "messages are only supported with mode auto/chat/chat3b/chat4b/alt/quality")
            if mode == "chat3b":
                if "chat3b" not in self.specs:
                    raise ValueError("chat3b unit is not available (3B instruct model missing)")
                keys = ["chat3b"]
                reason = f"chat3b: {len(history)} mesajlik konusma (3B paged)"
            elif mode == "chat4b":
                if "chat4b" not in self.specs:
                    raise ValueError("chat4b unit is not available (Qwen3-4B model missing)")
                keys = ["chat4b"]
                reason = f"chat4b: {len(history)} mesajlik konusma (4B GPU-int8)"
            elif mode == "alt":
                if "alt" not in self.specs:
                    raise ValueError("alt unit is not available (SmolLM model missing)")
                keys = ["alt"]
                reason = f"alt: {len(history)} mesajlik konusma (Qwen disi model)"
            elif mode == "quality":
                keys, reason = route(
                    history[-1]["content"], "quality",
                    chat="chat" in self.specs,
                    chat3b="chat3b" in self.specs,
                    chat4b="chat4b" in self.specs,
                    alt="alt" in self.specs)
                if keys == ["deep"]:
                    fallback = ("chat4b" if "chat4b" in self.specs else
                                ("chat3b" if "chat3b" in self.specs else "chat"))
                    keys = [fallback]
                    reason += f" (mesajda {fallback} Instruct'a dusuldu)"
                if not any(k in self.specs for k in ("chat", "chat3b", "chat4b")):
                    raise ValueError("no chat unit available for quality mode")
            else:
                if "chat" not in self.specs:
                    raise ValueError("chat unit is not available (model missing or --no-chat)")
                keys = ["chat"]
                reason = f"chat: {len(history)} mesajlik konusma"
            query = history[-1]["content"]
        elif units:
            keys = list(dict.fromkeys(units))
            bad = [k for k in keys if k not in self.specs]
            if bad:
                raise ValueError(f"unknown unit(s): {bad}")
            reason = "kullanici birimleri secti: " + ", ".join(keys)
        else:
            keys, reason = route(query, mode, chat="chat" in self.specs,
                                 chat3b="chat3b" in self.specs,
                                 chat4b="chat4b" in self.specs,
                                 alt="alt" in self.specs)
        budget = float(self.budget_gb if budget_gb is None else budget_gb)
        use_rag = self.rag_enabled if rag is None else bool(rag)
        rag_ctx = self._rag_for(query, mode, keys, use_rag)
        extra_system = rag_ctx.system if rag_ctx is not None else None
        if rag_ctx is not None:
            reason += f" + yerel RAG ({len(rag_ctx.sources)} kaynak)"
        cache_key = self._cache_key(query, mode, max_new_tokens, merge, budget,
                                    keys, history,
                                    rag_sig=(rag_ctx.signature
                                             if rag_ctx is not None else ""))
        if self.cache_size:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
            if cached is not None:
                self.counters["requests"] += 1
                self.counters["cache_hits"] += 1
                elapsed_hit = round(time.perf_counter() - t_all, 5)
                self.counters["total_s"] += elapsed_hit
                out = dict(cached)
                out["request_id"] = self.counters["requests"]
                out["cached"] = True
                out["cached_orig_elapsed_s"] = cached["elapsed_s"]
                out["elapsed_s"] = elapsed_hit
                out["free_vram_gb"] = (
                    round(torch.cuda.mem_get_info()[0] / 2**30, 2)
                    if torch.cuda.is_available() else None)
                self._log_request(out)
                return out
            self.counters["cache_misses"] += 1
        units = [Unit(k, self._abs(self.specs[k].model_path),
                      self.specs[k].est_gb,
                      prompt=self._prompt_for(
                          k, query, history,
                          list_format=(mode == "quality" and
                                       list_kind(query) == "tr"),
                          extra_system=extra_system),
                      max_new_tokens=int(max_new_tokens)) for k in keys]
        # Paged units always run alone (they share the GPU and stream weights);
        # only dense units are packed into parallel groups.
        dense = [u for u in units if self.specs[u.name].kind == "dense"]
        paged = [u for u in units if self.specs[u.name].kind == "paged"]
        groups = plan(dense, budget) if dense else []
        groups += [[u] for u in paged]
        t_queue = time.perf_counter()
        with self._lock:
            queue_s = time.perf_counter() - t_queue
            self.counters["queue_s"] += queue_s
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            timings = {"load_s": 0.0, "gen_s": 0.0, "unload_s": 0.0,
                       "queue_s": queue_s}
            group_meta = []
            for g in groups:
                kinds = {self.specs[u.name].kind for u in g}
                if kinds == {"paged"}:
                    t0 = time.perf_counter()
                    self.pool.evict_all()
                    for u in g:
                        self._run_paged(u)
                    timings["gen_s"] += max(u.gen_s for u in g)
                    timings["load_s"] += sum(u.load_s for u in g)
                    wall = time.perf_counter() - t0
                elif kinds == {"dense"}:
                    t0 = time.perf_counter()
                    for eng in self._paged.values():
                        eng.release_gpu()
                    r = self.pool.run(g)
                    for k in ("load_s", "gen_s", "unload_s"):
                        timings[k] += r[k]
                    wall = time.perf_counter() - t0
                else:
                    raise RuntimeError("mixed dense/paged group is not supported")
                group_meta.append({
                    "units": [u.name for u in g],
                    "kind": next(iter(kinds)),
                    "wall_s": round(wall, 3),
                })
            if mode == "quality" and list_kind(query):
                if (list_kind(query) == "other" or
                        all(u.name != "chat3b" for u in units)):
                    for u in units:
                        u.text = repair_list_text(u.text or "")
            merged = merge_outputs(units, merge)
            elapsed = time.perf_counter() - t_all
            peak_mb = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                       if torch.cuda.is_available() else None)
            free_gb = (round(torch.cuda.mem_get_info()[0] / 2**30, 2)
                       if torch.cuda.is_available() else None)
            warnings = []
            if torch.cuda.is_available() and peak_mb is not None:
                total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
                if peak_mb / 1024.0 > total_gb - 0.2:
                    warnings.append(
                        f"peak {peak_mb:.0f}MB is at the {total_gb:.1f}GB VRAM limit "
                        f"(driver may spill to shared memory)")
            for u in units:
                est = self.specs[u.name].est_gb
                if est > budget:
                    warnings.append(
                        f"unit {u.name} needs {est:.1f}GB > budget {budget:.1f}GB; "
                        f"it runs alone and exceeds the declared budget")

        self.counters["requests"] += 1
        self.counters["total_s"] += elapsed
        self.counters["gen_s"] += timings["gen_s"]
        response = {
            "query": query,
            "history_len": (len(history) if history is not None else 0),
            "answer": merged,
            "mode": mode,
            "route": {"reason": reason, "units": keys},
            "plan": group_meta,
            "n_groups": len(groups),
            "parallel": len(groups) == 1 and len(units) > 1,
            "merge": merge,
            "budget_gb": budget,
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "elapsed_s": round(elapsed, 3),
            "per_unit": {
                u.name: {
                    "label": self.specs[u.name].label,
                    "text": (u.text or "").strip(),
                    "score": (round(u.score, 4) if u.score is not None else None),
                    "gen_s": round(u.gen_s, 3),
                } for u in units
            },
            "peak_vram_mb": peak_mb,
            "free_vram_gb": free_gb,
            "warnings": warnings,
            "rag": self._rag_payload(rag_ctx),
            "request_id": self.counters["requests"],
            "cached": False,
        }
        if self.cache_size:
            with self._cache_lock:
                self._cache[cache_key] = response
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        self._log_request(response)
        return response

    def stream_answer(self, query: str, mode: str = "auto",
                      max_new_tokens: int = 24, merge: str = "best",
                      budget_gb: float | None = None,
                      units: list[str] | None = None,
                      messages: list[dict] | None = None,
                      rag: bool | None = None):
        """Yield NDJSON-style events: start, token, final.

        Token streaming is implemented for a single unit; multi-unit requests
        fall back to one final event with the merged answer. The GPU lock is
        held for the whole generator, so concurrent requests queue.
        """
        t_all = time.perf_counter()
        history = self._clean_messages(messages) if messages is not None else None
        if history is not None:
            if units:
                raise ValueError("messages and units are mutually exclusive")
            if mode not in ("auto", "chat", "chat3b", "chat4b", "alt", "quality"):
                raise ValueError(
                    "messages are only supported with mode auto/chat/chat3b/chat4b/alt/quality")
            if mode == "chat3b":
                if "chat3b" not in self.specs:
                    raise ValueError("chat3b unit is not available (3B instruct model missing)")
                keys = ["chat3b"]
                reason = f"chat3b: {len(history)} mesajlik konusma (3B paged)"
            elif mode == "chat4b":
                if "chat4b" not in self.specs:
                    raise ValueError("chat4b unit is not available (Qwen3-4B model missing)")
                keys = ["chat4b"]
                reason = f"chat4b: {len(history)} mesajlik konusma (4B GPU-int8)"
            elif mode == "alt":
                if "alt" not in self.specs:
                    raise ValueError("alt unit is not available (SmolLM model missing)")
                keys = ["alt"]
                reason = f"alt: {len(history)} mesajlik konusma (Qwen disi model)"
            elif mode == "quality":
                keys, reason = route(
                    history[-1]["content"], "quality",
                    chat="chat" in self.specs,
                    chat3b="chat3b" in self.specs,
                    chat4b="chat4b" in self.specs,
                    alt="alt" in self.specs)
                if keys == ["deep"]:
                    fallback = ("chat4b" if "chat4b" in self.specs else
                                ("chat3b" if "chat3b" in self.specs else "chat"))
                    keys = [fallback]
                    reason += f" (mesajda {fallback} Instruct'a dusuldu)"
                if not any(k in self.specs for k in ("chat", "chat3b", "chat4b")):
                    raise ValueError("no chat unit available for quality mode")
            else:
                if "chat" not in self.specs:
                    raise ValueError("chat unit is not available (model missing or --no-chat)")
                keys = ["chat"]
                reason = f"chat: {len(history)} mesajlik konusma"
            query = history[-1]["content"]
        elif units:
            keys = list(dict.fromkeys(units))
            bad = [k for k in keys if k not in self.specs]
            if bad:
                raise ValueError(f"unknown unit(s): {bad}")
            reason = "kullanici birimleri secti: " + ", ".join(keys)
        else:
            keys, reason = route(query, mode, chat="chat" in self.specs,
                                 chat3b="chat3b" in self.specs,
                                 chat4b="chat4b" in self.specs,
                                 alt="alt" in self.specs)
        budget = float(self.budget_gb if budget_gb is None else budget_gb)
        use_rag = self.rag_enabled if rag is None else bool(rag)
        rag_ctx = self._rag_for(query, mode, keys, use_rag)
        extra_system = rag_ctx.system if rag_ctx is not None else None
        if rag_ctx is not None:
            reason += f" + yerel RAG ({len(rag_ctx.sources)} kaynak)"
        if len(keys) != 1:
            yield {"event": "final",
                   "response": self.answer(query, mode=mode,
                                           max_new_tokens=max_new_tokens,
                                           merge=merge, budget_gb=budget_gb,
                                           units=units, messages=history,
                                           rag=use_rag)}
            return
        key = keys[0]
        spec = self.specs[key]
        prompt = self._prompt_for(
            key, query, history,
            list_format=(mode == "quality" and list_kind(query) == "tr"),
            extra_system=extra_system)
        _list_kind = list_kind(query)
        repair_lists = (mode == "quality" and
                        (_list_kind == "other" or
                         (_list_kind == "tr" and key == "chat")))
        cache_key = self._cache_key(query, mode, max_new_tokens, merge, budget,
                                    keys, history,
                                    rag_sig=(rag_ctx.signature
                                             if rag_ctx is not None else ""))
        if self.cache_size:
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache.move_to_end(cache_key)
            if cached is not None:
                self.counters["requests"] += 1
                self.counters["cache_hits"] += 1
                elapsed_hit = round(time.perf_counter() - t_all, 5)
                self.counters["total_s"] += elapsed_hit
                out = dict(cached)
                out["request_id"] = self.counters["requests"]
                out["cached"] = True
                out["cached_orig_elapsed_s"] = cached["elapsed_s"]
                out["elapsed_s"] = elapsed_hit
                self._log_request(out)
                yield {"event": "final", "response": out, "cached": True}
                return
            self.counters["cache_misses"] += 1

        path = self._abs(spec.model_path)
        unit = Unit(key, path, spec.est_gb, prompt=prompt,
                    max_new_tokens=int(max_new_tokens))
        t_queue = time.perf_counter()
        with self._lock:
            queue_s = time.perf_counter() - t_queue
            self.counters["queue_s"] += queue_s
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            yield {"event": "start", "unit": key, "label": spec.label,
                   "reason": reason, "queue_s": round(queue_s, 3),
                   "rag": self._rag_payload(rag_ctx),
                   "loaded": path in self._paged}
            timings = {"load_s": 0.0, "gen_s": 0.0, "unload_s": 0.0,
                       "queue_s": queue_s}
            t_group = time.perf_counter()
            if spec.kind == "paged":
                self.pool.evict_all()
                self._release_other_paged(path)
                t_load = time.perf_counter()
                eng = self._get_paged(path)
                tok = self._get_tok(path)
                timings["load_s"] = time.perf_counter() - t_load
                ids = tok(prompt, return_tensors="pt").input_ids
                ids = self._fit_prompt(eng, ids, int(max_new_tokens))
                pieces: list[str] = []
                tids: list[int] = []
                shown = ""
                t_gen = time.perf_counter()
                for tid in eng.generate_stream(ids,
                                               max_new_tokens=int(max_new_tokens),
                                               temperature=0.0, seed=1234,
                                               eos_id=tok.eos_token_id):
                    tids.append(tid)
                    # Decode the whole answer so far: a single token can be a
                    # partial UTF-8 character, and per-token decoding would
                    # print a replacement glyph for Turkish letters or emoji.
                    # Withhold trailing U+FFFD until the bytes resolve.
                    current = tok.decode(tids, skip_special_tokens=True)
                    safe = current.rstrip("\ufffd")
                    if safe.startswith(shown):
                        piece = safe[len(shown):]
                        shown = safe
                    else:
                        n = 0
                        while (n < min(len(shown), len(safe)) and
                               shown[n] == safe[n]):
                            n += 1
                        piece = safe[n:]
                        shown = safe
                    if piece:
                        pieces.append(piece)
                        if not repair_lists:
                            yield {"event": "token", "text": piece}
                timings["gen_s"] = time.perf_counter() - t_gen
                text = "".join(pieces)
            else:
                for e in self._paged.values():
                    e.release_gpu()
                t_load = time.perf_counter()
                entry = self.pool.get(path, spec.est_gb)
                timings["load_s"] = time.perf_counter() - t_load
                tok, model = entry["tok"], entry["model"]
                from transformers import (StoppingCriteria, StoppingCriteriaList,
                                          TextIteratorStreamer)
                ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
                streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                                skip_special_tokens=True)
                stop = threading.Event()
                errs: list[BaseException] = []

                class _Stop(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        return stop.is_set()

                def work() -> None:
                    try:
                        model.generate(
                            ids, max_new_tokens=int(max_new_tokens),
                            do_sample=False, pad_token_id=tok.eos_token_id,
                            streamer=streamer,
                            stopping_criteria=StoppingCriteriaList([_Stop()]))
                    except BaseException as ex:  # surfaced after the join
                        errs.append(ex)

                th = threading.Thread(target=work, daemon=True)
                th.start()
                pieces = []
                t_gen = time.perf_counter()
                try:
                    for chunk in streamer:
                        if chunk:
                            pieces.append(chunk)
                            if not repair_lists:
                                yield {"event": "token", "text": chunk}
                finally:
                    stop.set()
                    th.join(timeout=10)
                timings["gen_s"] = time.perf_counter() - t_gen
                if errs:
                    raise errs[0]
                text = "".join(pieces)
            if repair_lists:
                text = repair_list_text(text)
            wall = time.perf_counter() - t_group
            unit.text = text
            unit.gen_s = timings["gen_s"]
            unit.score = None
            elapsed = time.perf_counter() - t_all
            peak_mb = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                       if torch.cuda.is_available() else None)
            free_gb = (round(torch.cuda.mem_get_info()[0] / 2**30, 2)
                       if torch.cuda.is_available() else None)
            warnings = []
            if torch.cuda.is_available() and peak_mb is not None:
                total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
                if peak_mb / 1024.0 > total_gb - 0.2:
                    warnings.append(
                        f"peak {peak_mb:.0f}MB is at the {total_gb:.1f}GB VRAM "
                        f"limit (driver may spill to shared memory)")
            if spec.est_gb > budget:
                warnings.append(
                    f"unit {key} needs {spec.est_gb:.1f}GB > budget "
                    f"{budget:.1f}GB; it runs alone and exceeds the declared budget")
            self.counters["requests"] += 1
            self.counters["total_s"] += elapsed
            self.counters["gen_s"] += timings["gen_s"]
            response = {
                "query": query,
                "history_len": (len(history) if history is not None else 0),
                "answer": text.strip(),
                "mode": mode,
                "route": {"reason": reason, "units": keys},
                "plan": [{"units": [key], "kind": spec.kind,
                          "wall_s": round(wall, 3)}],
                "n_groups": 1,
                "parallel": False,
                "merge": merge,
                "budget_gb": budget,
                "timings": {k: round(v, 3) for k, v in timings.items()},
                "elapsed_s": round(elapsed, 3),
                "per_unit": {key: {"label": spec.label,
                                   "text": text.strip(), "score": None,
                                   "gen_s": round(unit.gen_s, 3)}},
                "peak_vram_mb": peak_mb,
                "free_vram_gb": free_gb,
                "warnings": warnings,
                "rag": self._rag_payload(rag_ctx),
                "request_id": self.counters["requests"],
                "cached": False,
            }
            if self.cache_size:
                with self._cache_lock:
                    self._cache[cache_key] = response
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
            self._log_request(response)
            yield {"event": "final", "response": response}

    @staticmethod
    def _cache_key(query: str, mode: str, max_new_tokens: int, merge: str,
                   budget: float, keys: list[str],
                   messages: list[dict] | None = None,
                   rag_sig: str = "") -> str:
        return json.dumps({
            "q": " ".join(query.split()),
            "messages": messages,
            "mode": mode,
            "n": int(max_new_tokens),
            "merge": merge,
            "budget": round(float(budget), 3),
            "units": list(keys),
            "rag": rag_sig,
        }, sort_keys=True, ensure_ascii=True)

    def _log_request(self, response: dict) -> None:
        """Append one JSON line per request; never break serving on I/O errors."""
        if not self.log_path:
            return
        keys = ("request_id", "query", "history_len", "mode", "route", "plan",
                "n_groups", "parallel", "merge", "budget_gb", "timings",
                "elapsed_s", "peak_vram_mb", "free_vram_gb", "warnings",
                "rag", "cached", "cached_orig_elapsed_s")
        rec = {k: response.get(k) for k in keys}
        rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------- stats
    def stats(self) -> dict:
        paged = {Path(p).name: {"layers": e.n_layers, "quant": e.quant}
                 for p, e in self._paged.items()}
        n = max(1, self.counters["requests"])
        return {
            "counters": {k: (round(v, 2) if isinstance(v, float) else v)
                         for k, v in self.counters.items()},
            "avg_elapsed_s": round(self.counters["total_s"] / n, 2),
            "avg_gen_s": round(self.counters["gen_s"] / n, 2),
            "pool": self.pool.stats(),
            "paged": paged,
            "cache": {"size": len(self._cache), "capacity": self.cache_size,
                      "hits": self.counters["cache_hits"],
                      "misses": self.counters["cache_misses"]},
            "queue": {"in_flight": self._lock.locked(),
                      "wait_s_total": round(self.counters["queue_s"], 2)},
            "deep_ckpt": (Path(self.deep_ckpt).name if self.deep_ckpt else None),
            "chat_model": (Path(self.chat_model).name if self.chat_model else None),
            "chat3b_model": (Path(self.chat3b_model).name if self.chat3b_model else None),
            "chat4b_model": (Path(self.chat4b_model).name if self.chat4b_model else None),
            "rag": {
                "enabled": bool(self.rag_enabled),
                "available": bool(self.rag_store is not None and
                                  self.rag_store.available),
                "docs": (len(self.rag_store.docs)
                         if self.rag_store is not None else 0),
                "path": (str(self.rag_path.relative_to(ROOT))
                         if self.rag_path.is_relative_to(ROOT)
                         else str(self.rag_path)),
            },
            "log": (str(Path(self.log_path).relative_to(ROOT))
                    if self.log_path else None),
            "budget_gb": self.budget_gb,
            "free_vram_gb": (round(torch.cuda.mem_get_info()[0] / 2**30, 2)
                             if torch.cuda.is_available() else None),
        }
