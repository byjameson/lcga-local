"""LCGA Local - tek komutla calisan yerel sohbet asistani.

    python llm.py

Ilk istekte model yuklenir (4B: ilk seferde ~20 s, int8 onbellegi
olusturulduktan sonra ~3 s), sonraki cevaplar akis halinde gelir.
Varsayilan mod `quality`: Qwen3-4B-Instruct-2507 varsa onu, yoksa 3B/1.5B
Instruct modelini kullanir. Sunucu ve API gerekmez.

Komutlar: /yardim /model /temizle /durum /cik
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


HELP = """\
Komutlar:
  /yardim              bu yardimi goster
  /model               kullanilabilir modelleri listele
  /model <ad>          model/modus sec (quality, chat4b, chat3b, chat, deep, fast, ...)
  /temizle             konusma gecmisini sil
  /durum               son cevabin rota, sure, RAG kaynak ve VRAM ozeti
  /cik                 cikis

Normal yazmaya baslayin; cevap akis halinde gelir. Ctrl+C ile cevabi kesebilirsiniz.
Yerel RAG yalnizca kaynakla guclu eslesme oldugunda devreye girer; aksi halde
normal 4B yolu kullanilir. `--no-rag` ile kapatabilirsiniz.
"""


def _fmt_stats(resp: dict) -> str:
    route = resp.get("route", {})
    units = ", ".join(route.get("units", []))
    t = resp.get("timings", {})
    peak = resp.get("peak_vram_mb")
    parts = [f"rota={units or '?'}",
             f"sure={resp.get('elapsed_s', 0):.1f}s"]
    if t.get("load_s"):
        parts.append(f"yukleme={t['load_s']:.1f}s")
    if t.get("gen_s"):
        parts.append(f"uretim={t['gen_s']:.1f}s")
    if peak:
        parts.append(f"peak={peak:.0f}MB")
    rag = resp.get("rag", {})
    if rag.get("used"):
        titles = [str(s.get("title", "?")) for s in rag.get("sources", [])]
        parts.append("rag=" + ", ".join(titles[:2]))
    elif rag.get("enabled") and not rag.get("available"):
        parts.append("rag=corpus yok")
    parts.append(f"gecmis={resp.get('history_len', 0)}")
    return " | ".join(parts)


def make_service(budget: float, rag: bool = True):
    from lcsa.service import Service

    svc = Service(budget_gb=budget, log_requests=False, cache_size=0,
                  enable_chat=True, rag=rag)
    return svc


def ask(svc, text: str, mode: str, max_new: int,
        history: list[dict], system: str | None, stats: bool) -> dict | None:
    messages = list(history)
    if system:
        messages = [{"role": "system", "content": system}] + messages
    messages.append({"role": "user", "content": text})

    gen = svc.stream_answer(text, mode=mode, max_new_tokens=max_new,
                            messages=messages)
    final = None
    printed = False
    streamed_any = False
    started = time.perf_counter()
    try:
        for ev in gen:
            if ev.get("event") == "start":
                label = ev.get("label", ev.get("unit", "model"))
                if not ev.get("loaded", True):
                    print("[model yukleniyor, ilk seferde ~20 sn, sonra ~3 sn]", flush=True)
                if stats:
                    print(f"[{label}]", flush=True)
                print("Asistan: ", end="", flush=True)
                printed = True
            elif ev.get("event") == "token":
                streamed_any = True
                print(ev.get("text", ""), end="", flush=True)
            elif ev.get("event") == "final":
                final = ev.get("response", {})
                answer = str(final.get("answer", ""))
                if not printed and answer:
                    print("Asistan: ", end="", flush=True)
                    print(answer, end="", flush=True)
                    printed = True
                elif printed and not streamed_any and answer:
                    # List repair and multi-unit answers arrive as one final
                    # event without token events.
                    print(answer, end="", flush=True)
            elif ev.get("event") == "error":
                print(f"\n[hata] {ev.get('error', 'bilinmeyen hata')}",
                      flush=True)
    except KeyboardInterrupt:
        print("\n[cevap kesildi]", flush=True)
        return None
    except Exception as ex:  # keep the chat alive on a bad request/OOM
        print(f"\n[hata] {type(ex).__name__}: {ex}", flush=True)
        return None
    finally:
        try:
            gen.close()
        except Exception:
            pass
    if printed:
        print()
    if final is None:
        print("[hata] cevap alinamadi")
        return None
    if stats and final:
        print(f"[{_fmt_stats(final)}]", flush=True)
    dt = time.perf_counter() - started
    final["wall_s"] = round(dt, 2)
    return final


def repl(svc, mode: str, max_new: int, system: str | None, stats: bool) -> int:
    history: list[dict] = []
    last: dict | None = None
    units = [k for k in ("chat4b", "chat3b", "chat", "alt", "deep", "balanced", "fast")
             if k in svc.specs]
    print("LCGA Local - 6GB icin yerel sohbet")
    print(f"Mod: {mode} | birimler: {', '.join(units)}")
    rag = svc.stats().get("rag", {})
    if rag.get("enabled") and rag.get("available"):
        print(f"Yerel RAG: {rag.get('docs', 0)} kaynak (gerekince devreye girer)")
    print("Yazmaya baslayin. /yardim ile komutlar.\n")
    while True:
        try:
            text = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/cik", "/exit", "/quit"):
            break
        if text in ("/yardim", "/help", "/?"):
            print(HELP)
            continue
        if text == "/temizle":
            history.clear()
            print("[gecmis temizlendi]")
            continue
        if text == "/model":
            print("Modlar: quality, " + ", ".join(units))
            print(f"Etkin: {mode}")
            continue
        if text.startswith("/model "):
            new_mode = text.split(None, 1)[1].strip()
            if new_mode not in units and new_mode != "quality":
                print(f"[bilinmeyen mod: {new_mode}]")
                continue
            mode = new_mode
            print(f"[mod: {mode}]")
            continue
        if text == "/durum":
            if last is None:
                print("[henuz cevap yok]")
            else:
                print(_fmt_stats(last))
            continue
        if text.startswith("/"):
            print("[bilinmeyen komut; /yardim]")
            continue

        resp = ask(svc, text, mode, max_new, history, system, stats)
        if resp is None:
            continue
        answer = str(resp.get("answer", "")).strip()
        if answer:
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": answer})
            history = history[-8:]
        last = resp
    print("Gorusuruz.")
    return 0


def main() -> int:
    _utf8()
    ap = argparse.ArgumentParser(description="LCGA Local yerel sohbet")
    ap.add_argument("prompt", nargs="*", help="tek seferlik soru (REPL acilmaz)")
    ap.add_argument("--mode", default="quality",
                    help="quality (varsayilan), chat4b, chat3b, chat, deep, ...")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--budget-gb", type=float, default=5.0)
    ap.add_argument("--system", default=None,
                    help="ozel sistem istemi (varsayilan urun istemi)")
    ap.add_argument("--stats", action="store_true", help="rota/sure bilgisini goster")
    ap.add_argument("--no-rag", action="store_true",
                    help="yerel kaynak taramasini kapat")
    args = ap.parse_args()
    if not 1 <= args.max_new <= 512:
        raise SystemExit("--max-new 1..512 olmali")

    svc = make_service(args.budget_gb, rag=not args.no_rag)
    if args.no_rag:
        svc.rag_enabled = False
        svc.rag_store = None
    if not any(k in svc.specs for k in ("chat", "chat3b", "chat4b")):
        print("[hata] hicbir Instruct modeli bulunamadi; modeller eksik.",
              file=sys.stderr)
        return 2
    if args.prompt:
        text = " ".join(args.prompt).strip()
        resp = ask(svc, text, args.mode, args.max_new, [], args.system, True)
        return 0 if resp else 1
    return repl(svc, args.mode, args.max_new, args.system, args.stats)


if __name__ == "__main__":
    raise SystemExit(main())
