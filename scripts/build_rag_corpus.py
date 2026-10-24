"""Build the small local RAG corpus from Wikipedia plain-text extracts.

Usage:
    python scripts/build_rag_corpus.py
    python scripts/build_rag_corpus.py --limit 10   # quick smoke test

The generated file is ``data/rag/world_facts.jsonl``. Each row contains a
title, a short article extract, a source URL and the language. Wikipedia text
is CC BY-SA; see ``data/rag/README.md``. The product never calls Wikipedia at
runtime and works offline once this file exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "rag" / "world_facts.jsonl"
UA = "LCGA-Local/0.1 (local research; offline RAG corpus)"

# Broad but bounded: countries/capitals, planets, elements, geography, body,
# animals, language and a few history/science topics. Missing pages are skipped
# rather than guessed.
TOPICS = {
    "tr": [t.strip() for t in """
    Türkiye | Ankara | İstanbul | Fransa | Paris | Mısır | Kahire | Almanya |
    Berlin | Japonya | Tokyo | Avustralya | Canberra | Kanada | Ottawa |
    İtalya | Roma | İspanya | Madrid | Yunanistan | Atina | Rusya | Moskova |
    Çin | Pekin | Hindistan | Yeni Delhi | Brezilya | Brasília |
    Amerika Birleşik Devletleri | Washington, D.C. | Birleşik Krallık |
    Londra | İran | Tahran | Irak | Bağdat | Suudi Arabistan | Riyad |
    Azerbaycan | Bakü | Hollanda | Amsterdam | Belçika | Brüksel |
    İsviçre | Bern | Avusturya | Viyana | İsveç | Stockholm | Norveç | Oslo |
    Finlandiya | Helsinki | Danimarka | Kopenhag | Polonya | Varşova |
    Ukrayna | Kiev | Portekiz | Lizbon | Meksika | Meksiko | Arjantin |
    Buenos Aires | Güney Afrika | Pretoria | Kenya | Nairobi | Nijerya |
    Abuja | Endonezya | Cakarta | Pakistan | İslamabad | Afganistan | Kabil |
    Güneş | Merkür | Venüs | Dünya | Mars | Jüpiter | Satürn | Uranüs |
    Neptün | Ay | Altın | Gümüş | Oksijen | Hidrojen | Karbon | Demir |
    Bakır | Azot | Helyum | Sodyum | Potasyum | Kalsiyum | Cıva | Kurşun |
    Uranyum | Su | Kaynama noktası | Okyanus | Pasifik Okyanusu |
    Atlas Okyanusu | Hint Okyanusu | Arktik Okyanusu | Asya | Avrupa |
    Afrika | Kuzey Amerika | Güney Amerika | Antarktika | Sahra | Kalp |
    İnsan vücudu | Beyin | Akciğer | Karaciğer | Böbrek | Kan | DNA |
    Fotosentez | Yerçekimi | Işık hızı | Ses hızı | Bir yıl | Ay (takvim) |
    Hafta | Gün | Saat | Renk | Gökkuşağı | Nil | Amazon Nehri |
    Everest Dağı | Kedi | Köpek | Balık | Kuş | Aslan | Fil | Mavi balina |
    Zürafa | Penguen | Yunus | Arı | Karınca | Pi sayısı | Altın oran |
    Asal sayı | Pisagor teoremi | Osmanlı İmparatorluğu |
    Roma İmparatorluğu | II. Dünya Savaşı | I. Dünya Savaşı | Rönesans |
    Sanayi Devrimi | Türkçe | Mustafa Kemal Atatürk
    """.split("|") if t.strip()],
    "en": [t.strip() for t in """
    Ankara | Istanbul | Turkey | France | Paris | Egypt | Cairo | Germany |
    Berlin | Japan | Tokyo | Australia | Canberra | Canada | Ottawa | Italy |
    Rome | Spain | Madrid | Greece | Athens | Russia | Moscow | China |
    Beijing | India | New Delhi | Brazil | Brasília | United States |
    Washington, D.C. | United Kingdom | London | Iran | Tehran | Iraq |
    Baghdad | Saudi Arabia | Riyadh | Azerbaijan | Baku | Netherlands |
    Amsterdam | Belgium | Brussels | Switzerland | Bern | Austria | Vienna |
    Sweden | Stockholm | Norway | Oslo | Finland | Helsinki | Denmark |
    Copenhagen | Poland | Warsaw | Ukraine | Kyiv | Portugal | Lisbon |
    Mexico | Mexico City | Argentina | Buenos Aires | South Africa |
    Pretoria | Kenya | Nairobi | Nigeria | Abuja | Indonesia | Jakarta |
    Pakistan | Islamabad | Afghanistan | Kabul | Sun | Mercury (planet) |
    Venus | Earth | Mars | Jupiter | Saturn | Uranus | Neptune | Moon | Gold |
    Silver | Oxygen | Hydrogen | Carbon | Iron | Copper | Nitrogen | Helium |
    Sodium | Potassium | Calcium | Mercury (element) | Lead | Uranium |
    Water | Boiling point | Ocean | Pacific Ocean | Atlantic Ocean |
    Indian Ocean | Arctic Ocean | Asia | Europe | Africa | North America |
    South America | Antarctica | Sahara | Heart | Human body | Brain | Lung |
    Liver | Kidney | Blood | DNA | Photosynthesis | Gravity | Speed of light |
    Speed of sound | Year | Month | Week | Day | Hour | Color | Rainbow |
    Nile | Amazon River | Mount Everest | Cat | Dog | Fish | Bird | Lion |
    Elephant | Blue whale | Giraffe | Penguin | Dolphin | Bee | Ant | Pi |
    Golden ratio | Prime number | Pythagorean theorem | Ottoman Empire |
    Roman Empire | World War II | World War I | Renaissance |
    Industrial Revolution | Turkish language | Mustafa Kemal Atatürk
    """.split("|") if t.strip()],
}


_LOCAL = threading.local()
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
_MIN_INTERVAL = 1.15


def _throttle() -> None:
    """Keep the build polite; MediaWiki rate-limits bursts aggressively."""
    global _LAST_REQUEST
    with _RATE_LOCK:
        wait = _LAST_REQUEST + _MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST = time.time()


def _session() -> requests.Session:
    s = getattr(_LOCAL, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": UA})
        _LOCAL.session = s
    return s


def _fetch_one(lang: str, title: str) -> dict | None:
    session = _session()
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "titles": title,
        "format": "json",
        "redirects": 1,
        "formatversion": 2,
    }
    for attempt in range(3):
        try:
            _throttle()
            r = session.get(f"https://{lang}.wikipedia.org/w/api.php",
                            params=params, timeout=40)
            if r.status_code == 429:
                retry_after = r.headers.get("retry-after", "")
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = 30.0
                time.sleep(max(2.0, min(60.0, wait)))
                continue
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", [])
            if not pages:
                return None
            page = pages[0]
            if page.get("missing"):
                return None
            text = str(page.get("extract", "")).strip()
            if len(text) < 100:
                return None
            # Keep the file small; the retriever extracts short passages.
            text = text[:16000]
            resolved = str(page.get("title") or title)
            url = (f"https://{lang}.wikipedia.org/wiki/"
                   f"{quote(resolved.replace(' ', '_'))}")
            return {
                "id": f"{lang}:{resolved}",
                "title": resolved,
                "lang": lang,
                "text": text,
                "source": url,
                "license": "CC BY-SA 4.0",
            }
        except Exception as ex:
            if attempt == 2:
                print(f"[skip] {lang}:{title}: {type(ex).__name__}: {ex}")
                return None
            time.sleep(1 + attempt)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only fetch the first N topics per language")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel Wikipedia requests (default 1; rate-limit safe)")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for lang in ("tr", "en"):
        titles = list(dict.fromkeys(TOPICS[lang]))
        if args.limit:
            titles = titles[:args.limit]
        tasks.extend((lang, title) for title in titles)
    rows = []
    seen = set()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(_fetch_one, lang, title) for lang, title in tasks]
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as ex_:
                print(f"[skip] {type(ex_).__name__}: {ex_}", flush=True)
                continue
            if row and row["id"] not in seen:
                seen.add(row["id"])
                rows.append(row)
                safe = row["title"].encode("ascii", "backslashreplace").decode()
                print(f"[{len(rows):3d}] {row['lang']}:{safe} "
                      f"{len(row['text'])} chars", flush=True)
    rows.sort(key=lambda r: (r["lang"], r["title"].casefold()))
    tmp = OUT.with_name(OUT.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)
    print(f"[done] {len(rows)} docs -> {OUT} "
          f"({OUT.stat().st_size/2**20:.2f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
