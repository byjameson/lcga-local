# Yerel RAG corpus'u

`world_facts.jsonl`, `scripts/build_rag_corpus.py` tarafindan Wikipedia'nin
duz metin (plain-text) cikarimlarindan uretilir. Her satirda baslik, metin,
kaynak URL, dil ve lisans alani vardir. Metinler Wikipedia'dan alinmistir ve
**CC BY-SA 4.0** lisansi altindadir; kaynak baglantilari satirlarda saklanir.

- Guncel dosya: **326 dokuman / 4.793.376 bayt (4,57 MiB)**.
- Runtime'da internet kullanilmaz; urun bu dosya ile cevrimdisi calisir.
- Dosya yoksa `RagStore.available=False` olur ve urun normal 4B yoluna doner.
- Bu korpus genel bir arama motoru degildir. Yalnizca icerdigi sayfalar
  hakkinda kaynakli cevap verebilir; kapsam disi sorularda retrieval
  kullanilmaz.
- Yenilemek icin: `python scripts/build_rag_corpus.py`.

## Olcumler

- Tam korpusta indeks yukleme **1,55 s** (ikinci olcum 2,14 s), CPU-only.
- `exp48` retrieval tanisi: kapsanan **50/50** soruda beklenen olgu baglamda,
  **50/50** sorguda tek kaynak; kontrollerde RAG **20/20** kapali.
- `exp48` (50 kapsanan + 20 kontrol, baseline/RAG, `max_new=64`): kapsanan
  keyword **42/50 -> 49/50**; yeni 30 soruda **24/30 -> 30/30**; RAG cevaplari
  manuel okumada **50/50** olgusal olarak dogru; RAG atif **50/50**, RAG
  kullanimi **50/50**; kontrollerde RAG/atif **0/20**. Tek keyword kacagi
  `tr-ocean` (dogru cevap “Buyuk Okyanus”, beklenen literal “Pasifik”).
  Strict exact **4/36 -> 0/36** anlamli degil (RAG cumle ve atif ekliyor).
- Ortalama sure **5,82/5,85 s**, uretim **5,74/5,33 s**, peak
  **4517,8/4547,7 MB**.
- Bu sette kaynak basliklari 50/50 dogrudan konuyla ilgiliydi; yine de kapsam
  disi/benzer sayfalarda hata olabilir. Atif bicimi dogruluk kaniti degildir;
  ham cevaplar `results/exp48_rag_eval.json`, kaynak denetimi
  `results/exp48_retrieval_check.txt`, inceleme
  `results/exp48_manual_review.md`.

## Degerlendirme uyarisi

`experiments/exp48_rag_eval.py` icindeki "covered" sorular bu korpustaki
konularla ortusur; bu nedenle sonuc **genel bilgi benchmark'i degildir**.
Kapsanan sette kaynakli cevap iyilesmesini, "uncovered" kontrol setinde ise
RAG'in devreye girip girmediğini ve normal sohbeti bozup bozmadigini olcer.
