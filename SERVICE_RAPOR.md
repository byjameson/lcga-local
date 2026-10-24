# SERVICE_RAPOR — yerel ürün + sorgu → plan → çalıştır → birleştir

**Tarih:** 20.10.2026 · **21.10 güncellemesi:** §0 (ürün yüzeyi + Qwen3-4B + RAG)
· **22.10 güncellemesi:** `exp48` 50 kapsanan + 20 kontrol RAG değerlendirmesi
· **Kod:** `llm.py`, `lcsa/service.py`, `lcsa/paged_qwen.py`, `lcsa/rag.py`,
`serve.py`, `tests/test_service_unit.py`, `tests/test_http_service.py`,
`tests/test_paged_qwen.py`, `tests/test_paged_quant_cache.py`,
`tests/test_rag.py`, `scripts/download_qwen3_4b.py`,
`scripts/build_rag_corpus.py`, `scripts/rag_add_pages.py`,
`scripts/check_rag_facts.py`, `experiments/exp48_rag_eval.py`,
`experiments/exp42_chat4b_quality.py`, `exp43_quality4b_e2e.py`,
`exp44_system_prompt_4b.py`, `experiments/exp47_rag_eval.py`,
`experiments/exp47_retrieval_check.py`, `experiments/exp47_resummarize.py`,
`experiments/exp22_service_bench.py`,
`exp23_ckpt_eval.py`, `exp24_ood_probe.py`, `exp25_chat_eval.py`,
`exp25b_system_prompt.py`, `exp25c_multiturn_probe.py`, `exp26_quality60.py`,
`exp27_list_prompt.py`, `exp28_openai_smoke.py`, `exp29_memory_prefix.py`,
`exp29b_memory_name.py`, `exp30_chat3b_quality.py`, `exp31_router_eval.py`,
`exp32_auto_eval.py`, `exp33_quality_route_eval.py`, `exp36_learned_router.py`,
`exp37_alt_model_eval.py`, `exp39_list_format_probe.py`,
`exp40_hybrid_list_router.py`, `exp46_multiturn_4b.py`,
`prompts_holdout.py` ·
**Ölçüm:**
`results/exp22_service.json`, `exp23_ckpt_eval.json`, `exp24b_ood.json`,
`results/exp25_chat_eval.json`, `exp25b_system_prompt.json`,
`exp25c_multiturn.json`, `exp26_quality60_rescored.json`,
`exp27_list_prompt.json`, `exp28_openai_smoke.json`,
`exp29_memory_prefix.json`, `exp29b_memory_name.json`,
`exp29c_ui_memory.json`, `exp30_60_chat3b_quality.json`,
`exp34_holdout_chat3b_quality.json`, `exp31_60_router_eval.json`,
`exp32_auto_eval.json`, `exp33_*_quality_route.json`,
`exp36_learned_router.json`, `exp37_*_alt_quality.json`,
`exp39_list_format_probe.json`, `exp40_hybrid_list_router.json`,
`exp41_http_hybrid_smoke*.json`, `exp42_60_chat4b_quality.json`,
`exp42_holdout_chat4b_quality.json`, `exp42_resident60_chat4b_quality.json`,
`exp42_resident_holdout_chat4b_quality.json`, `exp43_60_quality4b_e2e.json`,
`exp43_holdout_quality4b_e2e.json`, `exp44_system_prompt_4b.json`,
`exp45_quant_cache` (JSONL), `exp46_multiturn_4b.json`,
`exp48_rag_eval.json`, `exp48_retrieval_check.txt`,
`exp48_manual_review.md`, `exp48_console.log`,
`exp47_rag_eval.json`, `exp47_retrieval_check.txt`, `exp47_review.txt`,
`exp47_manual_review.md`, `quality_router_smoke.log`, `results/results.jsonl`

## 0. 2026-10-21 guncellemesi - urun yuzeyi + Qwen3-4B

**Urun yuzeyi artik HTTP degil:** `python llm.py` (veya `sohbet.cmd`) ile
acilan yerel REPL. `serve.py` opsiyonel gelistirici demosu olarak duruyor;
auth/CORS/rate-limit bu sprintin hedefi degil.

**Qwen3-4B-Instruct-2507, int8 GPU-resident:** 36 katmanin tamami int8 olarak
GPU'da (3465 MB), tek bf16 slot (193 MB) uzerinden katman katman dequant.
Embed 742 MB; olculen peak **4513 MB**, ilk yukleme **18.8 s**, decode
**~3.4 tok/s**, bosta **0.59 GB**. Ayni model CPU paging ile 1.49 tok/s ve
4804 MB idi; resident-int8 **2.3x** hizli ve daha az VRAM. `PagedQwen(...,
resident_quant=True)`; yalnizca `chat4b` icin acilir.

**Int8 quant-cache (`exp45`):** 36 katmanin int8 agirliklari + satir
olcekleri + small tensorleri **3468 MB** dosyada tutulur
(`cache/qwen3_4b_instruct_2507_int8_resident.pt`). Ilk kosu nicemleme yapar
(14.1 s) ve cache'i 3.2 s'de yazar; sonraki kosular model dosyasi/konfig
dogrulamasini gecip nicemlemeyi atlar. Olculen cache-hit yukleme **2.44 s**
(cache okuma 1.29 s), `Service.answer` **2.34 s**, CLI **2.3 s**; exact parity
testi (`test_paged_quant_cache`) gecti. Sureler sicak dosya onbelleginde
olculdu; soguk disk okumasi ayri olculmedi.

**Cok-tur hafiza probe'u (`exp46`, 4B):** son 8 mesaj penceresinde ad, tercih,
onceki konu ve duzeltme **4/4** dogru; ilk olgu pencere disina cikarildiginda
benzersiz sayi **0/1** hatirlandi. Bu, pencere davranisini dogrular; genel
hafiza iddiasi degildir.

**Yerel RAG (`exp48`, 326 sayfalik korpus):** `data/rag/world_facts.jsonl`
**326 dokuman / 4.793.376 bayt (4,57 MiB)**; CPU-only BM25, runtime'da internet
yok. Urun varsayilani acik, `--no-rag` kapatir. Kesin olgu sorularinda tek
kaynak, genis sorularda yalnizca konuyu bagimsiz destekleyen ikinci kaynak
kullanilir. Tam korpusta retrieval tanisi **50/50** kapsanan soruda beklenen
olgu baglamda, **50/50** sorguda tek kaynak, **20/20** kontrolde RAG yok;
indeks yukleme **1,55 s** (onceki olcum 2,14 s). `exp48` 50 kapsanan + 20
kontrol sorusunu baseline/RAG olarak 4B ile kostu (`max_new=64`, 140 gecerli
satir). Kapsanan keyword **42/50 baseline -> 49/50 RAG**; yeni 30 soruda
**24/30 -> 30/30**; 50/50 RAG cevabi manuel okumada olgusal olarak dogru;
RAG atif **50/50**, RAG kullanimi **50/50**; kontrollerde RAG **0/20**, atif
**0/20**. Tek keyword kacagi `tr-ocean`: cevap dogru Turkce ad **Buyuk
Okyanus**, beklenen literal "Pasifik" degil. Strict exact **4/36 -> 0/36**
(RAG cumle/atif ekledigi icin uygun metrik degil). Ortalama **5,82/5,85 s**,
uretim **5,74/5,33 s**, peak **4517,8/4547,7 MB**. Bu 50 sorunun tamaminda
kaynak basligi dogrudan konuyla ilgiliydi; `tr-nile`, `tr-whale`, `tr-nl` ve
`tr-ocean` gibi eski kaynak karismalari duzeldi. Sinir: 326 sayfa genel
benchmark degil; siralama bu dar sette tek kaynaga indirildi, kapsam disi veya
cok benzer sayfalarda hata yapabilir; atif bicimi tek basina dogruluk kaniti
degil. Ham cevaplar `results/exp48_rag_eval.json`, kaynak denetimi
`results/exp48_retrieval_check.txt`, inceleme notlari
`results/exp48_manual_review.md` icinde.

**Servis:** yeni `chat4b` birimi (4.9 GB ilan), `quality` rotasinda 4B varsa
genel/kod/liste -> `chat4b`, kisa nezaket -> `chat`; 4B yoksa eski 3B policy.
Streaming'de ceviri UTF-8 icin biriken token id'leri decode edilir; yarim
karakterde U+FFFD bastirilir. CLI: `/yardim /model /temizle /durum /cik`,
kalici son 8 mesaj, Ctrl+C ile kesme, `--stats`, tek seferlik soru (konumsal
arguman; `python llm.py --stats "..."`).

**Olcumler (exp42/43):**

| Set | Yol | Keyword | Exact | Liste | Sure | Peak |
|---|---|---|---|---|---|---|
| 60 | 3B paged | %88.5 | %92.3 | %12.5 | 2.28 s | 4751 MB |
| 60 | 4B paged | %92.3 | %84.6 | %12.5 | 5.65 s | 4746 MB |
| 60 | **4B resident raw** | **%92.3** | **%84.6** | **%12.5** | **3.21 s / 2.72 tok/s** | 4519 MB |
| 60 | **urun quality (4B resident)** | **%94.2** | %84.6 | %87.5 | 3.23 s / 2.99 tok/s | 4526 MB |
| 35 | 3B paged | %86.7 | %90.0 | %20.0 | 2.52 s | 4751 MB |
| 35 | 4B paged | %83.3 | %90.0 | %40.0 | 7.06 s | 4746 MB |
| 35 | **4B resident raw** | **%83.3** | **%90.0** | **%40.0** | **3.96 s / 2.71 tok/s** | 4519 MB |
| 35 | **urun quality (4B resident)** | %83.3 | %90.0 | **%100** | 3.97 s / 3.17 tok/s | 4526 MB |

60 + 35 holdout keyword oranlarinda ham 3B/4B basa bas; 4B Turkce olgularda
daha iyi, bazi talimat/cevirilerde daha zayif. **Ayni 95 probda 4B resident
(36/36 int8) ile paged (19 katman bf16) ham metrikleri birebir ayni cikti**
(60: %92.3/%84.6/%12.5; holdout: %83.3/%90.0/%40.0); ortalama uretim
5.65/7.06 s -> **3.21/3.96 s**. Bu, bu sette kuantizasyon kaybi gorulmedigini
gosterir; genel kalite garantisi degildir. Urun yolu 60'ta 3B hybrid'in
(exp40 %91.7) uzerinde, holdout'ta altinda (%83.3 vs %85.7); fark gurultu
icinde. 4B'nin kazanci: daha yeni/dogal cevaplar, benzer decode hizi, daha az
VRAM; bedeli: ilk seferde 18.8 s yukleme (cache sonrasi 2.4 s) ve bazi talimat
hatalari (212F, "Koyde kopek var", Misir -> Kairo).

**Testler:** **79** (53 service/HTTP + 3 Qwen3 CPU parity + 2 resident-cache
parity + 21 RAG). Yeni: `test_chat4b_mode_requires_unit`,
`test_quality_prefers_chat4b_when_available`, `test_openai_chat4b_model`,
`tests/test_paged_qwen.py`, `tests/test_paged_quant_cache.py`,
`tests/test_rag.py`. Tam kosu **79/79**, 1,47 s; `py_compile` temiz.

## 1. Ne yapildi

Araştırma parçaları (paged motor, model havuzu, scheduler, merge) **tek bir
serviste** birleştirildi; üstüne stdlib HTTP API + demo arayüzü ve gerçek
sorgularla uçtan uca ölçüm eklendi. Dış bağımlılık yok (FastAPI/uvicorn yok).

| Bileşen | Dosya | Görev |
|---|---|---|
| Servis çekirdeği | `lcsa/service.py` | rota → plan → yürütme → birleştirme + metrikler |
| HTTP + demo | `serve.py` | `/`, `/health`, `/units`, `/stats`, `/metrics`, `POST /v1/answer`, `POST /v1/stream`, `POST /v1/preload`, `GET /v1/models`, `POST /v1/chat/completions` |
| Ölçüm | `experiments/exp22_service_bench.py` | 8 gerçek sorgu + 3 senaryo |
| Checkpoint | `lcsa/service.py` + `serve.py` | `--deep-ckpt auto`: mmap state_dict → int8 paged |
| Kalite | `exp23_ckpt_eval.py`, `exp24_ood_probe.py`, `exp25_chat_eval.py`, `exp25b_system_prompt.py`, `exp25c_multiturn_probe.py`, `exp26_quality60.py`, `exp27_list_prompt.py`, `exp29_memory_prefix.py`, `exp29b_memory_name.py`, `exp30_chat3b_quality.py`, `exp31_router_eval.py`, `exp32_auto_eval.py`, `exp33_quality_route_eval.py` | held-out CE + OOD + chat probu + prompt/çok-tur/60-prob/liste/bellek/3B kalite ve router ölçümü |
| Testler | `tests/test_service_unit.py`, `tests/test_http_service.py`, `tests/test_paged_qwen.py`, `tests/test_paged_quant_cache.py`, `tests/test_rag.py` | 53 CPU-only rota/plan/merge/cache/mesaj/HTTP/OpenAI + 3 Qwen3 parity + 2 cache parity + 21 RAG = **79** test |

## 2. Mimari

**Birimler (unit):**

| Anahtar | Model | VRAM (ilan) | Tür | Rol |
|---|---|---|---|---|
| `fast` | Qwen2.5-0.5B | 1,1GB | dense bf16 | hızlı, kısa/genel |
| `balanced` | Qwen2.5-1.5B | 3,3GB | dense bf16 | dengeli kalite |
| `chat` | Qwen2.5-1.5B-Instruct | 3,3GB | dense bf16 | instruct sohbet; model varsa otomatik |
| `deep` | Qwen2.5-3B | 4,7GB | **int8 paged** | 3B base/checkpoint; VRAM'e sığmaz; kod/akıl yürütme |
| `chat3b` | Qwen2.5-3B-Instruct | 4,7GB | **int8 paged** | 3B instruct kalite modu; model tamamsa otomatik |
| `alt` | SmolLM2-1.7B-Instruct | 3,4GB | dense bf16 | **deneysel**: Qwen dışı model adaptör testi; `auto`/`quality` seçmez |

**Akış:**
1. **Rota** (`route`): kural tabanlı — selamlama/soru → `chat` (instruct varsa);
   kod/akıl yürütme ipuçları → `deep`; diğer → `fast` + `balanced` (paralel,
   `best` merge). `mode=quality` ayrı, ölçülmüş bir görev yönlendiricisidir:
   Türkçe liste → `chat3b` + numaralı-satır talimatı; İngilizce liste → `chat`
   + format düzeltme; sosyal kısa istem → `chat`; diğerleri → `chat3b`; gerekçe
   her cevapta döner (öğrenilmiş router değil). `mode=alt` yalnızca açık seçimde
   Qwen dışı SmolLM2 modelini çalıştırır; `auto` ve `quality` bu birimi kullanmaz.
2. **Plan** (`scheduler.plan`): birimleri ilan edilen VRAM'e göre gruplar;
   sığanlar aynı grupta **paralel** (ayrı CUDA stream), sığmayanlar **sıralı**.
3. **Yürütme:** dense gruplar `ModelPool` (LRU, host→GPU reload) ile; paged
   birim `PagedQwen` (int8, katman katman) ile. Paged motordan dense'e geçerken
   `PagedQwen.release_gpu()` çağrılır (aşağıdaki hata).
4. **Birleştirme** (`merge_outputs`): `best` (ortalama token logprob) veya
   `concat`; tek birimde etiketsiz düz metin.
5. **Metrik:** istek süresi, yükleme/üretim, plan, peak VRAM, uyarılar; her istek
   `results/service_requests.jsonl`'e JSON satırı olarak yazılır (`--no-log`).
   `units` alanı rota yerine açık birim seçimine izin verir; paged birimler her
   zaman tek başına (sıralı) çalışır, dense birimler bütçeye göre paralel paketlenir.
   `messages` alanı (OpenAI tarzı, son 8 mesaj) çok-turlu sohbeti `chat` birimine
   yollar; `query` ile birlikte kullanılmaz. LRU cache, kuyruk metriği
   (`timings.queue_s`) ve `/metrics` Prometheus çıktısı vardır.

## 3. Gerçek sorgu ölçümü (`exp22b`)

**Not:** Bu ölçüm chat birimi eklenmeden önce alındı; bugün `auto` selamlama ve
soru benzeri sorguları `chat` birimine yollar. Tablodaki rota tarihseldir.

3B int8 paged + 1.5B + 0.5B, `n_new=12`, bütçe 5,0GB, tek tohum:

| Sorgu | Rota | Grup | Süre | Peak |
|---|---|---|---|---|
| The capital of France is | fast | 1 | 9,68 s (ilk disk yüklemesi) | 967MB |
| Merhaba, nasılsın? | fast | 1 | 0,59 s | 975MB |
| Once upon a time… | fast+balanced | 1 (paralel) | 6,05 s | 3943MB |
| Python kodunda… | **deep** | 1 | 17,31 s (motor ilk yükleme) | 4630MB |
| Why is the sky blue? | **deep** | 1 | 4,80 s | 4630MB |
| The quick brown fox… | fast+balanced | 1 (paralel) | 3,02 s | 4629MB |
| Hesapla: 3 + 7 | **deep** | 1 | 7,99 s | 4646MB |
| (aynı fact sorusu, sıcak) | fast | 1 | **1,54 s** | 4645MB |

**Özet:** 8/8 istek hatasız; medyan **5,42 s**, medyan üretim **1,14 s**;
peak **4646MB** (bütçe 5,0GB, taşma yok); ilk istek 9,68 s, sıcak tekrar 1,54 s.

**Senaryolar (eşit sıcak başlangıç, `exp22d`, n=3, 12 token):**
- **Paralel vs sıralı** (aynı iki birim): **0,754 s vs 0,980 s → 1,30×**
  (üretim 0,749 vs 0,976 → **1,30×**). İlk koşudaki 0,88× yanıltıcıydı: paralel
  istek modelleri host'tan yeniden yüklerken (1,31 s) sıralı istek sıcaktı.
  Eşit koşulda paralellik ölçülü kazanç sağlıyor. (M4'teki 1,74× "tek tek
  çalıştırma" referansına göreydi; aynı deneyde sıralı gruba göre 1,53×.)
- **Bütçe uyarısı:** bütçe 2,0GB iken 1.5B birimi (3,3GB) sığmıyor → cevap
  `warnings` alanında `unit balanced needs 3.3GB > budget 2.0GB` döner; birim
  yalnız çalışır (peak ~4,0GB). Plan bütçeyi **gruplama** için kullanır; tek
  birimi reddetmez — ürün bunu açıkça uyarır.
- **Deep motor önbelleği:** ilk 6,56 s → ikinci 3,56 s (motor GPU'da kalıyor).

**Kalite (dürüst):** 4 olgusal prob doğru ("Paris", "Berlin", "Rayleigh
scattering", "10"); selamlama probu base 0.5B'de anlamsız ("- 1000000000").
Artık held-out CE/OOD (§4), chat probu ve prompt ablasyonu (§5) var; `chat`
birimi instruct ama 1.5B, liste takibi ve çok-tur hafıza zayıf. Serbest sohbet
için insan değerlendirmesi yok.

## 4. Kendi checkpoint'imiz serviste (exp23 / exp24b)

**Bağlama:** `deep` birimi `checkpoints/qwen3b_full_adaf_t300_best.pt`
(WT2 train'de Adafactor, 300 adım, en iyi CE 2,3217) ile yükleniyor.
`torch.load(..., mmap=True)` + `PagedQwen(state_dict=...)`: 435 HF anahtarı,
5,75GB dosya; motor 8,1 s'de kuruldu (ilk HTTP denemesinde 8,9 s grup).
`serve.py --deep-ckpt auto` dosya varsa checkpoint'i, yoksa base'i kullanır;
`--deep-ckpt none` base'e döner. `/units` ve `/health` checkpoint adını döner.

**Held-out WT2 CE (`exp23`):** aynı int8 paged motor, 4×256 test chunk:

| Varyant | CE | Yükleme | 8 token üretim | Peak |
|---|---|---|---|---|
| base 3B | 2,6106 | 6,02 s | 2,63 s | 4897MB |
| **bizim checkpoint** | **2,3700** | 6,17 s | 2,51 s | 4897MB |

ΔCE **−0,2406 (−%9,22)**; int8 kuantizasyon hatası aynı (max rel 3,94e-3).
Tek model/tek tohum/4 chunk → "ihtiyatlı kazanç"; asıl iddia servis edilen
motorun aynı kalması ve held-out CE'nin düşmesi.

**OOD probu (`exp24b`, 24 prob):** kısa fact (8), kod (4), Türkçe (12)
tamamlama; öğretmen-zorlamalı CE ve top-1:

| Alan | n | CE base | CE checkpoint | ΔCE | top-1 |
|---|---|---|---|---|---|
| fact | 8 | 1,1627 | 1,2073 | **+0,0446** | 0,771 → 0,771 |
| code | 4 | 1,3359 | 1,3038 | **−0,0321** | 0,708 → 0,646 |
| tr | 12 | 2,2908 | 2,3929 | **+0,1021** | 0,587 → 0,587 |
| tümü | 24 | 1,7556 | 1,8161 | **+0,0606** | — |

Yorum: WT2 fine-tune **alan içinde** net kazanç, **alan dışında** küçük
gerileme (özellikle Türkçe); top-1 değişmiyor. n=24, tek tohum, kısa problar →
güçlü bir genelleme iddiası değil. Bu yüzden dil-bazlı ikinci 3B birim
eklenmedi; tek `deep` birimi checkpoint ile servis ediliyor ve sınır burada
açıkça yazılı.

**İşletim doğrulaması (HTTP):** `POST /v1/preload {"units":["fast"]}` → 4,94 s
(soğuk disk), sonraki fast cevap **0,537 s**. `POST /v1/answer` ile
`units:["fast","deep"]` → **2 grup** (fast dense 0,256 s → deep paged 8,871 s),
toplam **9,128 s**, peak **4763MB**. Geçersiz birim → HTTP 400; istek günlüğü
`results/service_requests.jsonl` (istek #1, #2). **LRU cevap cache'i**
(`--cache`, varsayılan 32): aynı fast istek **9,089 s** (soğuk) → **0,00005 s**
cache hit. **Kuyruk:** iki eşzamanlı fast istekte ikincisi **8,814 s** bekledi
(elapsed 9,263 s), `/stats` bunu `queue.wait_s_total` olarak veriyor.
`GET /metrics` Prometheus metin formatında sayaçları döner. CPU-only birim
testleri: **53/53** (`tests/test_service_unit.py` + `test_http_service.py`;
`alt` rotası, `lcga-alt`, liste format düzeltmesi, `chat4b` rotası dahil) +
**3/3** Qwen3 CPU parity (`tests/test_paged_qwen.py`) + **2/2** resident
cache parity (`tests/test_paged_quant_cache.py`) + **21/21** RAG
(`tests/test_rag.py`). Toplam **79**.

## 5. Sohbet birimi ve token akışı (`exp25`, `/v1/stream`)

**Chat birimi:** Qwen2.5-1.5B-Instruct (dense 3.3GB), `models/Qwen2.5-1.5B-Instruct`
varsa otomatik kaydolur; `--no-chat` kapatır. Chat promptu tokenizer'ın
`apply_chat_template`'i ile system+user mesajına çevrilir; varsayılan system
promptu **Türkçe** ("Sen yardımcı bir asistansın. Kullanıcının dilinde, kısa ve
doğru cevap ver."). Rota: selamlama ve soru benzeri sorgular → `chat`;
kod/akıl yürütme → `deep`; diğer → fast+balanced.

**Kalite probu (`exp25`, 24 kısa prob, greedy 48 token, EN system promptu):**

| Ölçüt | base 1.5B | instruct 1.5B |
|---|---|---|
| keyword isabet (tümü) | %54,5 | **%68,2** |
| tam cevap (exact) | %25 | **%75** |
| TR talimat isabet | %20 | **%60** |
| EN fact isabet | %83,3 | **%100** |
| TR fact isabet | %33,3 | %33,3 |
| liste (3 madde) | 0/2 | 0/2 |
| ort. üretim | 1,62 s | **0,90 s** |
| tok/s | 19,1 | 18,6 |
| yükleme / peak | 3,88 s / 2957MB | 3,71 s / 2959MB |

Yorum: instruct **talimat takibini** ve İngilizce olguları belirgin iyileştiriyor;
Türkçe olgular aynı (zayıf), liste takibi 48 token sınırında başarısız. n=24,
keyword tabanlı, insan değerlendirmesi değil. Örnekler: chat "Merhaba! Ben bir
yapay zeka asistanım..." ve "Fransa'nın başkentisi Paris'dir."; base selamlamada
"100000..." tekrarına düşüyor.

**System prompt ablasyonu (`exp25b`, aynı 24 prob, tek model yüklemesi):**

| System prompt | keyword | exact | TR fact | liste | ort. üretim |
|---|---|---|---|---|---|
| EN (exp25) | %68,2 | %75 | %33,3 | 0/2 | 0,93 s |
| **TR (varsayılan)** | **%81,8** | **%100** | **%83,3** | 0/2 | **0,60 s** |
| TR + liste talimatı | %77,3 | %100 | %66,7 | **1/2** | 0,57 s |

TR promptu Türkçe olgularda belirgin iyi (5/6: Paris, Ankara, Jüpiter, 100°C,
12 ay; altın sorusu yanlış). Cevaplar kısaldığı için exact/keyword oranları
kısalıktan da faydalanıyor — dürüst yorum: kazanç gerçek ama ölçüt kısa cevabı
ödüllendiriyor. Liste talimatı listeyi 1/2'ye çıkarırken TR fact'i 4/6'ya
düşürdü; varsayılan **TR** seçildi. Bu yüzden servis artık TR system promptu
kullanıyor; `exp25` tablosu EN promptla alınmış tarihsel referanstır.

**60 problu kontrol (`exp26`, exp25'in 24 probu + 36 yeni; greedy 64 token):**
Türkçe ASCII katlama düzeltmesiyle yeniden puanlandı (1 satır değişti; eski
satırlar `results.jsonl`'de tarihsel kanıt olarak duruyor).

| Ölçüt | base 1.5B | instruct 1.5B |
|---|---|---|
| keyword isabet (tümü) | %55,8 | **%82,7** |
| exact | %15,4 | **%84,6** |
| TR fact | %25,0 | **%75,0** |
| EN fact | %83,3 | **%100** |
| TR talimat | %27,3 | **%81,8** |
| EN talimat | %90,9 | %81,8 (exact %50 → **%100**) |
| matematik | %50,0 | **%66,7** (exact %0 → **%60**) |
| 3 satırlı liste | %50,0 | **%62,5** |
| ort. üretim | 1,95 s | **0,60 s** |
| peak VRAM | 2957,8MB | 2961,0MB |

Kalan hatalar: İstanbul/Ankara, Pasifik/Atlantik, altının sembolü, en büyük
organ, bir çeviri, "düzine" ve "yarısı", satır içi listeler. EN talimatta
keyword hafif düşüyor (base'in alternatif ifadeleri) ama exact %100; bu
metriğin kısa cevabı ödüllendirdiği unutulmamalı. n=60, keyword tabanlı, insan
değerlendirmesi değil. **`exp27` liste formatı:** aynı 6 liste probunda düz
prompt **5/6**, kullanıcı eki **2/6**, system eki **3/6** — hedefli liste
talimatı işe yaramadı, entegre edilmedi.

**OpenAI uyumlu uçlar:** `GET /v1/models` ve `POST /v1/chat/completions`
(non-stream + SSE `chat.completion.chunk`, `data: [DONE]`). `messages` rolleri
`system`/`user`/`assistant`; system mesajı varsayılan asistan promptunu geçersiz
kılar. `lcga-chat` 1.5B'ye, `lcga-chat3b` 3B paged birimine, `lcga-quality`
ölçülmüş görev router'ına, `lcga-alt` opsiyonel SmolLM2 adaptörüne gider; diğer
model kimlikleri `chat`'e düşer. `model`
istemci uyumu için geri yansıtılır, `temperature` kabul edilir ama üretim
deterministiktir; `tools`/`n>1` HTTP 400 OpenAI hata şekliyle reddedilir. Canlı smoke (`exp28`): non-stream soğuk
**6,96 s** (ilk model yüklemesi), cache hit **0,018 s**; stream TTFT
**0,443 s**, toplam **2,697 s**, 12 içerik parçası, `[DONE]` var. Canlı system
mesajlı istek `Paris.` / `finish_reason=stop` döndü. İlk TTFT ölçümü tüm gövdeyi
okuduğu için geçersizdi; düzeltilmiş satır artımlı okumayla alındı.

**Çok-tur (`messages`, deneysel):** `query` yerine OpenAI tarzı
`[{"role","content"}, ...]` kabul edilir; son 8 mesaj, son rol `user`; istek
`chat` birimine zorlanır, `units` veya `mode=deep` ile birleşmez. Prompt tüm
geçmişle formatlanır. **`exp25c` 5 senaryo / 2 tur:** ikinci turda bilgiyi
hatırlama keyword isabeti **2/5 (%40)**; "önceki mesajları hatırla" talimatı da
**2/5** (kazanç yok). Örnek: "Kedimin adi Pamuk." → "Kedimin adi ne?" doğru;
"Benim adim Ali." → "Ben sizin adınız." yanlış. **`exp29`** aynı 5 senaryoda
"Anladim." ack'i ve son kullanıcı turuna `[Bilinen bilgiler: ...]` önekiyle:
düz **3/5**, önekli **4/5**; **`exp29b`** ad sorusu 4 farklı ifadede **3/4**
doğru (ad, "Kullanicinin adi ne?" veya tekrar edilen olguyla). Demo UI'da artık
**"gecmis (deneysel)"** kutusu var (varsayılan kapalı), konuşma günlüğü +
temizleme düğmesi; UI **5 şablonlu** (ad, kedi, renk, hava, toplantı) deterministik
bir bellek katmanı ekliyor ve ad sorusunu yeniden yazıyor. Tarayıcıda 3 tur
denendi: iki hatırlama turunda da "Ali" geldi (`exp29c`, n=2). Bu şablon
mühendisliği, genel hafıza değil; diğer ifadeler/açık uçlu olgular 1.5B'nin zayıf
geçmiş davranışına düşer.

**Streaming (`POST /v1/stream`):** `start` → `token`* → `final` olayları
(`text/event-stream`). Tek birimde gerçek token akışı; çoklu birimde tek `final`.
Sıcak TTFT: **fast 0,249 s** (4 token toplam 0,484 s), **deep 0,449 s** (5 token
2,003 s); soğukta yükleme baskın (9,8 s / 8,2 s). UI'da "stream" kutusu; GPU
kilidi akış boyunca tutulur, istemci koparsa jeneratör kapatılır. Tarayıcı
smoke testi: chat cevabı akışla göründü, final meta 0,793 s, peak 2974MB.

## 5b. 3B Instruct kalite modu ve ölçülen router (`exp30`–`exp41`)

`chat3b`, Qwen2.5-3B-Instruct'ı aynı int8 paged motorda çalıştırır. İki
safetensors parçası toplam **6,02GB decimal** (3,696 + 2,052GiB); model
tamamlanınca otomatik kaydolur, `--no-chat` ile kapatılmaz (ayrı bir birim).
Paged üretim Qwen'in `<|im_end|>` (151645) ve EOT (151643) tokenlarında durur;
bu düzeltilmeden 3B model "Paris'dır." sonrası uydurma bir `user ...` turu
üretiyordu. Düzeltme `PagedQwen.generate/generate_stream` ve servis çağrısında
`eos_id=tok.eos_token_id` ile yapıldı.

| Set | Birim | Keyword | Exact | 3 satır liste | Ort. üretim | Peak |
|---|---|---:|---:|---:|---:|---:|
| `exp26`, n=60 | `chat` 1.5B | %82,7 | %84,6 | %62,5 | 1,114 s | 3248MB |
| `exp26`, n=60 | `chat3b` 3B | **%88,5** | **%92,3** | %12,5 | 3,025 s | 4411MB |
| `exp34` holdout, n=35 | `chat` 1.5B | %73,3 | %70,0 | **%80,0** | 1,085 s | 3231MB |
| `exp34` holdout, n=35 | `chat3b` 3B | **%86,7** | **%90,0** | %20,0 | 3,461 s | 4411MB |

3B olgularda, talimatlarda ve aritmetikte genelde daha iyi; **strict liste
formatında (varsayılan) belirgin kötü**. Hatalar hâlâ var: "Misir'in baskenti Iskocya",
"en buyuk col Coca-Cola", "kalpte 20-25 odak" gibi cevaplar. n=60+35, keyword/
exact tabanlı; insan değerlendirmesi yok.

**Kalite politikası (hibrit):** `mode=quality` artık:
- **Türkçe liste** (`listele`) → `chat3b` + “tam 3 numaralı satır” talimatı;
- **İngilizce/diğer liste** → `chat` 1.5B + deterministik format düzeltme;
- **kısa sosyal** → `chat`; **diğer** → `chat3b`.

Aynı ölçülmüş satırlar üzerinde offline politika simülasyonu (`exp40`; yeni
uçtan uca benchmark değil, yükleme/geçiş süresi hariç):

| Set | Politika | Kalite | Ort. üretim | Birimler |
|---|---|---:|---:|---|
| `exp26`, n=60 | always-chat / always-chat3b | %78,3 / %78,3 | 1,11 / 3,02 s | — |
| `exp26`, n=60 | **hibrit quality** | **%91,7** | **2,87 s** | chat 5 / chat3b 55 |
| `exp34` holdout, n=35 | always-chat / always-chat3b | %71,4 / %74,3 | 1,09 / 3,46 s | — |
| `exp34` holdout, n=35 | **hibrit quality** | **%85,7** | **3,60 s** | chat 3 / chat3b 32 |

60 prob sonucu, fazla geniş `maddede` liste ipucu kaldırıldıktan sonra
%86,7'den **%88,3**'e, hibrit listeyle **%91,7**'ye çıktı. Ancak bu skor
**keyword/exact/format tabanlıdır**; liste metriği yalnızca üç satır olup
olmadığını kontrol eder, içeriği doğrulamaz. Manuel incelemede 12 liste
cevabının 11'i açıkça doğru, 1'i kısmiydi (holdout “sebze” cevabında zeytin
vardı). `exp39` bunun neden önemli olduğunu gösterdi: 3B + numaralı talimat
11/11 format aldı (ort. 5,515 s), fakat İngilizce “primary colors” isteminde
tekrarlı/bozuk metin üretti; bu yüzden İngilizce listeler 1.5B'de bırakıldı.
Holdout, bu hata analizi sırasında incelendiği için **%85,7 tanısal, temiz
genelleme değil**.

**Öğrenilmiş router denemesi (`exp36`):** yalnızca 60 satırla eğitilen hashed
n-gram + lojistik regresyon, holdout'ta **%82,9 / 3,03 s** aldı; eski kural
router'la aynı kalite, anlamsız küçük gecikme farkı. Kaliteyi artırmadığı için
entegre edilmedi.
**Negatif sonuç:** `exp31` confidence cascade'i (chat ortalama logprob'u düşükse
3B'ye yükselt) 60 probda always-chat'i geçmedi; test yarısında eşik hiç
tetiklenmedi. Bu yüzden cascade entegre edilmedi. `exp32` eski `auto` rota
referansı 24 probda keyword %81,8, exact %100, strict liste %0, ortalama üretim
1,107 s; liste kalitesinde zayıf.

**Canlı doğrulama (`exp41`, soğuk cache):** Türkçe liste 11,98 s, İngilizce
liste 6,40 s, fact 3,67 s, açık `alt` 6,86 s, OpenAI İngilizce liste 3,30 s;
ısınmış akışta İngilizce liste 0,91 s. Bu duvarlar model yükleme/geçişini içerir,
üretim ortalamasıyla karıştırılmamalı. İngilizce liste `quality` akışında
format düzeltme tek final olayı olarak gönderilir; token olayı yoktur.
HTTP `/v1/answer` quality TR/EN liste ve matematik rotalarını doğruladı; OpenAI
`model=lcga-quality` non-stream ve stream çalıştı. `/v1/models` model varsa
dört kimlik döner: `lcga-chat`, `lcga-chat3b`, `lcga-quality`, `lcga-alt`.
**Qwen dışı adaptör:** `exp37` SmolLM2-1.7B-Instruct'ı aynı `Service` üzerinden
`mode=alt` ile çalıştırır; amaç kalite iddiası değil, prompt/tokenizer/pool/
generation katmanının farklı bir model ailesiyle çalıştığını göstermektir.
Ölçüm: holdout keyword **%50**, exact **%20**, liste-format %100, 0,812 s,
peak 3558MB; 60 prob keyword **%55,8**, exact **%15,4**, liste-format %75,
0,893 s. Liste-format %100 olsa da içerik hatalıydı (“meyve” yerine domates);
bu yüzden `quality` bu modeli kullanmaz.

**Sınır:** Quality router kural tabanlı ve küçük sette ölçüldü; "öğrenilmiş
router" veya genel kalite garantisi değil. 3B modelin VRAM avantajı yok (4,4–4,8GB
peak); varsayılan liste formatı 1.5B'den kötü, numaralı talimat yalnızca Türkçe
listede ölçüldü. Bu yüzden `quality` varsayılan değil. `alt` deneyseldir;
`auto`/`quality` tarafından seçilmez ve ürün kalite iddiasına dahil değildir.

## 6. Bulunan ve düzeltilen dört hata (dürüstlük notu)

**(a) Paged motor + dense modeller VRAM çakışması.** İlk koşuda (`exp22`)
paged 3B motoru GPU'da kalıyordu; üstüne 0.5B+1.5B yüklenince sürücü
**shared memory'ye taştı**: peak **8554,8MB**, aynı istek 18,80 s (düzgün
koşuda 3,02 s), "paralel" senaryo 0,798× (sıralıdan yavaş). Düzeltme:
`PagedQwen.release_gpu()` (GPU kopyalarını bırakır, CPU ağırlıkları korur) +
`_layer_gpu`'da tembel yeniden kurulum + `embed/norm`'un da bırakılması.
İkinci koşuda (`exp22b`) peak **4646MB**, taşma yok. İlk koşu satırları
silinmedi — hatanın kanıtı olarak `results.jsonl`'de duruyor.

**(b) Tek birimde etiket öneki.** `merge_outputs("best")`, skoru olmayan tek
paged birimde `[deep] ...` döndürüyordu; tek birimde düz metin olacak şekilde
düzeltildi (`lcsa/scheduler.py`).

**(c) Kıyas kusuru ve uyarı metni (düzeltildi).** `exp22b`'deki paralel/sıralı
çifti eşit koşulda değildi: paralel istek modelleri host'tan yeniden yüklerken
(1,31 s) sıralı istek sıcaktı → yanıltıcı 0,88×. `exp22d` eşit sıcak başlangıçla
**1,30×** verdi. `exp22c` ise "peak > bütçe" uyarısının fazla katı olduğu ara
koşudur (fiziksel VRAM taşması değil); uyarı artık yalnız VRAM sınırına yaklaşınca
ve bütçeyi aşan birim için ayrı ayrı veriliyor. Tüm ara satırlar `results.jsonl`'de
korunuyor.

**(d) Paged instruct EOS'u (düzeltildi).** `PagedQwen` varsayılan olarak yalnız
EOT (151643) tokenında duruyordu; Qwen2.5-Instruct sohbeti `<|im_end|>` (151645)
ile bitiyor. İlk 3B smoke cevabı bu yüzden `Paris'dır.\nuser\nNe anlama gelir...`
şeklinde prompt taklidine devam etti. `eos_id` artık int veya tuple kabul ediyor;
servis tokenizer'ın `eos_token_id`'sini geçiyor. Düzeltilmiş smoke: `Paris'dır.`,
1,81 s üretim, peak 4750,7MB.

## 7. Nasıl çalıştırılır

```powershell
python serve.py --port 8000                 # deep = checkpoint, chat = instruct (varsa)
python serve.py --port 8000 --deep-ckpt none --no-chat
python experiments\exp22_service_bench.py --n_new 12 --tag exp22b
python experiments\exp22_scen_fair.py --n_new 12 --reps 3 --tag exp22d
python experiments\exp23_ckpt_eval.py --chunks 4 --seq 256 --n_new 8
python experiments\exp24_ood_probe.py --tag exp24b
python experiments\exp25_chat_eval.py --tag exp25
python experiments\exp25b_system_prompt.py --tag exp25b
python experiments\exp25c_multiturn_probe.py --tag exp25c
python experiments\exp26_quality60.py --tag exp26
python experiments\exp27_list_prompt.py
python experiments\exp28_openai_smoke.py
python experiments\exp29_memory_prefix.py
python experiments\exp29b_memory_name.py
python experiments\exp30_chat3b_quality.py --tag exp30_60 --set exp26 --max-new 64
python experiments\exp31_router_eval.py --input results\exp30_60_chat3b_quality.json --tag exp31_60
python experiments\exp32_auto_eval.py --tag exp32 --max-new 64
python experiments\exp33_quality_route_eval.py --input results\exp34_holdout_chat3b_quality.json --tag exp33_holdout
python scripts\smoke_quality_router.py
python experiments\rescore_turkish.py
python -m unittest discover -s tests -p "test_*service*.py" -v
```

```powershell
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/v1/preload -H "Content-Type: application/json" `
     -d "{\"units\":[\"deep\"]}"
curl -X POST http://127.0.0.1:8000/v1/answer -H "Content-Type: application/json" `
     -d "{\"query\":\"The capital of France is\",\"mode\":\"ensemble\",\"max_new_tokens\":12}"
curl -N -X POST http://127.0.0.1:8000/v1/stream -H "Content-Type: application/json" `
     -d "{\"query\":\"Merhaba, nasilsin?\",\"mode\":\"chat\",\"max_new_tokens\":24}"
curl http://127.0.0.1:8000/v1/models
curl -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" `
     -d "{\"model\":\"lcga-chat\",\"messages\":[{\"role\":\"user\",\"content\":\"Merhaba\"}],\"max_tokens\":24}"
```

## 8. Sınırlar (bu ne DEĞİL)

- **Tek eşzamanlı istek:** GPU kilidi var; ThreadingHTTPServer olsa da GPU işi
  serileşir (6GB kart). Kuyruk süresi `timings.queue_s` ve `/stats`'ta görünür;
  zaman aşımı/iptal yok.
- **Rota kural tabanlı**, öğrenilmiş değil; yanlış seçim yapabilir. Gerekçe
  şeffaf döner.
- **VRAM tahminleri ilan edilen sabitler** (1,1 / 3,3 / 3,3 / 4,7GB), istek
  başına ölçülmüyor; activation payı model başına sabit varsayılıyor.
- **Kalite seti hâlâ sınırlı:** 4×256 held-out CE + 24 OOD + 24/60 chat + 35
  holdout seti var; hiçbiri insan/LLM judge değil, keyword/exact tabanlı.
  Chat 1.5B instruct, TR promptla TR fact **9/12 (%75)**, 3 satırlı liste
  **5/8 (%62,5)**; altın, okyanus, İstanbul/Ankara gibi hatalar sürüyor.
  3B `chat3b` de aynı hataların bir kısmını yapıyor (Misir → İskoçya,
  en büyük çöl → Coca-Cola, kalp odacığı → 20-25) ve liste formatında daha kötü.
- **Quality router kural tabanlı ve küçük örneklemli:** 60 probda tasarlandı;
  hibrit politika 60'ta **%91,7**, 35 holdout'ta **%85,7** (always-chat %71,4,
  always-3B %74,3) format/keyword skoru aldı. Holdout liste hataları incelenirken
  kullanıldığı için tanısal; genelleme veya öğrenilmiş router iddiası değil.
  `exp31` confidence cascade negatif sonuç verdiği için entegre edilmedi.
- **Prompt duyarlılığı:** chat kalitesi system prompta göre değişiyor
  (`exp25b`: keyword %68 → %82); varsayılan TR promptu seçildi. Hedefli liste
  talimatı (`exp27`) düz prompttan kötüydü.
- **Çok-tur hafıza zayıf:** ham `messages` API'si `exp25c` 2/5; `exp29` önekli
  4/5, ama bu 5 şablonlu bir bellek katmanı. UI'daki "gecmis" kutusu deneysel ve
  varsayılan kapalı; genel konuşma hafızası yok, ürün iddiası "çok-turlu asistan"
  değil.
- **OpenAI ucu deterministik:** `temperature` yok sayılır; tool calling, `n>1`,
  auth, CORS ve rate limit yok. `lcga-chat`, `lcga-chat3b`, `lcga-quality` ve
  `lcga-alt` desteklenir; yerel demo/entegrasyon için uygundur, çok kullanıcılı
  üretim için değil.
- **Checkpoint domaini:** deep birimi WT2 fine-tune'u; alan dışında (özellikle
  Türkçe) küçük CE gerilemesi var (§4). Deep hâlâ instruct değil; chat ayrı.
- **Gecikme:** cold start 8–10 s (disk), deep motor ilk yükleme ~6–8 s (mmap
  checkpoint) veya ~10,6 s (base safetensors); chat 1.5B ~3,7 s. Sıcak istekler
  0,6–3 s, deep 3,5–8 s; 12 token içindir.
- **Streaming:** yalnız tek birimde token akışı; çoklu birim tek `final` döner.
  SSE `EventSource` (GET) değil, POST + `fetch` akışıdır. Akış başladıktan
  sonraki üretim hataları HTTP 200 içinde `error` olayı olarak döner; gövde/
  parametre doğrulaması (`messages`, `units`, `mode`) HTTP 400'dür.

## 9. Sonraki adımlar

1. 60+35 prob tamam; sıradaki **insan/LLM judge** ve hata analizi (özellikle
   TR fact, liste formatı, çeviri). Seti 100'e çıkarmak tek başına yeterli değil.
2. Hibrit liste politikasını daha büyük, temiz bir test setinde doğrula; mevcut
   holdout hata analizinde kullanıldı. Türkçe/İngilizce ayrımı dışındaki
   kategori eşiklerini ölçmeden genişletme.
3. Öğrenilmiş router (`exp36`) eski kural rota ile eşit kaldı; yeni bir sinyal
   veya daha büyük veri olmadan tekrar denemek anlamlı değil. `exp31` cascade'i
   negatif olduğu için yalnızca yeni bir sinyalle yeniden denenmeli.
4. Çok-tur hafıza: geçmişten olgu çıkarımı/özeti veya daha büyük instruct model;
   `exp25c` 2/5 olduğu için önce ölç, sonra iddia et.
5. Streaming iptali/zaman aşımı, çoklu birim akışı; SSE GET/EventSource.
6. OpenAI ucu için auth/CORS/rate limit ve tool-calling desteği; şu an
   deterministik ve tek kullanıcı.
7. Seçenek: 0.5B draft ile 3B hedefe speculative decoding (6GB'de bellek riski);
   E=64'ü 2000 adıma tamamlamak (bilimsel olarak gerekli değil).
