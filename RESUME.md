# DURUM — 21.10.2026 · **ÜRÜN: TEK KOMUT YEREL SOHBET + QWEN3-4B**

> Ürün yüzeyi artık API/web değil: **`python llm.py`** (veya `sohbet.cmd`).
> Qwen3-4B-Instruct-2507 **int8 GPU-resident**: 36/36 katman GPU'da, tek bf16
> dequant slotu; **4513 MB peak, ~3.4 tok/s**. İlk yükleme **18.8 s**; int8
> quant-cache oluştuktan sonra ölçülen süreç başlangıcı **~2.4 s** (sıcak dosya
> önbelleği; soğuk disk ayrı ölçülmedi). Varsayılan
> `quality`: genel/kod/liste → `chat4b`, kısa nezaket → 1.5B; 4B yoksa
> `chat3b` → `chat`. `serve.py` yalnızca opsiyonel geliştirici demosu.
> **79 test** (53 service/HTTP + 3 Qwen3 parity + 2 cache parity + 21 RAG).
> `results/results.jsonl` **1625 satır**, tamamı parse. Ayrıntı: `URUN.md`,
> `SERVICE_RAPOR.md` §0.
>
> Aşağıdaki **20.10 akşamı** bölümü tarihseldir; 4B/CLI öncesi 3B kalite
> çalışmasını anlatır. Güncel kalite: 60 prob ürün yolu **%94.2 keyword,
> %84.6 exact, %87.5 liste, 3.23 s**; 35 holdout **%83.3 / %90.0 / %100,
> 3.97 s** (`exp43`). 4B ve 3B, 60 + 35 holdout keyword oranlarında başa baş; 4B Türkçe
> olgularda daha iyi, bazı talimat/çevirilerde daha zayıf. Yerel RAG (`exp48`,
> 50 kapsanan + 20 kontrol): keyword **42/50 → 49/50** (yeni 30'da
> **24/30 → 30/30**), manuel cevap doğruluğu **50/50**, RAG atıfı **50/50**,
> kontrollerde RAG **0/20**; tek keyword kaçağı doğru Türkçe ad “Büyük Okyanus”
> (beklenen literal “Pasifik”). Bu dar set, genel benchmark değil.

## 21.10 kısa durum

- **Ürün:** `llm.py` REPL; kalıcı son 8 mesaj, streaming, `/yardim /model
  /temizle /durum /cik`, Ctrl+C ile kesme, UTF-8 düzeltmesi (yarım karakterde
  U+FFFD bastırma), `--stats`, tek seferlik soru. `sohbet.cmd` çift tık.
- **4B motoru:** `PagedQwen(resident_quant=True)`. Tüm katmanlar int8 (3465 MB)
  GPU'da; her katman tek bf16 slotta (193 MB) dequant edilir. CPU paging'e göre
  **2.3× hızlı** (1.49 → 3.4 tok/s) ve daha az VRAM (4804 → 4513 MB).
- **Cold-start cache:** `PagedQwen(quant_cache=...)` ilk koşuda int8 ağırlıkları
  + ölçekleri + small tensörleri **3468 MB** dosyaya yazıyor (3.2 s); sonraki
  koşular nicemlemeyi atlıyor. Ölçülen cache-hit yükleme **2.44 s** (cache
  okuma 1.29 s), `Service.answer` **2.34 s**, CLI **2.3 s**; exact parity testi
  geçti. Model dosyası/config değişirse cache otomatik yeniden üretilir.
- **Qwen3 desteği:** `head_dim` config'ten, `q_norm/k_norm`, opsiyonel bias,
  transformers v5 `rope_parameters.rope_theta`. CPU parity **3/3**.
- **Yerel RAG:** CPU-only BM25 (`lcsa/rag.py`) + Wikipedia türevi
  `data/rag/world_facts.jsonl`: **326 doküman / 4.793.376 bayt (4,57 MiB)**;
  indeks yükleme ölçülen **1,55 s** (ikinci ölçüm 2,14 s). Kesin olgu
  sorularında tek kaynak, geniş sorularda bağımsız destekli en fazla iki kaynak
  ve `[1]` atıfı; `/durum` kaynak başlıkları. Ürün yolunda varsayılan;
  `--no-rag` ile kapanır. Kaynak yalnızca yeterli kelime örtüşmesinde eklenir;
  genel web araması değil. `exp48`: 50 kapsanan + 20 kontrolde CPU retrieval
  **50/50**, tek kaynak **50/50**, kontroller **20/20** RAG'siz; GPU keyword
  **42/50 → 49/50** (yeni 30'da **24/30 → 30/30**), manuel cevap doğruluğu
  **50/50**, atıf **50/50**; kontrollerde RAG/atıf **0/20**. Tek keyword kaçağı
  `tr-ocean` doğru Türkçe ad “Büyük Okyanus”. Ham cevaplar
  `results/exp48_rag_eval.json`, kaynak denetimi
  `results/exp48_retrieval_check.txt`, inceleme
  `results/exp48_manual_review.md`.
- **Servis:** `chat4b` birimi eklendi; `quality` 4B'yi seçer; `_get_paged`
  yalnızca `chat4b` için resident_quant açar. Streaming decode düzeltildi.
- **Ölçüm:** `exp42` 60/35 ham 3B vs 4B; `exp43` 60/35 uçtan uca `quality`
  (rota 59/60 ve 34/35 chat4b), `exp44` sistem promptu negatif sonucu
  (uzun “uydurma” promptu yanlış “doğrulandı” iddiaları üretti; mevcut kısa
  prompt kaldı), `exp46` 4B hafıza probe'u (8 mesaj içinde **4/4**, pencere
  dışına çıkan benzersiz sayı **0/1**).
- **Test:** 53 service/HTTP + 3 parity + 2 cache parity + 21 RAG = **79**;
  tam kosu **79/79**, 1,47 s; `py_compile` temiz.
- **Bilinen hatalar:** `--no-rag` yolunda Mısır → “Kairo” (Kahire değil),
  Türkçe “en büyük okyanus” → Hint Okyanusu, su → 212°F, “Köyde köpek var”
  çevirisi. RAG bu dar `exp48` setinde 50/50 RAG cevabını olgusal olarak
  düzeltti; genel garanti değil. İlk 4B yüklemesi 18.8 s (cache sonrası ~2.4 s);
  son 8 mesaj hafıza; web araması yok, yerel RAG yalnızca 326 sayfalık korpusta
  çalışır; kapsam dışı/benzer sayfalarda kaynak hatası olabilir.

## 20.10 akşamı (tarihsel)

- **Ürün kabuğu:** `lcsa/service.py`, `serve.py` (stdlib HTTP + demo UI).
  Beş Qwen birimi: `fast` 0.5B, `balanced` 1.5B, `chat` **1.5B-Instruct**
  (model varsa otomatik; `--no-chat`), `deep` 3B (**int8 paged** + checkpoint),
  `chat3b` 3B-Instruct (**int8 paged**). Model tamamsa opsiyonel `alt`
  (SmolLM2-1.7B, Qwen dışı, deneysel) eklenir. Rota kural tabanlı; gerekçe her
  cevapta döner; plan bütçeye göre paralel/sıralı.
- **Chat (`exp25`, 24 prob):** keyword isabet **%54,5 → %68,2**, exact
  **%25 → %75**, TR talimat **%20 → %60**, EN fact **%83 → %100**; TR fact
  %33 (EN promptla), liste 0/2. Üretim 1,62 → **0,90 s**, yükleme 3,7 s, peak
  2959MB. **`exp25b` system prompt ablasyonu:** TR promptla keyword **%81,8**,
  exact **%100**, TR fact **%83,3** (5/6); liste 0/2. Varsayılan artık **TR
  system promptu**; cevaplar kısa olduğu için exact/keyword kısalıktan da
  faydalanıyor (dürüst yorum). n=24, keyword tabanlı → ihtiyatlı.
- **60 problu chat kontrolü (`exp26`):** exp25'in 24 probu + 36 yeni, greedy 64
  token, TR ASCII katlama ile puanlandı. chat vs base: keyword **%82,7 vs %55,8**,
  exact **%84,6 vs %15,4**, TR fact **%75 vs %25**, EN fact **%100 vs %83**,
  TR talimat **%81,8 vs %27,3**, matematik **%66,7 vs %50**, 3 satırlı liste
  **%62,5 vs %50**. Kalan hatalar: İstanbul/Ankara, Pasifik/Atlantik, altın
  sembolü, en büyük organ, bir çeviri, düzine/yarısı, satır içi listeler. n=60,
  keyword tabanlı; insan değerlendirmesi değil.
- **3B Instruct kalite modu (`exp30`/`exp34`):** Qwen2.5-3B-Instruct iki
  shard, toplam **6,02GB decimal**, int8 paged, peak **~4,4–4,8GB**. 60 prob:
  keyword **%82,7 → %88,5**, exact **%84,6 → %92,3**, ama strict liste
  **%62,5 → %12,5**, üretim **1,114 → 3,025 s**. 35 prob holdout: keyword
  **%73,3 → %86,7**, exact **%70 → %90**, liste **%80 → %20**, üretim
  **1,085 → 3,461 s**. 3B hâlâ Türkçe olgu hataları yapıyor (Misir→İskoçya,
  en büyük çöl→Coca-Cola, kalp odacığı→20-25). n küçük, keyword/exact tabanlı;
  insan değerlendirmesi yok.
- **Hibrit `quality` (`exp40`):** Türkçe liste → 3B + numaralı-satır talimatı;
  İngilizce liste → 1.5B + deterministik format düzeltme; sosyal → 1.5B; diğer →
  3B. 60 probda **%91,7** (always %78,3), **2,87 s**, chat 5/chat3b 55; 35
  holdout'ta **%85,7** (always-chat %71,4, always-3B %74,3), **3,60 s**, chat
  3/chat3b 32. Bu, aynı satırlar üzerinde **offline politika simülasyonu**; yükleme
  hariç. Liste metriği yalnızca format; manuel incelemede 12 cevabın 11'i doğru,
  1'i kısmi. Holdout hata analizinde kullanıldığı için **tanısal**, temiz
  genelleme değil. 60'taki artışın bir kısmı `maddede` cue düzeltmesi (86,7→88,3).
- **Format tuzağı (`exp39`):** 3B + numaralı talimat 11 liste probunda format
  **11/11**, ama İngilizce “primary colors” isteminde tekrarlı/bozuk metin
  üretti; bu yüzden İngilizce listeler 1.5B'de, yalnızca Türkçe `listele`
  istemleri 3B'de. Format-only skor tek başına yeterli değil.
- **Canlı HTTP (`exp41`, soğuk cache):** TR liste 11,98 s, EN liste 6,40 s, fact
  3,67 s, `alt` 6,86 s, OpenAI EN liste 3,30 s; ısınmış akışta EN liste 0,91 s.
  Bu duvarlar yükleme/geçiş içerir; üretim ortalaması değil. EN liste akışı
  format düzeltmeyi tek `final` olayı olarak verir (token olayı yok).
- **Öğrenilmiş router (`exp36`):** sadece 60 satırla eğitilen hashed n-gram +
  lojistik regresyon, 35 holdout'ta **%82,9 / 3,03 s**; eski kural router ile aynı
  kalite, anlamsız küçük gecikme farkı. Kaliteyi artırmadığı için entegre
  edilmedi (eşitlik/negatif sonuç).
- **Qwen dışı adaptör (`exp37`):** opsiyonel `alt` birimi SmolLM2-1.7B-Instruct'ı
  aynı `Service`, chat template, pool ve üretim yoluyla çalıştırır; `mode=alt`,
  OpenAI `lcga-alt`, UI ve `/v1/models` kaydı var. Holdout keyword **%50**, exact
  **%20**, liste-format %100 (içerik hatalı), 0,812 s; 60 prob keyword **%55,8**,
  exact **%15,4**, 0,893 s. Kalite iddiası değil; `auto`/`quality` seçmez.
- **Negatif cascade (`exp31`):** chat güveni düşükse 3B'ye yükseltme, 60 probda
  always-chat'i geçmedi; test yarısında eşik hiç tetiklenmedi → entegre
  edilmedi. `exp32` eski `auto`: 24 probda keyword %81,8, exact %100, liste %0.
- **EOS düzeltmesi:** paged instruct yalnız EOT'ta duruyordu; `<|im_end|>`
  (151645) eklendi. Önceki smoke "Paris'dır.\nuser..." diye prompt taklidine
  devam ediyordu; şimdi "Paris'dır." ve `eos_id=tok.eos_token_id`.
- **Liste ipucu (`exp27`):** 6 liste probunda düz **5/6**, kullanıcı eki **2/6**,
  system eki **3/6** → hedefli ipucu entegre edilmedi.
- **OpenAI uyumlu API (`exp28` + `exp41`):** `GET /v1/models`,
  `POST /v1/chat/completions` non-stream + SSE `[DONE]`; canlı stream TTFT
  **0,443 s**, toplam **2,697 s**, 12 parça; non-stream cache hit 0,018 s.
  `lcga-chat` 1.5B, `lcga-chat3b` 3B, `lcga-quality` router, `lcga-alt`
  (opsiyonel Qwen dışı test); `temperature` deterministik, tools/n>1 400.
  Canlı quality stream: "72", `[DONE]` var.
- **Çok-tur (`messages`, deneysel):** `/v1/answer` ve `/v1/stream` `query` yerine
  son 8 mesajlık OpenAI tarzı geçmiş alabilir; `chat` birimine zorlanır.
  **`exp25c` 5 senaryo / 2 tur:** hatırlama keyword isabeti **2/5 (%40)**;
  “önceki mesajları hatırla” talimatı da 2/5 (kazanç yok). Örnek: kedi adı
  doğru, kişi adı yanlış (“Ben sizin adınız.”). Yapı çalışıyor, güvenilir
  hafıza 1.5B'de yok; UI'da **"gecmis (deneysel)"** kutusu var (varsayılan
  kapalı), konuşma günlüğü ve temizleme düğmesi eklendi.
- **Şablon bellek (`exp29`/`29b`/`29c`):** son kullanıcı turuna
  `[Bilinen bilgiler: ...]` öneki ham geçmişi **3/5 → 4/5** yaptı; ad sorusu
  4 ifadede **3/4** doğru. UI 5 şablon (ad, kedi, renk, hava, toplantı) ve ad
  sorusu yeniden yazımı kullanıyor; tarayıcı 3 turda 2/2 “Ali”. Bu dar bir
  şablon katmanı, genel hafıza değil.
- **Streaming (`POST /v1/stream`):** `start`/`token`/`final`; sıcak TTFT
  **fast 0,249 s**, **deep 0,449 s**; 4 token 0,484 s, 5 token 2,003 s;
  soğukta yükleme baskın (9,8 / 8,2 s). Çoklu birim tek `final` döner.
  Tarayıcı UI smoke testi geçti (chat cevabı akışla geldi, final 0,793 s).
- **Cache/kuyruk/metrik:** aynı istek **9,089 s → 0,00005 s** (LRU `--cache`);
  iki eşzamanlı fast'te ikincisi **8,814 s** kuyrukta (`timings.queue_s`);
  `GET /metrics` Prometheus; `--no-log` istek günlüğünü kapatır.
- **Kendi checkpoint (`exp23`):** `deep` = `qwen3b_full_adaf_t300_best.pt`
  (`torch.load(mmap=True)` → `PagedQwen(state_dict=...)`); yükleme **6,2 s**,
  peak **4897MB**. Held-out WT2 CE **2,6106 → 2,3700 (−%9,22)**; 8 token
  2,63 → 2,51 s, 4×256 chunk, tek tohum.
- **OOD (`exp24b`, 24 prob):** tümü CE **+0,0606**; TR **+0,1021** (n=12),
  fact **+0,0446** (n=8), code **−0,0321** (n=4); top-1 değişmedi. Dil-bazlı
  ikinci 3B birim eklenmedi; sınır raporda açık.
- **Testler:** CPU-only **50/50** (`test_service_unit.py` +
  `test_http_service.py`); preload, `units`, cache, kuyruk, chat auto-route,
  `quality` route, `chat3b`, `alt`, stream, 400, `/v1/models`, OpenAI
  non-stream/stream/validation, `lcga-quality`, `lcga-alt`, liste düzeltmesi.
- **Bulunan + düzeltilen hata:** paged 3B motoru GPU'da kalınca dense modellerle
  çakıştı; ilk koşuda peak **8555MB**, aynı istek 18,80 s. `PagedQwen.release_gpu()`
  + `_layer_gpu` tembel yeniden kurulum + `embed/norm` bırakma. İlk koşu satırları
  silinmedi (hatanın kanıtı).
- **Paralel/sıralı (eşit sıcak, `exp22d`, n=3):** **0,754 s vs 0,980 s = 1,30×**;
  ilk kıyastaki 0,88× reload asimetrisiydi. Bütçeyi aşan birim yine çalışır;
  servis `warnings` döner.

## Nasıl çalıştırılır

```powershell
python llm.py                                    # ürün: yerel sohbet (quality)
python llm.py --stats "Fransa'nın başkenti neresi?"
python scripts\download_qwen3_4b.py              # 4B yoksa (~8.0GB)
python scripts\smoke_qwen3_4b.py --resident-quant
python serve.py --port 8000                      # opsiyonel geliştirici demosu
python experiments\exp42_chat4b_quality.py --set exp26 --tag exp42_60
python experiments\exp42_chat4b_quality.py --set holdout --tag exp42_holdout
python experiments\exp43_quality4b_e2e.py --set exp26 --tag exp43_60
python experiments\exp43_quality4b_e2e.py --set holdout --tag exp43_holdout
python experiments\exp44_system_prompt_4b.py --tag exp44
python scripts\check_rag_facts.py
python experiments\exp48_rag_eval.py
python -m unittest discover -s tests -v            # 79
python -m unittest tests.test_paged_qwen -v
```

Eski (tarihsel) deneyler:

```powershell
python experiments\exp22_service_bench.py --n_new 12 --tag exp22b
python experiments\exp25_chat_eval.py --tag exp25
python experiments\exp30_chat3b_quality.py --tag exp30_60 --set exp26 --max-new 64
python experiments\exp40_hybrid_list_router.py
```

## İsteğe bağlı sonraki işler (öncelik sırası)

1. **Açık uçlu kalite + insan/LLM judge:** mevcut 95 prob keyword/exact tabanlı
   ve küçük; 100'e çıkarmak tek başına yetmez. Aynı cevapları semantik olarak
   da puanla; özellikle çeviri, talimat ve “bilmiyorum” davranışı.
2. **Çok-tur hafıza:** 4B ile `exp25c` tarzı probe (ad, tercih, önceki konu);
   sonra özet/olgu hafızası dene. Şu an yalnız son 8 mesaj var; “Ali” testi
   tek başına genel hafıza kanıtı değil.
3. **Router:** 4B/3B'yi dilim bazında yönlendirme (4B olgu, 3B bazı talimat)
   ancak taze holdout'ta gürültüden büyük kazanç varsa. `exp36` öğrenilmiş
   router, `exp31` cascade negatif; aynı veriyle tekrar deneme.
4. **Semantik liste kontrolü:** `repair_list_text` biçimi düzeltiyor, içeriği
   değil; 3 satır + konu uyumu birlikte puanlanmalı (zeytin/tomato dersi).
5. **Opsiyonel/askıda:** API auth/CORS/rate-limit/tool-calling ürün hedefi
   değil; `serve.py` geliştirici demosu. 0.5B draft + 4B hedef speculative
   decoding 6GB'de riskli; E=64 eğitimi bilimsel olarak gerekli değil.

**Tamamlanan:** int8 quant-cache (18.8 s → ölçülen 2.4 s), `exp45`; exact
cache parity testi. 4B resident-int8 vs paged ablasyonu (`exp42 resident`, 95
prob): keyword/exact/liste metrikleri **birebir aynı** (60: %92.3/%84.6/%12.5;
holdout: %83.3/%90.0/%40.0), üretim 5.65/7.06 s → **3.21/3.96 s**. 4B hafıza
probe'u (`exp46`): kısa 4/4, pencere dışı 0/1. Cache-hit süresi sıcak dosya
önbelleğinde ölçüldü; soğuk disk ayrı ölçülmedi.

## Bilinen tuzaklar (kısa)

- **4B resident_quant 36/36 katmanı int8 yapar; paged mod 19 katmanı bf16
  bırakır.** İkisinin çıktısı birebir değildir; 95 probda (`exp42 resident`)
  keyword/exact/liste metrikleri aynı çıktı ama bu genel bitwise eşitlik veya
  genel kalite garantisi değildir.
- **Eski `exp43` JSON'larındaki `load_s` (6.48/5.8 s) 1.5B dense chat
  yüklemesidir.** `Unit.load_s` düzeltildi ama koşular yeniden üretilmedi;
  gerçek 4B yüklemesi için CLI’nin **18.8 s** değerini kullan.
- **`llm.py` streaming'de biriken token id'lerini decode eder**; her token'ı
  ayrı decode etmek Türkçe karakterleri U+FFFD'ye böler. Yarım karakter
  tamponu korunur.
- 4B ~4.5GB peak kullanır; tarayıcı/oyun açıkken OOM olabilir. Paged/dense
  geçişlerinde `release_gpu()` şart; aynı anda iki paged model yüklenmez.
- 4B ilk kez **18.8 s** yüklenir (int8 nicemleme); cache oluştuktan sonraki
  süreç başlangıcı ölçülen **2.4 s**'dir (sıcak dosya önbelleği; soğuk disk
  ayrı ölçülmedi). İlk cevabın duvar süresini decode hızı sanma; sonraki
  isteklerde ~3.0–3.4 tok/s.
- **Cache 3468 MB'dir ve `cache/qwen3_4b_instruct_2507_int8_resident.pt`**
  altındadır. Model dosyaları/config değişirse doğrulama başarısız olur ve
  cache yeniden üretilir; `LCGA_NO_QUANT_CACHE=1` ile kapatılabilir.
- Model algılama **tüm safetensors shard'ları + config** ister; yarım dizinde
  kaydolmaz. 4B modeli `models/Qwen3-4B-Instruct-2507` altında olmalı.
- CLI varsayılanı `--max-new 160`, değerlendirme 64; liste istemleri 64'te
  kesilebilir. `repair_list_text` biçimi düzeltir, eksik içeriği üretmez.
- Qwen3 config'te `rope_theta` üst düzeyde değil, `rope_parameters` altında;
  eski transformers alanı okuyunca parity bozulur.
- `--resume` veri akışını baştan başlatır (aynı tohum) → pencereler kesintisiz koşudan farklı.
- HF'ye `labels=ys` çift kaydırma; `torch.device('cuda') != torch.device('cuda:0')` (`.type` kullan).
- `.to()` tied Parameter'ı bozabilir; write tool İ/ı'yı bozabilir (UTF-8 + replace ile düzelt).
- Uzun GPU işlerini WMI/arka planla başlat; bitince otomatik bildirim gelir, poll etme.
- Checkpoint `mmap` ile açılır; `state_dict` bırakılsa da model tensörleri dosyayı tutar.
- `messages` geçmişi 1.5B'de güvenilir değil (`exp25c` 2/5); API desteği kalite iddiası değildir.
- Stream generator'ı GPU kilidini tutar; istemci koparsa handler `gen.close()` çağırır.

## Tamamlananlar (kısa)

M1 paged int8 · M2 E=4/E=16/E=64 · M3 prefetch · M4 scheduler + `best` merge + LRU havuz ·
M5 3B SGD/Adafactor/master + 300 adım + kuantizasyon teşhisi + `--buffers` ·
**Ürün kabuğu: servis + demo + 8 gerçek sorgu** ·
**Kendi checkpoint serviste + exp23 held-out CE (−%9,2) + exp24b OOD (24 prob)** ·
**Chat 1.5B-Instruct + exp25 (exact %25→%75) + exp25b TR prompt (keyword %68→%82) + exp26 60-prob (keyword %82,7, exact %84,6) + exp25c çok-tur (2/5) + token streaming + cache/kuyruk/metrik** ·
**OpenAI uyumlu `/v1/models` + `/v1/chat/completions` + deneysel UI geçmişi + 5 şablonlu bellek** ·
**3B Instruct paged (`chat3b`) + EOS düzeltmesi + 60/35 prob kalite ölçümü** ·
**Ölçülmüş `quality` router + canlı HTTP/OpenAI doğrulaması** ·
**Qwen3-4B-Instruct-2507 int8 GPU-resident (4513MB peak, 3.4 tok/s, 18.8s ilk yükleme; cache sonrası 2.4s) + 4B `quality` rotası** ·
**`llm.py` ürün REPL'i + `sohbet.cmd` + streaming/UTF-8 + çok tur + exp42/43/44/45/46** ·
**Yerel RAG (326 sayfa, CPU BM25) + `exp48` (50 kapsanan/20 kontrol; keyword 42/50 → 49/50, manuel 50/50)** ·
**79 test** · README + `URUN.md` + raporlar güncel.
