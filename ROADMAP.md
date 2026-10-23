# HEDEF ROADMAP — VRAM'den büyük model eğitimi + sorguya göre birim yükleme

> Vizyon: Tüketici GPU'sunda (6GB) VRAM'e sığmayan modelleri **eğitmek** ve **inference** etmek;
> inference'ta soruya göre yalnız gerekli birimleri yüklemek, sığanları paralel, sığmayanları
> sıralı çalıştırıp cevabı birleştirmek.

## Mimari gerçek (dürüst çerçeve)

Dense transformer'da **her token her katmandan geçer** → "soruya göre katman seçimi" olmaz.
Sorguya göre birim seçimi ancak şu mekanizmalarla mümkün:

| Birim türü | Seçim | Bizde |
|---|---|---|
| MoE uzmanı (top-k) | token bazında | MoE-lite E=4 ✅ |
| Erken çıkış / anytime derinlik | soru zorluğuna göre | P4 ✅ (L1–L4 %3 bant) |
| Adapter / LoRA branşı | sorgu bazında | P1 LoRA altyapısı ✅ |
| Retrieval / bilgi shard'ı | sorgu bazında | ❌ (ayrı hat) |
| Katman bloğu | seçilemez; sıralı paging | P1 streaming ✅ (eğitim) |

## Kilometre taşları

| # | Yetenek | Durum | Kabul kriteri |
|---|---|---|---|
| **M1** | **Paged inference**: 3B/bf16 modeli 6GB'de çalıştır; k katman resident, gerisi CPU'dan stream | 🟡 demo OK (bf16 2,75; int8 4,18; CPU 3,01) | 3B modelden örnek üretim; tok/s + peak VRAM; 0.5B'de HF ile logit paritesi (3,3e-5) |
| **M2** | **MoE expert paging**: E=8/16, uzmanlar CPU'da, top-1 yükle | ✅ demo (E=4): CE birebir 1,37843; paged 36,1 vs resident 39,5 tok/s; 15,4MB/pass · **E=16/64 ölçek**: talep 34,3/54,0 MB/pass; prefetch **4,07×/3,46×** azaltıyor (hit %75,5/%71,1); VRAM tasarrufu 3,35×/10,7× | Kalite ≈ resident; aktif-param oranı ve tok/s |
| **M3** | **Sorgu planlayıcı + prefetch**: promptun ilk geçişinden uzman kullanımını tahmin et, önceden yükle | 🟡 prefetch ✅: talep kopyası **15,4→1,24MB/pass (12,4×)**, hit %0,58→**%92**, isabet %92,7; sorgu planlayıcı sırada | Prefetch isabet oranı; gecikme kazancı |
| **M4** | **Scheduler + kompozisyon**: sığan birimler paralel, sığmayanlar sıralı; çıktı birleştirme | 🟡 demo: 0.5B+1.5B birlikte sığınca **paralel (gen 1,74×, 3931MB peak)**; bütçe 2,5GB → **sıralı fallback** (2996MB); çıktılar referansla birebir; **cevap birleştirme `best`** (logprob-güven) seçiyor; **model havuzu + LRU** ile tekrar trafik 2,74×; **uçtan uca servis** (`lcsa/service.py` + `serve.py`) 3 birimle 8 gerçek sorguda çalışıyor (§13) | Uçtan uca çok-birimli cevap; latency modeli |
| **M5** | **Tam-parametre >VRAM eğitim**: CPU Adam offload + faz kümeleri + streaming | 🔄 3B: SGD −0,0962 / **Adafactor −0,1792 (150 adım)** / **−0,1818 (300 adım + kosinüs; doygun)**; 6,6–7,7s/adım; peak 2,43–4,93GB; paged serving OK; bf16 kuantizasyon duvarı teşhis; fp32 master 0.5B'de OK (3B: RAM sınırı); **profil: adımın %71'i PCIe**; **`--buffers` 1,26× (5,20s), parite birebir**; `--prefetch`/`--pinned` kazançsız; **`--double-buffer` 3B SGD 5,28→4,40s (1,20×), Adafactor 6,69→5,91s (1,13×), parite birebir, +0,17GB peak** | D2H writeback'ini de çift-tamponlamak |
| **M6** | **int8/int4 ağırlık**: transfer hacmi 2–4× düşer | 🔄 int8 4,18 tok/s (örnek birebir, %0,39 rel err); int4 3,50 + bozuk örnek (negatif) | int8→int4 paged tok/s; kalite kaybı ölçümü |
| **M7** | **Çok-GPU / iletişim**: P9b sonucu + gerçek çok-GPU testi | 🟡 | Sync trafiği −%50, kalite paritesi |
| **M8** | **QLoRA / NF4 adaptör**: 6GB'da daha büyük tabanı adapte et | 🟡 3B PoC OK: NF4+LoRA **%1,73** eğitilebilir, peak **4,25GB**, **2,56s/adım** (seq512), WT2 CE **2,5154→2,3188** (10 adım); merge (`exp50`, 6,17GB) + int8 paged **2,74 tok/s**, peak 4731MB; SFT (`exp51`, Dolly 200 adım/512 satır) eval loss **2,4405→1,9040 (−0,54)**, peak 3,90GB, 1,74s/adım (20 adım smoke: −0,70); SFT merge+int8 paged **2,84 tok/s**, peak 4733MB; **HF 7B denemesi peak 8,4–8,7GB → spill; kendi streaming QLoRA'mız (exp52) 7B'yi 6GB'da eğitiyor: peak 4,90GB, 5,9s/adım, CE 2,2304→2,1907 (10 adım), LoRA %0,116; SFT (exp53, Dolly 100 adım/512 satır) eval loss **2,6850→1,8710 (−0,81)**, peak **5,78GB**, **5,99s/adım**; **200 adım/mlp: 2,4782→1,5670 (−0,91), peak 5,66GB, 5,25s/adım, LoRA 20,19M; 300 adım/mlp: 2,6185→1,7634 (−0,86), peak 5,52GB, 5,55s/adım**; **14B streaming: CPU-embed fix ile peak 3,90GB, 8,5s/adım → 6GB'a sığıyor (embed GPU'dayken 7,01GB spill idi); 20 adımda eval CE 2,3270→2,2215 (−0,105), peak sabit 3,90GB, 8,63s/adım; 200 adım Dolly SFT: eval loss 2,7531→1,4771 (−1,28), peak 4,14GB, 8,77s/adım; disk'ten mmap akış (exp54) RAM yoluyla birebir parite (3B: eval CE 2,73626, loss 3,347981)** | 32B disk-streaming run; lm_head'i de CPU/chunk'a almak |

## Ürün katmanı (20.10)

`lcsa/service.py` + `serve.py`: M1/M4/M5 parçalarını **tek serviste** birleştirir —
sorgu → rota → VRAM planı → paralel/sıralı yürütme → birleştirme; stdlib HTTP API
ve tarayıcı demo arayüzü. 8 gerçek sorgu (`exp22b`): medyan **5,42 s**, sıcak
fast 0,6–1,5 s, deep (3B int8 paged) 3,5–8 s, peak **4,65 GB** (taşma yok).
İlk koşuda bulunan paged/dense VRAM çakışması `release_gpu()` ile düzeltildi.
Eşit sıcak başlangıçta paralel gruplama **1,30×** (0,75 vs 0,98 s); ilk koşudaki
0,88× reload asimetrisiydi. 2,0GB bütçede 1.5B (3,3GB) yine çalışır ve servis
`warnings` döner. **Deep birimi kendi 3B checkpoint'imizi servis ediyor**
(`qwen3b_full_adaf_t300_best.pt`, mmap state_dict → int8 paged): held-out WT2 CE
**2,611 → 2,370 (−%9,2)**, aynı gecikme; 24 OOD probda toplam **+0,061** CE
(TR +0,10, fact +0,045, code −0,032; top-1 sabit) — alan içi kazanç, alan dışı
küçük maliyet. `POST /v1/preload`, `units` ile açık seçim, istek günlüğü,
LRU cache, `/metrics`, deneysel `messages` (çok-tur), OpenAI uyumlu
`/v1/models` + `/v1/chat/completions` ve UI geçmişi var. **Chat birimi**
(Qwen2.5-1.5B-Instruct, model varsa otomatik): 24 probda exact **%25 → %75**,
TR talimat **%20 → %60**; TR system promptuyla keyword **%82**, exact **%100**,
TR fact **%83** (n=24). **60 problu kontrol (`exp26`):** keyword **%82,7**,
exact **%84,6**, TR fact **%75**, liste **%62,5**; kalan hatalar raporda.
Çok-tur hatırlama `exp25c` 5 senaryoda **2/5**; `exp29` önekli **4/5**, ad
sorusu **3/4** — yapı + 5 şablonlu bellek var, genel hafıza yok.
**Token streaming** (`/v1/stream`): sıcak TTFT **0,249 s** fast / **0,449 s**
deep; çoklu birim tek `final`. **OpenAI smoke (`exp28`):** stream TTFT
**0,443 s**, toplam **2,697 s**, `[DONE]` var. CPU-only testler **40/40**.
Detay: `SERVICE_RAPOR.md` §3–5. Açık: insan/LLM judge, öğrenilmiş router,
çok-tur hafıza, çoklu birim streaming/iptal, auth/CORS, speculative decoding.

## Mevcut temel (kanıt)

- **P1**: 3B model 6GB'de block-streaming LoRA eğitimi, 1.40GB peak; parity testi geçti.
- **P8**: Küme-fazlı eğitim ≈ tam eğitim; 2 seed × 2 partition; adım süresi −%6–19.
- **MoE-lite**: E=4 top-1; router öngörülebilir (entropi 0.22–0.65; pozisyon NMI ~0).
- **P4**: anytime derinlik L1–L4 aynı kalite bandında.
- **P6**: naif no-sync kaliteyi bozuyor → yapılı senkron şart.
- **P9b**: phase-DDP koşuyor (aktif küme gradyanı senkronu).
- **M5**: 3,09B **tam-parametre** 6GB'de güncellendi (blok-streaming; grad paritesi 0.5B'de rel_rmse %0,2–2,3, cos≈1). **SGD temiz 150 adım: CE 2,5034→2,4072 (−0,0962), 6,64s/adım, peak 2,433GB**. **Adafactor (lr 1e-5, 150 adım): CE 2,5034→2,3242 (−0,1792 = SGD'nin 1,86×'i), 7,67s/adım, peak 4,93GB**; iki checkpoint de paged int8 ile 6GB'de sunuldu (3,93 / 3,88 tok/s, tutarlı metin). **bf16 kuantizasyon duvarı** ölçüldü (yerinde SGD adımları ULP altında → büyük ağırlıklar donuk); fp32 master opsiyonu 0.5B'de ~%10 CE kazancı, 3B'de +12,3GB RAM istediği için koşulamadı. **300 adım + kosinüs decay: −0,1818; doygunluk (150→300 arası yalnızca +0,005) → kazanç ilk ~100 adımda.** Eski kirli koşu: 100 adım −0,0375 (ayrı etiket). **Transfer profili (3 adım):** adımın **%71'i PCIe** (H2D 2,44 s + D2H 2,26 s; 4 geçiş ≈24,7GB → ~5,3GB/s), compute 1,81 s; Adafactor update 0,29→1,50 s; **`--prefetch` kazanç sağlamıyor** (pageable kopya ana thread'i bloke ediyor), pinned çift-tampon uygulanmadı. **`--buffers`** (blok başına GPU tamponu, forward write-back yok): 3B **6,56 → 5,20 s/adım (1,26×)**, loss'lar birebir; `--pinned` tek başına negatif (staging 1,55 s).
- **M2**: MoE expert paging parite ile doğrulandı — üç modda CE **1,37843**; pass'ta 15,4MB; hit %0,6. **Ölçek (E=16 → E=64):** talep 34,27 → 53,98 MB/pass; prefetch (pf2) **8,42 / 15,59 MB/pass** (4,07× / 3,46× azalma), hit **%75,5 / %71,1**; VRAM tasarrufu paged lehine 3,35× / 10,7×; E=64'te prefetch hızı paged'i geçiyor (12,13 vs 11,75 tok/s) — kopma eşiği E≈16–64.
- **M3**: Geçiş-matrisi prefetch (2 slot): talep kopyası **12,4× az** (15,368→1,24MB/pass), hit %0,58→**%92**, prefetch isabeti **%92,7** (626 kopya/580 kullanım); CE ve örnekler birebir; LRU2 %5,1'de kaldı → öngörü gerçek yapı yakalıyor.
- **M1/M6**: paged int8 3B **4,18 tok/s** (bf16 2,75) örnek birebir; **llama.cpp Q4 kıyası**: CPU 8t **12,4 t/s**, CUDA ngl99 **57,6 t/s** → dürüst çerçeve: 3B Q4 zaten 6GB'a sığar; paging değeri >VRAM eğitim + kalite kayıpsızlık + Q4'te sığmayan modeller.

## Sınırlar

- Veri: 466M token (TinyStories) → büyük modellerde "sistem demosu", frontier kalite yarışı değil.
- GPU: 6GB, PCIe ~8–16GB/s → paged decode hızı transferle sınırlı; 4-bit + resident oranı kritik.
- Dense modelde "soruya göre birim" yok; MoE/erken çıkış/adapter/retrieval ile kurulacak.
