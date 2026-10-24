# BASARILAR.md — Tüm Deneylerin Başarı / Sonuç Özeti

> **Kaynak:** `PROGRESS.md` deney günlüğü (15–18.10.2026) + `RAPOR.md` + `P1–P6_RAPOR.md` + `results/results.jsonl` (365 satır) + oturum konuşmaları.
> **Sınıflar:** ✅ kanıtlı başarı · 🟡 kısmi / karar başarısı · ➖ negatif ama öğretici.
> Derleme: 18.10.2026 gece (kullanıcı isteği: “başarılı olan deneyler hangileri — hepsini yaz”).

---

## 1) Çekirdek — LCGA seyrek attention (Qwen2.5‑0.5B, donuk taban + öğrenilen router)

Soru: O(N²) dense attention yerine **öğrenilebilir blok-seyrek** ile kaliteyi koruyup hızlanmak.

| # | Deney | Sonuç (rakam) | Sınıf |
|---|---|---|---|
| 1.1 | **Null-key bug’ı bulundu + kapatıldı**; temiz konfig | ρ=1.0 PPL **14.559 = dense birebir** → formülasyon sadık (bug’lıyken 21.84) | ✅ |
| 1.2 | **Uzun bağlam trendi** (v1, exp6) | T=4096: ρ=0.25 **+%26**, ρ=0.5 **+%9.5**; recall 0.778→**0.826** | ✅ |
| 1.3 | **Stage-2 (v2: LoRA+LM+KL)** | wt2 kalite farkı **+%52.7 → +%29.9** (dense 12.881 / sparse 16.732); KL 0.43→0.29 | ✅ |
| 1.4 | **B fazı (v3: LoRA LR 1e-4)** | wt2 dense **12.456** / sparse **16.559** (en iyi); **OOD unutma geri alındı**: TinyStories sparse 17.221→**9.158**, dense 10.370→**6.090** (v1: 7.963/5.557) | ✅ |
| 1.5 | **Attribution (v2ctl dense-kontrol)** | Gap azalmasının **%83 jenerik / %17 seyreklik-özgü** (iki protokolde tutarlı) | ✅ |
| 1.6 | **v2 bağlam eğrisi** | T=8k’da kalite farkı **+%7.5** (1k’da +%27.8); recall **0.868** | ✅ |
| 1.7 | **Router ucuzlatma (A fazı)** | @1024 **44.8 → 6.5 ms (~7x)**; A1 parity **10/10 bitwise**; A2 CUDA-graph (kapılı, fallback’li) | ✅ |
| 1.8 | **Uçtan uca hız (final, v2_a2)** | 120.7/230.7/446.6/928.8/1464.0/2041.0 ms (1k…16k) → **0.92–1.25x**; **crossover 8k → ≈5k** | ✅ |
| 1.9 | **Decode (C fazı)** | Tüm adım tek CUDA graph: dense **13.0–24.2 ms/adım bitwise parity**, sparse 13.5–19.1; eager’a karşı **4–9x**; sparse, dense-graph’dan 8k’da **1.27x** hızlı | ✅ |
| 1.10 | **Ölçüm tuzakları bulundu/kapatıldı** | GQA/MATH fallback (~100x sahte), autograd+LoRA taşması (6–30x sahte), WDDM enqueue artefaktı, decode slot-mask bug’ı | ✅ |

**Alt sonuç:** “Öğrenilmiş ilişki (router) + donuk taban + küçük eğitilebilir set (router/LoRA)” ile **kalite büyük ölçüde korunarak** hesap düşürüldü ve hız hedefi (uzun bağlamda dense’ten hızlı) kanıtlandı.

---

## 2) P1 — Parça parça (block-streaming) eğitim

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 2.1 | **6GB kartta 3B model** | naive **OOM** ↔ stream **1.40GB sabit peak**, 60 adımda held-out CE **2.3302→2.2143** (−0.116), adapter r16 (7.37M param) | ✅ |
| 2.2 | **1.5B**: naive 10.89GB/13.5s ↔ stream **1.09GB/4.6s** | ~10x az VRAM, ~3x hızlı; parity maks fark ~9e‑4 (0.5B’de 6.8e‑4) | ✅ |
| 2.3 | Sınır | VRAM’e sığan rejimde (0.5B) naive hâlâ daha hızlı | 🟡 |

Kanıt: `P1_RAPOR.md`, `results/exp8_*.json`, ckpt `p1_qwen3b_lora_r16_t60.pt`.

---

## 3) P2 — Byte-level (tokenizer’sız) prototip

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 3.1 | **Flat byte transformer** (~5M param) | test **1.6452 bpb** (valid 1.6582); örnekler akıcı | ✅ |
| 3.2 | Patch (BLT-lite) | test 2.0513 bpb **ama adım başı ~1.7x hızlı** (150.1 vs 250.7 ms) | 🟡 |
| 3.3 | **Süre doğrulaması** | iç loglardaki 16.55 ms WDDM enqueue artefaktıydı; gerçek 250.7 ms → bağımsız doğrulandı (`exp10_timing_verify.json`) | ✅ |

---

## 4) P3 — Kendi mini-LM (TinyStories + kendi BPE)

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 4.1 | 17.31M param, sıfırdan, kendi 8k BPE — **18.000 adım TAMAM** (19.10 13:37) | **val_loss 1.3890 / ppl 4.01 / bpb 0.4892** (442.4M token ≈0.95 epoch); a3_ctrl8k kontrolünden **−7.1% bpb** (−0.0372); 655 ms/adım @5.90G, 37.4k tok/s; akıcı hikâye örnekleri; kesintisiz resume zinciri (3000→8000→10000→18000, kayıpsız) | ✅ |

Kanıt: `P3_RAPOR.md`, `results/p3_resume.log`, ckpt `p3_best.pt` (bpb, ts_valid'in decode edilmiş gerçek baytıyla; a2/a3 ile aynı metin akışı doğrulandı).

---

## 5) P5 — “Tokenizer 2” arenası (karar)

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 5.1 | Aşama-1 (CPU sıkıştırma) | Süper-kelime aynı vocab’da **+%32–78** daha iyi sıkıştırma (16k: 7.474 byte/token vs 4.158) | ✅ |
| 5.2 | **Karar arenası 1x + 2x (bpb)** | **Klasik kazandı**: 1x ctrl8k **0.5612** < sw16k 0.5769 ≈ wb16k 0.5776 < sw8k 0.5890; **2x**: ctrl8k **0.5264** < sw16k 0.5422 < wb16k 0.5445 < sw8k 0.5529 — fark kapanmıyor | ✅ |
| 5.3 | Hız modu (süper-kelime) | Aynı 512-token bağlamda **1.8x metin**, ~%3 kalite bedeli → “hızlı inference” seçeneği | 🟡 |

Kanıt: `P5_RAPOR.md`, `results/a2_*.log` + `a3_*.log`.

---

## 6) P4 — LoopLM (döngülü derinlik / “uzun düşünme” denemeleri)

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 6.1 | **Parametre verimliliği** | loop2x4 **10.2M** → 0.5761; flat4 (10.2M) 0.5889’u geçti (%2.2); flat8 (17.3M) 0.5612’ye **%41 az paramla %2.7** yaklaştı | ✅ |
| 6.2 | **Anytime derinlik (P4.2)** | L~U{1..4} eğitim: sweep **L1 0.6029 / L2 0.5840 / L4 0.5852 / L8 0.6066** → L=1..4 ~%3 bantta | ✅ |
| 6.3 | “Uzun düşünme” | Sabit-eğitimli model derinliğe kilitleniyor (L=4 +%18, L=8 +%76); rastgele eğitim kilidi açıyor ama **fazla döngü kalite kazandırmıyor** (L4≈L2) | ➖ |

Kanıt: `P4_RAPOR.md`, `results/p4_*.log`, ckpt `p4_loop2x4_best.pt` / `p4_looprand14_best.pt`.

---

## 7) P6 — Senkron maliyeti (kullanıcının “100 GPU’ya böl, transfer olmasın” fikri)

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 7.1 | 4 işçi, aynı bütçe; full / local250 / nosync / soup | **full 0.6190 < local250 0.7117 (+%15.0) < nosync 0.7310 (+%18.1); soup 0.8820 (+%42.5)** — iletişimi kesmek tek modeli kaybettiriyor | ➖ |

Kanıt: `P6_RAPOR.md`, `results/e13_*.log`. (Tek GPU’da ölçüldü → “algoritmik kalite bedeli”; ağ trafiği ayrı ölçülmedi, dürüst sınır.)

---

## 8) P8 — İlgili parametre kümeleri × parçalı eğitim (19.10)

Sorular: (1) gradyan-etkileşimi parametreleri anlamlı kümelere ayırıyor mu? (2) kümeleri fazlarla sırayla eğitmek tam-eğitime göre ne kaybettirir; "ilgili" olmak rastgele kümeye göre avantaj mı?

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 8.1 | Gradyan-etkileşim kümelemesi (corr) | 18 birim → dengeli min-cut: **C1 = 6 orta-katman MLP'si (L1–L6)**, kalan 12 birim (tok+attention'lar+uç MLP'ler+norm) C0; kesim skoru related **21.19** vs aynı profilli rastgele **33.98** | ✅ (yöntem çalışıyor) |
| 8.2 | Fazlı parçalı eğitim (s3000→6000, `--phase 300`, tam lr) | @6000: **rel 1.5801 / 1.5773** (s1337/s1338) — kontrol 1.5846'nın önünde (2/2); rnd 1.5920 / 1.5861 / **1.5924** (2. partition); **rel−rnd = −0.0119 / −0.0088 / −0.0123** (2 seed × 2 partition, hepsi rel lehine); adım ~%6–8 hızlı (C1 fazında 529 ms vs 655) | ✅ replikasyon (2 seed + 2 partition; kalan: küçük farklar, tek ölçek) |
| 8.3 | Protokol hatası teşhisi (v1) | adım-başına küme alternasyonu tam lr'de **diverge** (1.71→3.25); 200-adım probe üçlüsü: `full` ✓ / `step_alt` ✗ (yeniden üretti) / `phase300` ✓ → hata mekanizması kanıtlandı ve düzeltildi | ✅ (süreç kazanımı) |

Kanıt: `P8_RAPOR.md`, `results/p8_rel.log`, `results/p8_rnd.log`, `results/p8v1_rel_aborted.log`, `results.jsonl`.

---

## 9) MoE-lite (exp15) — E=4 top-1 router (19.10)

Soru: parametre düzeyinde uzman yönlendirme (equalm total param, ~1/4 aktif MLP) dense'e göre ne kazandırır?

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 9.1 | Eşit toplam param (17,32M), aktif ≈10,25M/token, a2 protokolü | Adil kıyas (aynı pencereler; kontroller birebir üretildi): **0.5791 bpb**; dense-17,3M 0.5612'yi **geçemedi** (%3,2 rel.), flat4 (10,2M) 0.5889'u **geçti**, loop2x4 0.5761'in hafif gerisi | 🟡 kısmi |
| 9.2 | Router sağlığı | aux (α=0.01): tüm katmanlarda ~0,25 dengeli kullanım; çökme yok; 506 ms/adım (dense 480) | ✅ |
| 9.3 | Uzmanlaşma analizi (24.576 token) | Token-tipi uzmanlaşması net (noktalama / özne zamiri / nadir-uzun token); token-içi entropi 0,22–0,65 (shuffle 0,99); pozisyon NMI ≈0 → **leksem-hash davranışı**; anlamsal uzmanlaşma yok (bu ölçekte) | ✅ (betimleyici) |

Kanıt: `MoE_RAPOR.md`, `results/moe4.log`, ckpt `moe4_best.pt`, `results.jsonl` (reeval satırları).

---

## 10) Süreç ve altyapı başarıları

- ✅ **WMI ile oturumdan bağımsız uzun koşular** (PC/oturum kesintilerine dayanıklı: runner.out + DONE işaretleri).
- ✅ **365 satır `results.jsonl`** + raporlar (`RAPOR.md`, P1–P6) + `PROGRESS.md` devir günlüğü + `show_rows.py`/`summarize_results.py`/`make_plots.py` araçları.
- ✅ **MIT lisans + README** (paketleme).
- ✅ **Doğrulama disiplini**: parity testleri (bitwise), CPU/GPU smoke kapıları, selftest’ler, bağımsız timing doğrulamaları, ~10 bulunmuş ölçüm tuzağı.

---

## En büyük 5 başarı (önem sırası)

1. **LCGA hız hikâyesi kanıtlı:** crossover ≈5k; 16k’da **1.25x hızlı**; decode **4–9x** (CUDA graph, bitwise parity) — kalite v3’te büyük ölçüde korundu (16.559 vs dense 12.456, uzun bağlamda fark daralıyor).
2. **Az eğitilebilir parametreyle adaptasyon:** donuk taban + router/LoRA; v2→v3 kalite + OOD; attribution’la dürüst ayrıştırma (%83/%17).
3. **P1 parça-parça eğitim:** 6GB kartta **3B** model eğitildi (1.40GB peak), parity kanıtlı.
4. **P4: az parametreyle yakın zeka** (10.2M ≈ 17.3M’in %2.7’si) + **anytime** derinlik esnekliği.
5. **P5 kararı bilimsel olarak kapandı** (klasik tokenizer kalıyor; süper-kelime = hız modu) + **P2/P3 tohumları** (byte-level 1.6452 bpb; kendi mini-LM **bpb 0.4892 / ppl 4.01**, 18k adım).

## Dürüst negatifler (öğretici, kayıt için)

- v1 router eğitimi seçim kalitesini/PPL’i değiştirmedi (metrik dersi: kayıp değil recall/oracle’a bak).
- Coarse yol temiz konfigde saf zarar (kapatıldı).
- **“Bedava uzun düşünme” yok** (P4.1+P4.2) — döngü = verimlilik + esneklik.
- Süper-kelime kalite (bpb) kazandırmadı — hız modu olarak değerli.
- Senkron kesme (nosync/local250/soup) tek modeli kaybettiriyor.
- P1 sığan-rejimde naive hızlı; patch (P2) hızlı ama flat’ten zayıf; decode naive harness bug’ı (graph ile çözüldü).
- **Sorgu bazlı paralel gruplama ilk ölçümde yavaş göründü (0,88×) — bu bir reload asimetrisiydi**; eşit sıcak başlangıçta paralel **1,30×** (0,75 vs 0,98 s). Düzeltme kayda geçti.
- **Bütçe tek birimin sığmasını garanti etmez**: 2,0GB bütçede 1.5B (3,3GB) yine çalışır; servis `warnings` döner (plan gruplama içindir, kabul/ret değil).
- **Base Qwen modelleri sohbet beklentisini karşılamıyor** (selamlama probu “- 1000000000”); instruct model gerekli.

---

*Bu dosya, soru üzerine tüm fazların tarafsız derlemesidir; rakamlar yukarıda listelenen log/rapor/JSONL dosyalarından alınmıştır.*

---

## 11) M1 — Paged inference: 3B model 6GB kartta (19.10)

**Soru:** VRAM'e sığmayan modeli (Qwen2.5-3B bf16 = 6,17GB) 6GB kartta katman akıtarak çalıştırmak; matematik birebir mi?

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 11.1 | **Parite** (Qwen2.5-0.5B, fp32, GPU vs HF CPU) | max \|Δlogit\| **3,26e-5**, argmax **1,000** | ✅ |
| 11.2 | **3B / 6GB** (26 resident + 10 streamed) | tutarlı üretim; peak 4659MB; örnek "…Berlin." | ✅ |
| 11.3 | **Hız eğrisi** (pinned) | k=26 **2,75 tok/s**; k=20 1,98; k=14 1,57 | 🟡 |
| 11.4 | **Saf kopya** | **5,51 GB/s** (PCIe Gen4 x16 max; idle Gen1 x4) | ✅ ölçüm |
| 11.5 | **CPU-only bf16 baseline** | **3,01 tok/s** → paging bf16'da CPU'dan hızlı değil | ➖ dürüst |

**Yorum:** Paging primitifi ve doğruluk tamam; darboğaz token başına 1,44GB PCIe transferi. Değer, **int8/int4 (M6)**, uzun prompt/prefill ve daha büyük modellerde ortaya çıkacak. Kod: `lcsa/paged_qwen.py`; rapor: `M1_RAPOR.md`.

**Yeni dürüst negatifler:**
- P9b phase-DDP 1000 adımda full'e yetişmedi (+%12,1 bpb); "yarı senkron = aynı kalite" iddiası desteklenmiyor.
- Paged bf16 3B, aynı makinede CPU-only'den yavaş (2,75 vs 3,01 tok/s); quantization şart.

---

## 12) M5 + M2 + M6 — Tam-parametre eğitim, expert paging, pazar kıyası (19.10)

**Soru:** (a) VRAM'e sığmayan modeli 6GB'de **eğitebiliyor muyuz** (LoRA'sız, tam parametre)?
(b) MoE uzmanlarını yalnız gerektiğinde yükleyebiliyor muyuz? (c) Bu iş pazar araçlarına göre nerede?

### 12.1 M5 — 3B tam-parametre eğitim (blok-streaming SGD)

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 12.1a | 3B (3,0859B) **150 adım**, seq 512, WT2 | CE **2,5034 → 2,4072 (−0,0962)**; medyan **6,64 s/adım**; peak **2,433GB** | ✅ |
| 12.1b | Gradyan paritesi (0.5B, naive HF vs streaming) | loss birebir (fark 2,4e-7); rel_rmse **%0,2–2,3**; cos≈1 | ✅ |
| 12.1c | Uçtan uca: checkpoint → paged int8 6GB | tutarlı örnek ("…Berlin."); **3,93 tok/s**; peak 4735MB | ✅ |
| 12.1d | Chunked CE gradyan ölçeği hatası (T/chunk ×) | bulundu, düzeltildi; eski 100 adımlık koşu kirli işaretlendi (silinmedi) | ➖ dürüst kayıt |

### 12.2 M2 — MoE expert paging

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 12.2a | E=4 top-1, uzmanlar CPU'da, 1 slot | CE birebir **1,37843**; **36,1 tok/s** (resident 39,5); pass'ta 15,4MB | ✅ |
| 12.2b | LRU 1 / LRU 2 / preload karşılaştırması | aynı CE; hit %0,6–5,1; preload resident baseline | ✅ |
| 12.2c | **M3 prefetch** (2 slot + geçiş matrisi, ısınma 128 token) | talep kopyası **15,4→1,24 MB/pass (12,4×)**; hit **%92**; prefetch isabeti **%92,7**; parite birebir | ✅ |
| 12.2d | **Ölçek: E=16** (45,7M, 2000 adım) | CE beş modda **1,57647** (parite); talep 34,27 MB/pass; pf2 **8,42 MB/pass (4,07×)**, hit **%75,5**, kullanım %73; VRAM paged 28,8MB vs resident 96,3MB (**3,35×**) | ✅ |
| 12.2e | **Ölçek: E=64** (159,1M, 1500/2000 adım — kesintili) | CE beş modda **1,71569** (parite); talep **53,98 MB/pass**; pf2 **15,59 MB/pass (3,46×)**, hit **%71,1**, kullanım %68; VRAM paged 29,1MB vs resident 312,6MB (**10,7×**) | ✅ |

### 12.3 M6 + pazar kıyası (llama.cpp b11734)

| Yöntem | tg tok/s | pp64 | Not |
|---|---|---|---|
| llama.cpp Q4 **CUDA** (ngl99) | **57,6** | 1691 | 3B Q4 = 1,79GB → 6GB'a sığar |
| llama.cpp Q4 CPU (8t) | **12,4** | 108 | i5-13420H |
| llama.cpp Q4 CPU (6t) | 8,5–15,1 | 82–116 | çalıştırma/termal varyans |
| Biz: paged **int8** (GPU) | 3,9–4,2 | — | bf16 kalite kayıpsız; 4,7GB peak |
| Biz: paged bf16 (GPU) | 2,75 | — | kayıpsız |
| HF CPU bf16 | 3,01 | — | aynı makine |

**Yorum:** Tam-parametre eğitim **çalışıyor ve ölçülü öğreniyor** (temiz koşu, 150 adımda −0,0962 CE).
Expert paging mekanizması parite ile doğrulandı; ölçek iddiası E=16/64 sentetik testi bekliyor.
Inference'ta dürüst tablo: küçük/quantize modellerde llama.cpp açık ara önde; bizim farkımız
(1) **>VRAM eğitim**, (2) bf16/int8 **kalite kayıpsızlığı**, (3) Q4'te de sığmayan modellerde birim-akıtma.

### 12.4 M4 — Çok-birimli scheduler (0.5B + 1.5B, 24 token, greedy)

| # | Senaryo | Sonuç | Sınıf |
|---|---|---|---|
| 12.4a | Bütçe 5,0GB → tek grup | **paralel üretim 2,31s** vs referans 4,01s (**1,74×**); peak 3931MB; çıktı birebir | ✅ |
| 12.4b | Bütçe 2,5GB → iki grup | **sıralı fallback** çalıştı; peak 2996MB; çıktı birebir | ✅ |
| 12.4c | **Cevap birleştirme `best`** (ort. token logprob) | aynı soruda 1.5B seçildi (−0,133 vs −0,167); hikâye demo: farklı çıktılarda 1.5B devamı seçildi | ✅ |
| 12.4d | **Model havuzu + LRU** (`lcsa/pool.py`) | 3 istek: 25,16s → **9,17s (2,74×)**; tekrar istekte yükleme **0,00s**; dar bütçede evict+reload 0,35s (disk 1,40s) | ✅ |

**Yorum:** Scheduler kararı (paralel/sıralı) VRAM bütçesine göre doğru veriyor; paralel yürütme
sonucu değiştirmiyor; havuz tekrar-trafikte yükleme maliyetini siliyor. Kalan boşluk: otomatik
VRAM tahmini, sorguya göre birim seçimi; birleştirmede oylama/rerank yok (logprob-güven seçimi);
havuzda evict edilenler host RAM'de tutuluyor.

### 12.5 M5b — Adafactor + bf16 kuantizasyon duvarı + fp32 master (19.10 akşam)

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 12.5a | bf16 yerinde SGD adım büyüklüğü (0.5B) | uygulanan delta RMS 5,5e-9; elemanların **%0,0–0,1**'i değişiyor; norm ağırlıkları donuk → **kuantizasyon duvarı** | ➖ teşhis |
| 12.5b | Adafactor lr taraması (0.5B, 30 adım) | 1e-6 −0,011; **1e-5 −0,205**; 3e-5 −0,123; 1e-4 +5,97; 3e-4 +7,86 | ✅ |
| 12.5c | **3B Adafactor 150 adım** (lr 1e-5, seq 512) | CE **2,5034 → 2,3242 (−0,1792)** = SGD'nin **1,86×**; 7,67 s/adım; peak **4,93GB** | ✅ |
| 12.5d | Uçtan uca: Adafactor checkpoint → paged int8 | **3,88 tok/s**, peak 4735MB, tutarlı metin ("…Paris. Paris is the most visited city…") | ✅ |
| 12.5e | fp32 master (CPU, 0.5B, 30 adım) | SGD −0,0939→**−0,1049**; Adafactor 1e-5 −0,2052→**−0,2196**; Adafactor 1e-4 +5,97→**+0,81**; maliyet ~2× adım, +4GB/1B param RAM | ✅ 0.5B |
| 12.5f | fp32 master 3B | **koşulamadı**: +12,3GB RAM gerekiyor, makinede ~8,8GB boştu | ➖ sınır |
| 12.5g | 3B Adafactor **300 adım + kosinüs decay** | CE 2,5034 → **2,3217 (−0,1818)**; **doygunluk**: 150→300 adım arası yalnızca +0,005; 7,66 s/adım; en iyi ckpt `qwen3b_full_adaf_t300_best.pt` | ✅ |
| 12.5h | 3B **transfer profili** (`--profile`, 3 adım) | Adımın **%71'i PCIe**: H2D 2,44 s + D2H 2,26 s (4 geçiş ≈24,7GB, ~5,3GB/s); compute yalnız 1,81 s. Adafactor update 0,29→**1,50 s** (+1,21 s). **`--prefetch` kazanç yok** (6,56→6,57 s; pageable kopya ana thread'i bloke ediyor) | ✅ teşhis |
| 12.5i | **`--buffers`** (blok başına GPU tamponu; forward write-back yok) | 3B: **6,56 → 5,20 s/adım (1,26×)**, D2H 2,26→1,06; peak 2,43→2,55GB; **loss'lar birebir aynı**. 0.5B Adafactor 6 adım: 1,27→1,02 s/adım, altı loss da birebir. **`--pinned` tek başına negatif** (5,68 s; staging CPU kopyası 1,55 s) | ✅ 3B |

**Yorum:** bf16 yerinde güncelleme, adım büyüklüğünü ULP'ye yuvarlayıp çoğu ağırlığı donduruyor;
Adafactor'ün normalize güncellemesi lr ham gradyan rejimine ölçeklenince (1e-5) SGD'yi **1,86×**
geçiyor. fp32 master kuantizasyon kaybını kaldırıyor (+%10 CE) ama 3B'de RAM'e sığmıyor.

**Yeni dürüst negatifler:**
- Paged int8 3B (4,18) < llama.cpp Q4 CPU 8t (12,4) ve CUDA (57,6); "paging hızlı inference" iddiası 3B sınıfında geçerli değil.
- int4 paging örnek bozuyor (%7,1 rel err) — şimdilik rafa kalktı.
- Gradyan ölçek hatası ilk 3B koşusunu kirletti (kayıt: silinmedi, ayrı etiket, `M5_RAPOR.md` §3b).

---

## 13) Ürün kabuğu — uçtan uca servis (20.10)

**Soru:** Araştırma parçaları (paged motor, havuz, scheduler, merge) tek bir
serviste uçtan uca çalışıyor mu; gerçek sorgularda gecikme ve VRAM ne?

| # | Deney | Sonuç | Sınıf |
|---|---|---|---|
| 13a | Servis çekirdeği (`lcsa/service.py`) | rota → plan → yürütme → birleştirme; 3 çekirdek birim: 0.5B (dense), 1.5B (dense), 3B (**int8 paged**) + model varsa `chat` 1.5B-Instruct; rota gerekçesi her cevapta döner | ✅ |
| 13b | HTTP + demo (`serve.py`, stdlib) | `/`, `/health`, `/units`, `/stats`, `/metrics`, `POST /v1/answer`, `/v1/stream`, `/v1/preload`; tarayıcı arayüzü; ek bağımlılık yok | ✅ |
| 13c | **Gerçek sorgu seti** (`exp22b`, 8 sorgu, 12 token) | **8/8 hatasız**; medyan **5,42 s**; sıcak fast **1,54 s**; fast+balanced 3,02–6,05 s; deep 4,80–17,31 s; peak **4646MB** (bütçe 5,0GB, taşma yok) | ✅ |
| 13d | Paralel vs sıralı (eşit sıcak, `exp22d`, n=3) | **0,754 s vs 0,980 s = 1,30×** (üretim de 1,30×); ilk koşudaki 0,88× reload asimetrisiydi (paralel istek soğuktu) | ✅ |
| 13e | Paged motor + dense VRAM çakışması | İlk koşu **peak 8555MB** (shared memory’ye taşma), aynı istek 18,80 s; `PagedQwen.release_gpu()` sonrası **4646MB / 3,02 s** | ✅ bulundu ve düzeltildi |
| 13f | Deep motor önbelleği | ilk 6,56 s → ikinci **3,56 s** (motor GPU’da kalıyor; dense’e geçerken bellek bırakıyor) | ✅ |
| 13g | Bütçeyi aşan birim uyarısı | bütçe 2,0GB + 1.5B (3,3GB) → `warnings`: “unit balanced needs 3.3GB > budget 2.0GB”; birim yalnız çalışır (peak ~4,0GB) | ✅ şeffaflık |
| 13h | Kendi checkpoint serviste (`exp23`) | `deep` birimi `qwen3b_full_adaf_t300_best.pt` ile (mmap state_dict → int8 paged); yükleme 6,2 s, peak **4897MB**; `/health`, `/units` checkpoint adını döner | ✅ |
| 13i | **Held-out WT2 CE** | **2,611 → 2,370 (−%9,2)**, aynı paged motor ve aynı gecikme (8 token: 2,63 → 2,51 s); 4×256 chunk, tek tohum | ✅ ihtiyatlı |
| 13j | OOD prob (24 prob) | tümü CE **+0,061**; TR +0,102 (n=12), fact +0,045 (n=8), code −0,032 (n=4); top-1 değişmedi → alan içi kazanç, alan dışı küçük maliyet | ✅ dürüst sınır |
| 13k | Preload + birim seçimi + günlük | fast preload 4,94 s (soğuk), sıcak fast **0,537 s**; `units:["fast","deep"]` → 2 grup **9,128 s**, peak **4763MB**; geçersiz birim HTTP 400; `results/service_requests.jsonl` | ✅ |
| 13l | CPU-only servis testleri | **40/40** (26 servis + 14 HTTP/OpenAI: rota/plan/merge/cache/mesaj/SSE/400/models/chat completions) | ✅ |
| 13m | **Chat birimi** (Qwen2.5-1.5B-Instruct, `exp25`) | 24 kısa prob: keyword isabet **%54,5 → %68,2**, exact **%25 → %75**, TR talimat **%20 → %60**, EN fact **%83 → %100**; TR fact %33 (aynı), liste 0/2; üretim 1,62 → **0,90 s**, peak 2959MB | ✅ dürüst sınır |
| 13n | **Token streaming** (`/v1/stream`) | sıcak TTFT **fast 0,249 s / deep 0,449 s**; 4 token 0,484 s, 5 token 2,003 s; çoklu birim tek `final`; UI’da stream kutusu, tarayıcı smoke testi geçti (chat final 0,793 s) | ✅ |
| 13o | Cache + kuyruk + metrik | aynı istek **9,089 s → 0,00005 s** (LRU); iki eşzamanlı fast’te ikincisi **8,814 s** kuyrukta; `/metrics` Prometheus; `--cache`, `--no-log` | ✅ |
| 13p | Chat/streaming HTTP doğrulaması | auto: selamlama ve “Fransa’nın başkenti?” → `chat`, kod → `deep`; chat cevabı “Merhaba! ...”, stream TTFT 1,07 s (soğuk) | ✅ |
| 13q | **Chat system prompt** (`exp25b`) | EN → **TR** prompt: keyword **%68,2 → %81,8**, exact **%75 → %100**, TR fact **%33 → %83**; liste 0/2 (TR+format 1/2 ama fact %67); varsayılan TR seçildi | ✅ ihtiyatlı |
| 13r | **Çok-tur `messages` API** (`exp25c`) | `query` yerine son 8 mesajlık geçmiş; `chat` birimine zorlanır; 5 senaryo/2 tur hatırlama **2/5 (%40)**, “hatırla” talimatı da 2/5; yapı çalışıyor, güvenilir hafıza yok | ✅ negatif sonuç |
| 13s | **60 problu chat kontrolü** (`exp26`) | chat vs base: keyword **%82,7 vs %55,8**, exact **%84,6 vs %15,4**, TR fact **%75 vs %25**, EN fact **%100 vs %83**, TR talimat **%81,8 vs %27,3**, matematik **%66,7 vs %50**, 3 satırlı liste **%62,5 vs %50**; n=60, TR ASCII katlama ile yeniden puanlandı | ✅ ihtiyatlı |
| 13t | Liste formatı ipucu (`exp27`) | 6 liste probu: düz **5/6**, kullanıcı eki **2/6**, system eki **3/6**; hedefli ipucu işe yaramadı, entegre edilmedi | ✅ negatif sonuç |
| 13u | **OpenAI uyumlu API + UI geçmişi** (`exp28`) | `/v1/models`, `/v1/chat/completions` non-stream + SSE `[DONE]`; canlı stream TTFT **0,443 s**, toplam **2,697 s**; UI’da deneysel geçmiş kutusu + konuşma günlüğü; hafıza hâlâ 2/5 | ✅ yapı, sınırlı kalite |
| 13v | **Şablon bellek katmanı** (`exp29`/`29b`/`29c`) | Ham geçmiş 3/5, `[Bilinen bilgiler: ...]` öneki 4/5; ad sorusu 3/4 ifadede doğru; UI 5 şablon (ad/kedi/renk/hava/toplantı), tarayıcı 3 turda 2/2 “Ali”; genel hafıza değil | ✅ dar kapsam |

**Yorum:** Ürün kabuğu çalışıyor: sorguya göre birim seçimi, VRAM bütçesine göre
paralel/sıralı plan, paged 3B’nin 6GB’de çalışması, kendi checkpoint’imizin
servis edilmesi, instruct sohbet birimi, token akışı, OpenAI uyumlu uçlar ve
cevap birleştirme tek uçtan uca akışta. Soğuk başlangıç disk yüklemesiyle 8–10 s;
sıcak istekler 0,6–3 s, deep 3,5–8 s; chat 1.5B ~0,6–0,9 s üretim. **Kalite:**
held-out WT2 CE −%9,2 (iyi), 24 OOD probda toplam +0,061 CE (küçük gerileme,
TR’de +0,102); 60 problu chat setinde keyword **%82,7**, exact **%84,6**, TR fact
**%75**, liste **%62,5**; yine de insan değerlendirmesi yok, liste/TR fact ve
matematik hataları sürüyor. **Çok-tur `messages`** API'si ve UI geçmişi çalışıyor
ama 5 senaryoda hatırlama **2/5**; güvenilir konuşma hafızası yok. 40 CPU-only
test, tek eşzamanlı istek, kural tabanlı rota, sabit VRAM tahminleri ve tek-birim
streaming bilinen
sınırlar (`SERVICE_RAPOR.md`).
