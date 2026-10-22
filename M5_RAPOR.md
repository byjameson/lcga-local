# M5_RAPOR — 3,09B parametrenin tamamını 6GB VRAM'de eğitmek (blok-streaming SGD)

**Tarih:** 19.10.2026 · **Kod:** `lcsa/stream_full.py`, `experiments/exp17_full_train.py`,
`experiments/exp17_parity.py` · **Veri:** WikiText-2 (Qwen tokenizer, 2,52M token), seq 512

## 1. Sonuç özeti

| Koşu | Model | Adım | lr | CE önce | CE sonra | Δ | Adım süresi | Peak VRAM | Checkpoint |
|---|---|---|---|---|---|---|---|---|---|
| 3B (ölçekli gradyan, eski) | 3,0859B | 100 | 2e-4 | 2,5034 | 2,4660 | **−0,0375** | 6,70 s | **2,433GB** | `qwen3b_full_sgd_t100.pt` |
| 3B temiz gradyan | 3,0859B | 150 | 5e-4 | 2,5034 | **2,4072** | **−0,0962** | 6,64 s | **2,433GB** | `qwen3b_full_sgd_t150_clean.pt` |
| 0.5B temiz gradyan | 0,494B | 30 | 1e-3 | 3,1215 | 3,0277 | **−0,0939** | ~2 s | 1,27GB | (kaydedilmedi) |

**Ana iddia:** 3,0859B parametrenin **tamamı** — VRAM'e sığmayan bf16 ağırlıklarla — 6GB kartta
**SGD ile güncellendi**. Peak 2,43GB → kartta ~3,6GB boşluk kaldı. Temiz koşu (150 adım) iki eval
ve 6,17GB kayıtla birlikte ~17 dk sürdü.

## 2. Yöntem

- **Blok-streaming:** katman i CPU→GPU kopyalanır; forward o blokta GPU'da çalışır; blok **henüz
  GPU'dayken** o bloğun tüm parametreleri için SGD güncellemesi yapılır; blok CPU'ya döner.
  Bu sayede aynı anda GPU'da tek blok bulunur (147MB) → peak sabit.
- **Embedding giriş gradyanı:** backward sırasında embed'den geçen akış sonda yeniden hesaplanıp
  toplanır (embed ↔ lm_head tied; 594MB bf16).
- **Chunked CE:** logits 128 pozisyonluk parçalarda; V=151.936 → parça başına ~78MB fp32.
- **bf16 gradyan + bf16 güncelleme**, global norm clip 1.0, optimizer state yok (SGD).
- Adım bütçesi: 512 token × 36 katman × 2 (ileri/geri) + gradyanlar → ~5,5GB PCIe/makro-adım;
  ölçülen ~6,7 s/adım ≈ PCIe + çekirdek başlatma sınırı.

## 3. Doğrulama (naive ↔ streaming paritesi)

`exp17_parity.py`: aynı batch (seq 256), naive yol (tüm model GPU'da, HF tam grafik) vs streaming
(blok blok). 0.5B, bf16:

| Metrik | Sonuç |
|---|---|
| Kayıp (stream / naive) | 2,649752 / 2,649752 → **fark 2,4e-7** |
| Gradyan rel_rmse (embed, q_proj, down_proj) | **%0,2–2,3** |
| Gradyan kosinüs | **1,000** (±bf16 gürültüsü) |

→ Streaming motoru, tam-grafik eğitimin matematiğini (bf16 gürültüsü sınırında) birebir üretiyor.

### 3b. Bulunan ve düzeltilen hata (dürüstlük notu)

İlk sürümde `chunked_lm_loss_and_backward` her parçada `mean` CE alıp `backward()` çağırıyordu;
bu, biriken gradyanı **parça sayısı kadar** ölçekliyordu (T=512, chunk=128 → **4×**; T=256 →
2×). Kayıp değerleri doğru olduğu için **eval metrikleri etkilenmemişti**, ama ilk 3B koşusunun
güncelleme ölçeği kirlidir (etkin lr ~4×; clip sonrası tam 4× değil). Düzeltme: parça başına CE
`reduction="sum"` ile hesaplanıp `n_pos`'a bölünüyor → biriken gradyan tam olarak ortalama
kaybın gradyanı. Parite testi hatayı görünür kıldığı için yakalandı (rel_rmse ≈1 → 2× ölçek).
Eski checkpoint **silinmedi**; karşılaştırma için duruyor, temiz koşu ayrı adla kaydediliyor.

## 4. 3B koşusu (ilk, ölçekli gradyan)

- Model yükleme 6,9 s (host RSS 6,83GB), eğitici RSS 6,33GB, eğitim boyunca tepe 7,63GB.
- Eval: 16 × 512 token, 46,6 s → **CE 2,5034** (önce) → **2,4660** (sonra).
- Adım süreleri: ortanca **6,70 s** (6,39–7,13 aralığı), 100 adım duvar **674,5 s**.
- Peak VRAM **2,433GB** (alloc), 2429 MiB dolu; kartta 3,5GB+ serbest kaldı.
- Kayıp gürültülü ama düşüş eğiliminde: ilk 10 adım ort. ~2,71 → son 10 adım ~2,33.

### 4b. Temiz koşu (düzeltilmiş gradyan — asıl sonuç)

- 150 adım, lr **5e-4**, seq 512; adım ortancası **6,64 s** (6,06–7,12), duvar **1001 s**.
- Eval: **CE 2,5034 → 2,4072 (−0,0962)**; ilk 10 adım ort. ~2,70 → son adım 1,894.
- Peak VRAM **2,433GB**; host RSS 7,63GB; checkpoint 6,17GB (kayıt 6,5 s).
- **Uçtan uca:** kaydedilen checkpoint, `exp16` paged int8 motoruyla 6GB kartta çalıştırıldı →
  tutarlı metin, **3,93 tok/s**, peak 4735MB:
  > "The capital of France is Paris. What is the capital of Germany? The capital of Germany is Berlin."
- Aynı motor ölçekli-koşu checkpoint'i (`t100`) ile de tutarlı üretti ("…is Berlin."); iki checkpoint
  de aynı örnekte Berlin'e ulaştı.

## 5. Sınırlar (bu ne DEĞİL)

- **Pretraining değil:** WikiText-2 üzerinde fine-tune demosu; kalite yarışı iddiası yok.
- **Optimizer:** SGD + Adafactor (faktörlü 2. moment, ~7MB state); Adam yok (3B Adam state ~24,7GB → RAM'e sığmaz).
- **bf16 güncelleme/gradyan:** varsayılan yol bf16 yerinde güncelleme → kuantizasyonla sınırlı (§7a).
  Opsiyonel fp32 master (`--master`) 0.5B'de doğrulandı; 3B'de +12,3GB RAM istediği için bu
  makinede koşulamadı (§7d).
- **Tek GPU, tek akış:** çok-GPU/dağıtık eğitim test edilmedi.
- **Transfer sınırı:** 3B'de adım süresinin **%71'i PCIe kopyası** (4 geçiş, ~5,3GB/s); `--prefetch`
  pageable bellekte kazanç sağlamadı (§7e). Pinned çift-tampon henüz uygulanmadı.
- **Kısa koşu:** 100–150 adım mekanizma kanıtı; uzun eğri/bitiş kalitesi yok.
- İlk 3B koşusunun güncelleme ölçeği kirliydi (§3b); **ana sonuç temiz koşudur** (150 adımda −0,0962;
  eski kirli koşu 100 adımda −0,0375). İki checkpoint de silinmedi.

## 6. Sıradaki adımlar

- ~~**Adafactor / 8-bit Adam CPU-offload**~~ → **Adafactor yapıldı (§7)**; 8-bit Adam hâlâ açık
  (3B state ~6,2GB, RAM sınırı).
- **Pinned blok çift-tampon** → §7e ölçümü: 3B'de adımın **%71'i PCIe**; hedef **~4,0 s/adım**
  (1,6×); pinned staging uygulanmadı.
- **7B sınıfı smoke** (bf16 ~14GB): peak sabit kalmalı, süre 2× beklenir.
- **Eval-tabanlı checkpoint seçimi** (her N adımda doğrulama CE).
- **SFT görevi:** talimat/stil verisiyle gerçek fine-tune; LoRA ile karşılaştırma (kalite/fiyat).
- **Çok-GPU iletişim** (P9b sonucuyla): faz-küme gradyan senkronu + streaming.

## 7. Adafactor + bf16 kuantizasyon duvarı (19.10 akşam)

### 7a. Sorun: bf16 yerinde güncelleme = kuantizasyon

`scripts/step_diag.py` (0.5B, aynı batch, ilk 2 adım; izlenen parametreler q_proj, gate, embed, norm):

- **SGD lr 1e-3:** q_proj'e uygulanan delta RMS **5,5e-9** (ağırlık RMS 6,8e-2); elemmanların
  **%0,0–0,1**'i değişiyor; `norm.weight` **hiç değişmiyor**. Yani güncellemelerin çoğu bf16
  yuvarlamasında kayboluyor; yalnızca |w| küçük olan (ULP küçük) elemanlar hareket ediyor.
- **Adafactor lr 1e-4:** elemanların **%60–68**'i ~1 ULP değişiyor (delta RMS 9,3e-5..1,3e-4);
  bir adım sonraki batch CE **+0,28**.
- Sonuç: normalize optimizer'ın lr'si **ham gradyan rejimine** ölçeklenmeli (gradyan RMS'leri
  ~1e-5–1e-4; SGD lr 1e-3 ile adım ~1e-7).

### 7b. Adafactor lr taraması (0.5B, 30 adım, seq 256, eval 8×256)

| lr | CE Δ |
|---|---|
| 1e-6 | −0,0105 |
| **1e-5** | **−0,2052** |
| 3e-5 | −0,1231 |
| 1e-4 | +5,9733 |
| 3e-4 (warmup 10) | +7,8580 |

(SGD lr 1e-3 referansı: −0,0939; master'sız.)

### 7c. 3B Adafactor — asıl sonuç (150 adım, lr 1e-5, seq 512)

| | SGD (temiz) | **Adafactor** |
|---|---|---|
| CE öncesi | 2,5034 | 2,5034 |
| CE sonrası | 2,4072 | **2,3242** |
| Δ CE | −0,0962 | **−0,1792 (1,86×)** |
| adım ort. | 6,64 s | 7,67 s |
| peak VRAM | 2,433GB | 4,933GB |
| host RSS | 7,63GB | 7,63GB |

- Peak artışı Adafactor'ün geçici fp32 gradyan kopyalarından (en büyüğü tied embed 1,24GB);
  **4,93GB hâlâ 6GB altında**.
- Sabit lr 1e-5 150 adımda kararlı kaldı; kayıp ilk 10 adım ort. ~2,55 → son adım 1,79.

### 7c-2. 300 adım + kosinüs decay — doygunluk (19.10 gece)

Aynı protokol, 300 adım, lr 1e-5 → 1e-6 kosinüs; 50 adımda bir eval + en iyi checkpoint:

| adım | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| CE | 2,3516 | 2,3328 | 2,3267 | 2,3238 | 2,3237 | **2,3217** |

- Δ CE **−0,1818** (150 adımlık koşu: −0,1792) → **kazanç ilk ~100 adımda tamamlanıyor**;
  sonraki 150 adım yalnızca +0,005 getirdi.
- 7,66 s/adım (medyan), duvar 2622 s, peak 4,93 GB — 150 adımlık koşuyla aynı rejim.
- En iyi checkpoint: `checkpoints/qwen3b_full_adaf_t300_best.pt` (CE 2,3217 @ step 300).
- Dürüst yorum: "daha çok adım = daha iyi" değil. Aday neden: tek dizi (512 token) üzerinde
  sıralı okuma → veri çeşitliliği yok; karıştırılmış çok-dizi eğitimi MLP'de (mini-LM, exp11)
  vardı, burada yok (M5 sınırı).
- Checkpoint `checkpoints/qwen3b_full_adaf_t150.pt` (6,17GB); **paged int8** ile 6GB'de
  **3,88 tok/s**, tutarlı metin:
  > "The capital of France is Paris . Paris is the most visited city in the world , with 22 million tourists visiting in 2014 . The City of Light ,"
  (tag `m5_ckpt_gen_adaf_int8`; peak 4735MB).
- Adafactor state'i faktörlü (satır+kolon) → 3B için ~7MB; RAM'e etkisi yok.

### 7d. fp32 master — kuantizasyon duvarının kesin çözümü

`--master`: fp32 kopyalar CPU'da; güncelleme fp32'de birikir, bf16 ağırlıklar her adımda
master'dan tazelenir. Maliyet: RAM **+4 bayt/parametre** (0.5B: +1,98GB; RSS 4,06→5,04GB) ve
**~2× adım süresi** (0.5B SGD 1,1→2,09 s; Adafactor 1,4→2,92 s) + ekstra D2H gradyan trafiği.

0.5B, 30 adım:

| Ayar | master'sız Δ CE | master Δ CE |
|---|---|---|
| SGD lr 1e-3 | −0,0939 | **−0,1049** |
| Adafactor lr 1e-5 | −0,2052 | **−0,2196** |
| Adafactor lr 1e-4 | +5,9733 | **+0,8108** |

- Teşhis doğrulandı: master, lr 1e-4'teki **+5,97 felaketini +0,81'e** indirdi (kalıntı hata artık
  lr'nin fazlalığından, kuantizasyondan değil).
- Tutarlı ~%10 CE kazancı; kısa koşularda maliyeti karşılıyor.
- **3B'de master bu makinede mümkün değil:** +12,3GB RAM gerekir; 34GB'lık makinede koşu
  sırasında yalnızca ~8,8GB boştu. 0.5B/1.5B'de kullanılabilir (dürüst sınır).

### 7e. Transfer profili — darboğazın kesin ölçümü (20.10)

`--profile` ile adım içi süreler ayrıştırıldı (3 adım, son adım; senkron sınırları eklendi —
asenkron çekirdekler nedeniyle ilk sürümde D2H, compute beklemesini de içeriyordu, düzeltildi):

| Bileşen | SGD | SGD + `--prefetch` | Adafactor |
|---|---|---|---|
| H2D (katman ağırlıkları) | **2,44 s** | 0,07 s (ölçülen) | **2,43 s** |
| forward (36 blok) | 0,37 s | 0,37 s | 0,37 s |
| D2H (katman ağırlıkları) | **2,26 s** | 2,23 s | **2,26 s** |
| loss + lm_head backward | 0,16 s | 0,16 s | 0,16 s |
| blok backward | 0,99 s | 0,98 s | 0,98 s |
| update | 0,29 s | 0,29 s | **1,50 s** |
| **adım (medyan)** | **6,56 s** | 6,57 s | **7,95 s** |

- **Adımın %71'i PCIe:** 4 geçiş × 6,17GB ≈ **24,7GB/adım**, ölçülen ~**5,3GB/s** (pageable +
  36 küçük kopya + WDDM). Compute (fwd+bwd+upd+loss) yalnızca **1,81 s (%28)**.
- **`--prefetch` kazanç sağlamıyor:** H2D ölçülen bucket'ta 2,44 → 0,07 s'ye iniyor ama adım
  süresi **6,56 → 6,57 s** (değişmedi). Nedeni: pageable CPU belleğinde `non_blocking=True`
  gerçekte ana thread'i bloke ediyor; kopya yalnızca *ölçümden* çıkıyor, süreden çıkmıyor
  (kayıp ~2,48 s, yan akışta). **Dürüst sonuç: prefetch bu haliyle faydasız.**
- **Adafactor'ün maliyeti net:** update 0,29 → **1,50 s** (+1,21 s, CPU Adafactor); transferler
  aynı. Yani 3B Adafactor'ün +1,1 s/adım yavaşlığı tamamen optimizer'dan.
- **Sıradaki gerçek kaldıraç: pinned çift-tampon.** 168MB'lık iki pinned staging buffer ile
  kopyalar ~12GB/s'ye çıkarsa transfer 4,70 → ~2,1 s, adım **~4,0 s (≈1,6×)**; Adafactor
  ~5,4 s. Uygulanmadı (pinned bellek RAM'i kilitler; 6,17GB'ı pinlemek bu makinede riskli,
  katman-başına 2×168MB ile sınırlı tutulmalı).
- **7B bf16 bu makinede RAM'e sığmaz:** model 15,2GB, koşu sırasında boş RAM ~8,8GB. 7B int4
  (~3,8GB) teoride mümkün ama eğitimde kuantizasyon etkisi ayrı ölçüm ister (M5 sınırı).

### 7f. Blok başına GPU tamponu — forward write-back'ini kaldırma (20.10)

`--buffers`: her katmanın ağırlıkları kalıcı bir GPU tamponuna kopyalanır ve `p.data` bu tampona
yönlendirilir; böylece **forward'da ağırlıkları CPU'ya geri kopyalamak gerekmez** (ağırlıklar
değişmiyor). Yalnızca güncelleme yapan backward'da write-back yapılır.

| 3B, 3 adım | adım | H2D | D2H | host kopya | geçiş |
|---|---|---|---|---|---|
| baseline | 6,56 s | 2,44 | 2,26 | – | 4 |
| **`--buffers`** | **5,20 s (1,26×)** | 2,27 | **1,06** | – | **3** |
| `--pinned` | 5,68 s | 1,40 | 0,86 | **1,55** | 3 + CPU staging |
| 0.5B baseline | 1,04 s | 0,37 | 0,37 | – | 4 |
| 0.5B `--buffers` | **0,79 s** | 0,33 | 0,15 | – | 3 |
| 0.5B `--pinned` | 0,78 s | 0,18 | 0,11 | 0,18 | 3 + CPU staging |

- **Parite birebir:** her iki modelde de üç mod aynı loss dizisini verdi (3B:
  2,710462 / 2,536529 / 2,091281; 0.5B: 2,649752 / 3,349388 / 2,774425).
- Peak VRAM 2,43 → **2,55GB** (kalıcı katman tamponu, +0,12GB) — hâlâ 6GB altında.
- **`--pinned` tek başına negatif:** H2D 2,44→1,40 ve D2H 2,26→0,86 hızlanıyor ama pinned
  staging'e yapılan CPU kopyası adım başına **1,55 s** ekliyor; toplam 5,68 s. Pinned'in değeri
  ancak **çift-tamponla örtüşme** sağlanırsa ortaya çıkar (staging + H2D, compute'un arkasına
  saklanabilir). Bu, sıradaki adım.
- Yeni bayraklar varsayılan **kapalı** (eski satırların tekrarlanabilirliği için); 3B için
  önerilen: `--buffers`. Eval yolu da aynı modda doğrulandı (3B, 2 chunk, 5,15 s/adım).
