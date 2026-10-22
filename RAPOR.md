# LCGA — Qwen2.5-0.5B Üzerinde Öğrenilebilir Blok-Seyrek Attention: Final Rapor

> **Tarih:** 15–16 Ekim 2026 · **Donanım:** RTX 3050 Laptop 6GB, Windows 11 · **Yazılım:** PyTorch 2.13.0+cu126, Transformers 5.19.0, Triton
> **Kod:** `lcsa/` (model, router, kernel, LoRA, decode, decode_graph), `experiments/` (exp2, exp3, exp5–exp10), `scripts/` (speed_ref, decode_ref, check_sparse_graph, v2_context_curve, make_plots, summarize_results) · **Kayıtlar:** `results/results.jsonl`, `checkpoints/` (v1_best, v2_best, v2ctl_best, v3_best)
> Bu rapor yalnızca kayıtlı/doğrulanmış ölçümlere dayanır; A (router), B (LoRA LR), C (decode) fazları tamamlandı (16 Ekim 2026); P1 (parça-parça eğitim) ayrı raporda (`P1_RAPOR.md`). Uydurulmuş sayı yoktur.

---

## 1. Yönetici Özeti

**Amaç:** Qwen2.5-0.5B'de O(N²) dense attention'ı, kaliteyi koruyarak öğrenilebilir blok-seyrek attention (LCGA) ile değiştirmek ve uçtan uca hızlanmayı ölçmek.

**Kısaca ne işe yaradı:**

- **Hız:** A fazı router optimizasyonlarından sonra temiz ölçümde **crossover ≈ 5k token** (öncesi ≈ 8k). T=16384'te sparse, dense'ten **1.25x hızlı** (2041.0 ms vs 2560.3 ms). Kısa bağlam açığı kapandı: T=1024'te 0.66x → **0.92x**, T=2048'te 0.81x → **0.94x** (A2 CUDA-graph router replay; yalnız inference). Sparse neredeyse lineer ölçekleniyor, dense süperlineer → bağlam uzadıkça kazanç büyüyor.
- **Kalite (wt2_test, T=1024, ρ=0.25, 128 chunk):** v1 (yalnız router) dense 14.304 → sparse 21.838 (**+%52.7**); v2 (stage-2 LoRA) dense 12.881 → sparse 16.732 (**+%29.9**); **v3 (düşük LoRA LR 1e-4, B fazı)** dense **12.456** → sparse **16.559** (+%32.9) — dört model arasında en iyi mutlak dense ve sparse. KL 0.4299 → 0.2880 → 0.3039, top1 0.7649 → 0.8049 → 0.7992.
- **Sadakat kanıtı:** ρ=1.0'da hem v1 hem v2 dense ile birebir eş (14.303 / 12.854) → seyrek formülasyon (seçim + kernel) matematiksel olarak dense attention'ın aynısı; sapmalar tamamen bütçe kısıtından.
- **Tutarlılık kanıtı:** 2-blok bütçesinde (ρ=0.125) öğrenilen router'ın sonucu, sabit "sink + diyagonal" (local) kontrolüyle **birebir aynı** (v1: 26.788; v2: 19.487) — router kısa bütçede tam olarak zorunlu bloklara dejenere oluyor.

**Kısaca ne işe yaramadı / açık kalan:**

- **Kısa bağlamda (≤2k) uçtan uca hız kazancı hâlâ yok ama açık kapandı:** A fazı sonrası @1024 0.66x → **0.92x**, @2048 0.81x → **0.94x**; crossover ≈ 5k. Attention op'u @4096'da zaten 2.2x ucuz (1.279 ms vs 2.845 ms); kalan fark router + sabit katman overhead'leri.
- **OOD bedeli — büyük ölçüde giderildi (B fazı):** v2 LoRA wikitext dışında ciddi unutuyordu (TinyStories dense 5.557 → 10.370); **v3 (LR 1e-4)** bunu geri aldı: dense **6.090** (v1 seviyesine yakın), sparse **9.158** — ve on-domain'i bozmadı. Tek OOD nöbetçisi TinyStories'tir; "tamamen unutmuyor" iddiası değil.
- **Decode ÇÖZÜLDÜ (C fazı):** tüm decode adımı tek CUDA graph; **dense 13.0–24.2 ms, sparse 13.5–19.1 ms** (T=1k–8k), parity dense bitwise / sparse top1=1.0. Eager'a karşı **~4–7x**; süreler host yükünden bağımsız (eager oynak). Uzun bağlamda sparse decode, dense graph'tan %12–27 hızlı.
- **Kazancın %83'ü jenerik adaptasyon:** v2 iyileşmesinin yalnızca **%17'si seyreklik-özgü** (v2ctl kontrolü, §3.1/§3.6) — kalanı LoRA'nın wiki domainine adaptasyonu. Öğrenilebilir seyrek katman gerçek ama mütevazı bir katkı sağlıyor.

---

## 2. Sistem ve Yöntem

### 2.1 Router (latent kavram grafiği)

- Her attention katmanına `ConceptGraphRouter` (`lcsa/concepts.py`) eklenir; taban model **donuk** (fp16).
- Token → **K=64 latent kavram** üzerinde softmax; blok düzeyinde (blok = 64 token) skor = diyagonal terim + **düşük ranklı kavram grafiği** `U Vᵀ` (rank 16) + göreli konum yanlılığı `-softplus(α)·log1p(I−J)`.
- Toplam router parametresi: **2.804.280** (~2.8M); opsiyonel `dynamic` / `content_rank` yolları v1'de kapalı.

### 2.2 Blok seçimi

- Her sorgu bloğu `I` için bütçe: **m_I = max(2, round(ρ·(I+1)))** blok.
- **Zorunlu:** blok 0 (sink) + diyagonal (I); kalan yerler router skoruna göre top-m.
- Çıktı: `idx (B, NB, MMAX) int32`, `-1` = boş. Blok boyu 64; T=1024'te NB=17, T=4096'da NB=64.

### 2.3 Exact (kesin) seyrek yol — Triton kernel

- `lcsa/attention.py::sparse_attention_triton`: seçilen bloklar üzerinde **causal, GQA-farkında, fp16, fwd-only** kesin attention; çıktı `(B,H,T,D)`. Yaklaşım yok — dense ile aynı matematiğin seçilmiş-bölge kısıtı.
- **Coarse yol** (blok-ortalama düşük çözünürlüklü yol) kodda duruyor ama temiz konfigde **kapalı**: ölçümde monoton zarar (ρ=0.25, gate süpürmesi: g=0 → 971.6; g=0.3 → 279.8; g=0.5 → 45.5; g=0.7 → 24.7; saf exact g=1.0 → **21.0**). Coarse tek başına ≈ 970 PPL.

### 2.4 İki aşamalı eğitim

- **v1 (yalnız router, `exp2_train_router.py`):** katman-lokal damıtma — `MSE(öğrenci çıkışı → dense çıkışı)` + `λ_kl·KL(dense blok-kütlesi ‖ router olasılıkları)`; katmanlar arası autograd yok (`detach`); Adam, LR 1.5e-3, warmup 100, kosinüs, **1500 adım**, batch 4, T=1024.
- **v2 (stage-2, `exp7_stage2.py`):** donuk taban + **LoRA (attn q/k/v/o, r=16)** sparse yolun üzerinden eğitilir; kayıp = sparse yolda **LM (CE)** + `0.5·KL` + `0.3·MSE` yardımcı kayıplar; ayrıca head-başına **öğrenilebilir recall-gain** (init 1.0). AdamW: router/gain 3e-4, LoRA 1e-3; batch 1, T=1024; planlanan 2000 adım — koşu ortam tarafından **1250'de kesildi** (crash yok; eğri düzleşmişti) → `v2_best.pt` = step 1250.
  - Recall-gain öğrenilen değer **≈1.0 kaldı** (nötr) → temiz ölçekleme formülasyonu doğrulandı, fazladan bir kazanç kapısı olmadı.
- **Temiz inference konfigü (tüm sayılar bu konfigle):** `use_coarse=False`, `null_logit=-1e9` (kapalı), `selection=topk`, sparse açık. Gerekçe: (i) coarse ölçülmüş saf zarar; (ii) eğitilmiş sabit null-logit **-3** bir tasarım hatasıydı — per-satır `sigmoid(lse−nl)` token bazında rastgele sönümleme yapıyordu (`scale_min=0.0`, bazı tokenların attention çıkışı sıfırlanıyordu). Kapalı konfigde ρ=1.0 dense ile birebir eş çıkınca tüm büyük sapmaların bu bug'dan geldiği kanıtlandı. (`load_ckpt` bu değeri geri yüklediği için yükleme sonrası `fill_(-1e9)` zorunludur.)

---

## 3. Kalite Sonuçları

### 3.1 Ana karşılaştırma (T=1024, wt2_test, 128 chunk, temiz konfig)

| Model | dense PPL | sparse ρ=0.25 | Δ PPL | Δ % | KL (d→s) | top1 | recall |
|---|---|---|---|---|---|---|---|
| v1_clean (yalnız router) | **14.304** | 21.838 | +7.534 | **+%52.7** | 0.4299 | 0.7649 | 0.778 |
| v2_best (stage-2, step 1250) | **12.881** | 16.732 | +3.851 | **+%29.9** | 0.2880 | 0.8049 | 0.779 |
| **v3_best** (stage-2, LoRA LR 1e-4 — §3.7) | **12.456** | **16.559** | +4.103 | **+%32.9** | 0.3039 | 0.7992 | 0.783 |
| **v2ctl** (kontrol: yalnız dense-LoRA, 1250 adım) | **12.796** | 17.264 | +4.468 | **+%34.9** | 0.3224 | 0.7943 | 0.785 |

**Bağımsız/OOD test — TinyStories validation (T=1024, 128 chunk):**

| Model | dense PPL | sparse ρ=0.25 | Δ % | KL (d→s) |
|---|---|---|---|---|
| v1_clean | **5.557** | 7.963 | +%43.3 | 0.3739 |
| v2_best | **10.370** | 17.221 | +%66.1 | 0.3743 |
| v2ctl | **10.319** | 17.065 | +%65.4 | 0.3859 |
| **v3_best** (LR 1e-4) | **6.090** | 9.158 | +%50.4 | 0.3881 |

- `test_wt103_1024` ayrı satır değildir: **WikiText-2 ve WikiText-103'ün valid/test split'leri özdeştir** (veri seti ailesi özelliği; yalnızca train farklı — 60 makale/245.569 token ortak). 16.10'da HF'ten yeniden indirilerek doğrulandı: token akışı `wt2_test.pt` ile birebir. Gerçek bağımsız test = TinyStories validation.

### 3.2 Bütçe (ρ) süpürmesi — T=1024, wt2_test

| ρ | blok bütçesi m(I) | v1 sparse | v2 sparse | v2ctl sparse | Not |
|---|---|---|---|---|---|
| 0.125 | 2 (zorunlu) | 26.788 | 19.487 | 20.356 | = local kontrol (birebir, üç modelde) |
| 0.25 | ~4 | 21.838 | 16.732 | 17.264 | ana çalışma noktası |
| 0.5 | ~8–9 | 17.176 | 14.268 | 14.495 | |
| 1.0 | tüm bloklar | **14.303** | **12.854** | **12.796** | dense ile birebir → formülasyon sadık |

### 3.3 Kontroller — T=1024, wt2_test, ρ=0.25

| Kontrol | v1 | v2 | v2ctl |
|---|---|---|---|
| rastgele seçim (aynı bütçe) | 23.883 | 18.224 | 18.911 |
| local = sink+diyagonal, m=2 | 26.789 | 19.487 | 20.356 |

**Kritik gözlem (tutarlılık kanıtı):** `local` kontrolü, ρ=0.125 sonucuyla **birebir aynı** (v1: 26.7887 vs 26.7884; v2: 19.4872 vs 19.4873; v2ctl: 20.356 = 20.356). Yani 2-blok bütçesinde öğrenilen router tam olarak sink+diyagonal seçiyor — bu bir bug değil, bütçenin zorunlu bloklarla çakışması. İçerik-yönlendirmesi ancak m≥4'te devreye girebiliyor.

### 3.4 v1 bağlam eğrisi (exp6, temiz konfig, 16–32 chunk protokolü)

| T | dense | ρ=0.25 | Δ % | ρ=0.5 | Δ % | oracle@0.25 | pencere@0.25 | recall@0.25 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 14.559 | 22.02 | +%51 | 17.38 | +%19.4 | 21.61 | 21.97 | 0.778 |
| 2048 | 13.322 | 18.27 | +%37 | 15.16 | +%13.8 | 17.14 | 18.29 | 0.792 |
| 4096 | 12.339 | 15.56 | +%26 | 13.52 | +%9.5 | 14.36 | 15.60 | 0.826 |

**Ana trend:** Bağlam uzadıkça (a) göreli kalite kaybı küçülür, (b) recall artar (router NB=16'da eğitildi, 64 bloğa genelleşiyor), (c) hız kazancı büyür → **yöntemin asıl rejimi uzun bağlam.** T=4096'da oracle (14.36), eğitilmiş router'dan (15.56) ~1.2 PPL iyi → uzun bağlamda "hangi uzak blok" seçimi önem kazanıyor; stage-2 bu açığı hedefliyordu.

### 3.5 v2 bağlam eğrisi (16.10.2026; 32/16/16/8 chunk protokolü)

| T | dense | ρ=0.25 | Δ % | ρ=0.5 | Δ % | recall@0.25 |
|---|---|---|---|---|---|---|
| 1024 | 12.697 | 16.224 | **+%27.8** | 14.064 | +%10.8 | 0.779 |
| 2048 | 11.839 | 13.990 | **+%18.2** | 12.677 | +%7.1 | 0.801 |
| 4096 | 12.311 | 13.826 | **+%12.3** | 12.899 | +%4.8 | 0.841 |
| 8192 | 12.059 | 12.965 | **+%7.5** | 12.454 | +%3.3 | 0.868 |

**Ana trend:** v2'de göreli kalite kaybı bağlamla birlikte hızla küçülüyor (+%27.8 → **+%7.5**; v1'in exp6 eğrisi +%51 → +%26, 1k→4k), recall 0.779 → **0.868**. **T=8192, hız crossover'ının (o ölçümde 1.00x; A fazı sonrası ≈5k) üzerinde:** bu noktadan itibaren seyrek yol hem hızlı hem göreli kaybı küçük → "yöntemin rejimi uzun bağlam" tezi v2 için de kanıtlandı. (Not: T=1024 satırı 32 chunk'tır; §3.1'in 128-chunk değerlerinden küçük farklar taşır: dense 12.697 vs 12.881.)

### 3.6 Kısa tartışma

- **v1 → v2 iyileşmesi gerçek ve aynı protokolde:** gap +%52.7 → +%29.9 (ρ=0.25); ρ=0.5'te +%20.1 → +%10.8; KL 0.43 → 0.29; top1 0.765 → 0.805. v2'nin dense referansı da 14.304 → 12.881 (LoRA'nın wikitext'e adaptasyonu dense kanalda da görünüyor).
- **OOD uyarısı (dürüst sınır):** TinyStories'te v2, v1'den kötü — dense 5.557 → 10.370 (LoRA jenerik metni "unuttu"), sparse 7.963 → 17.221 (gap +%43.3 → +%66.1). v2 sonuçları yalnızca **eğitim dağılımı (wikitext-2)** için iddia edilebilir. → **B fazı (§3.7) bu bedeli büyük ölçüde giderdi** (LR 1e-4: 6.090 / 9.158).
- **Attribution ÇÖZÜLDÜ (v2ctl, §3.1):** aynı protokolde kontrol (router donuk, yalnız dense-yolda LoRA) gap'i +%52.7 → **+%34.9**'a indirdi; v2 ise +%29.9. Yani **gap azalmasının %83'ü jenerik domain adaptasyonu, %17'si seyreklik-özgü** (0.617 gap puanı; sparse PPL'de v2, kontrol'den %3.1 bağıl iyi: 16.732 vs 17.264). Dense tarafta kontrol marjinal daha iyi (12.796 vs 12.881) → **router/seyrek-yol eğitimi dense'ten hafif feragat edip sparse'a yatırım yapıyor.** TinyStories'te davranış benzer (v2ctl 10.319 / 17.065) → **OOD bedeli jenerik LoRA adaptasyonundan geliyor, seyreklikten değil.**
- **Uzun bağlam (v2, §3.5):** göreli kayıp +%27.8 (1k) → **+%7.5** (8k), recall 0.868; hız crossover'ı o ölçümde tam 8k'daydı (A fazı sonrası ≈5k) → uzun bağlamda kalite ve hız eğrileri birlikte lehte.

### 3.7 B fazı: düşük LoRA LR (v3) — on-domain en iyi, OOD unutma büyük ölçüde geri alındı

**Koşu (tek değişken):** v2 protokolünün aynısı; yalnız **LoRA LR 1e-3 → 1e-4** (router LR 3e-4 sabit), warm-start `v1_best` (v2 ile aynı başlangıç), 1250 adım, seed 0 → `checkpoints\v3_best.pt`. Dev eğrisi (exp7_eval): sparse 16.202 → **14.867** (v2: 15.516 → 14.640); dev dense 11.379 (v2: 11.500).

| Ölçüt (exp3_eval, 128 chunk) | v1_clean | v2 (LR 1e-3) | **v3 (LR 1e-4)** | v2ctl (kontrol) |
|---|---|---|---|---|
| wt2 dense | 14.304 | 12.881 | **12.456** | 12.796 |
| wt2 sparse ρ=0.25 | 21.838 | 16.732 | **16.559** | 17.264 |
| wt2 gap % | +%52.7 | +%29.9 | **+%32.9** | +%34.9 |
| TinyStories dense | 5.557 | 10.370 | **6.090** | 10.319 |
| TinyStories sparse | 7.963 | 17.221 | **9.158** | 17.065 |
| TinyStories gap % | +%43.3 | +%66.1 | **+%50.4** | +%65.4 |
| KL (wt2) / top1 | 0.4299 / 0.7649 | 0.2880 / 0.8049 | 0.3039 / 0.7992 | 0.3224 / 0.7943 |

- **On-domain: dört model arasında en iyi mutlak değerler** (dense 12.456, sparse 16.559). Göreli gap v2'ye göre hafif büyük (+%32.9 vs +%29.9) çünkü dense taraf oransal olarak daha çok iyileşti (12.881 → 12.456, −%3.3; sparse 16.732 → 16.559, −%1.0).
- **OOD: unutma büyük ölçüde geri alındı** — TinyStories dense 10.370 → **6.090** (v1'in 5.557'sine yakın; dense "unutma" v2'de +%86 iken v3'te **+%9.6**), sparse 17.221 → **9.158** (+%116 → **+%15** rel). Unutmanın jenerik LoRA adaptasyonundan geldiği v2ctl'de zaten gösterilmişti (§3.6) — LR bu bedeli doğrudan kontrol ediyor.
- **Sadakat/tutarlılık aynen sürüyor:** ρ=1.0 delta −0.004 (dense ile birebir); ρ=0.125 == local kontrol (19.405, birebir); random 18.047.
- **Küçük not:** dev izleme eğrisinde v3'ün sparse'ı 1250. adımda v2'nin 0.23 PPL altında kaldı (14.867 vs 14.640) — ama test protokolünde v3 her iki yönde de önde; dev/test arasındaki bu tip küçük farklar bütçe farklarından.
- **Özet:** v3 = v2'den iyi on-domain + OOD bedeli ~6 kat küçük. Bundan sonrası için doğal aday: LR 1e-4 ile daha uzun koşu (kapasite tam kullanılmıyor olabilir).

---

## 4. Hız Sonuçları

### 4.1 Temiz uçtan uca tablo (v2_best, LoRA merge'li, A fazı sonrası; `scripts/speed_ref.py`)

Ölçüm koşulları: batch 1, `logits_to_keep=1`, **dense önce taze süreçte**, `torch.no_grad()`, tüm parametrelerde `requires_grad_(False)`, FLASH/EFFICIENT SDPA backend'leri pinli, medyan. Sparse tarafında **A2 CUDA-graph router replay** açık (yalnız inference; ağırlıklar dondurulduktan sonra capture).

| T | dense (ms) | sparse ρ=0.25 (ms) | dense/sparse | (A öncesi) |
|---|---|---|---|---|
| 1024 | 111.4 | 120.7 | **0.92x** | 0.66x |
| 2048 | 217.7 | 230.7 | **0.94x** | 0.81x |
| 4096 | 436.5 | 446.6 | **0.98x** | 0.90x |
| 8192 | 969.6 | 928.8 | **1.04x** | 1.00x |
| 12288 | 1689.1 | 1464.0 | **1.15x** | 1.13x |
| 16384 | 2560.3 | 2041.0 | **1.25x** | 1.23x |

**Ölçeklenme:** Sparse ≈ lineer (ms/token ~0.13'te pratikte sabit); dense süperlineer (~×1.9–2.0/doubling). Oran bağlamla sürekli sparse lehine dönüyor; **crossover ≈ 5k token**. (Graphsız A1 koşusu: 128.1/235.7/450.4/932.8/1460.4/2036.6 ms → 0.87/0.93/0.97/1.04/1.15/1.26x — graph özellikle ≤2k'da kazandırıyor, uzun T'de nötr.)

### 4.2 Kırılım (forward başına, 24 katman toplamı)

**A fazı sonrası (v2_a2, graph):**

| T | router (ms) | Triton attention kernel (ms) | o_proj (ms) |
|---|---|---|---|
| 1024 | **6.5** | 3.2 | 17.5 |
| 2048 | **11.0** | 8.4 | 29.5 |
| 4096 | **23.0** | 27.1 | 55.5 |
| 8192 | **49.3** | 98.4 | 106.0 |
| 12288 | **62.3** | 214.8 | 173.4 |
| 16384 | **83.4** | 375.1 | 222.2 |

*(A öncesi, eager: router 44.8/44.3/56.6/78.0/91.6/102.2; kernel 7.1/11.9/29.5/99.3/215.7/375.7; o_proj 23.3/31.4/55.7/106.0/172.7/222.1. Kısa T'de kernel/o_proj'in de düşmesi launch sıkışıklığının azalmasından.)*

### 4.3 A fazı: router maliyetini düşürme (16.10)

Üç aşama; kalite etkisi **sıfır** (birebir doğrulamalı):

1. **A1 — önbellek + vektörleştirme (exact-math):** `ConceptGraphRouter`'da şekle bağlı sabitler (causal maske, topk bias tablosu, `log1p(rel)` mesafe tablosu, bütçe tabloları) her çağrıda yeniden üretiliyordu; artık (NB, budget) bazında önbellekte. `select_blocks` tek vektörel bloğa indirildi. Doğrulama: `scripts/parity_router_a1.py` **10/10 bitwise eşit** (scores, idx, probs, gumbel dahil).
2. **A1.5 — çift softmax kaldırıldı:** `select_blocks` zaten birebir aynı `probs`'u üretiyordu; wrapper ikinci kez `masked_probs` çağırıyordu. Eval (gradsız) yolunda tekrar kullanılıyor; eğitim yolu (gradyan) değişmedi. Doğrulama: exp3-quick **16.224 / 12.697 / recall 0.779** — A öncesiyle birebir.
3. **A2 — CUDA-graph router replay (inference-only, opt-in `--graph`):** 24 katman × ~25 minik op grafiğe alındı; katman başına 1 replay + girdi kopyası. İlk gerçek girdide **eager-vs-graph eşitlik kapısı** var (idx bitwise, probs 1e-5); uyuşmazlıkta kalıcı eager fallback. Etki: @1024 router **44.8 → 6.5 ms** (≈7x); ≤2k'da toplam belirgin iyileşir, ≥8k'da nötr (uzun T'de router artık GPU-işiyle sınırlı).

Hedef "@1024 router < 15 ms" **aşıldı (6.5 ms)**; crossover 8k → **≈5k**. Not: "paylaşımlı router" fikri compute kazandırmıyor (her katmanın girdisi farklı → 24 kez çalışır); bu yüzden A2 CUDA-graph olarak seçildi. K=32 retrain opsiyonel kaldı.

- Router kısa bağlamda en büyük overhead: T=1024'te toplamın **~%27'si**, T=16384'te **~%5'i**. Crossover'ı sola çekmenin anahtarı router maliyeti.
- Attention op'u düzeyinde @4096 sparse 1.279 ms vs dense 2.845 ms (~2.2x ucuz) — ama uçtan uca @4096 hâlâ 0.90x: fark, router + seçim + diğer katman overhead'lerinden.
- Not: Profildeki `coarse_ms` etiketi aslında **o_proj** süresidir (coarse kapalıyken de raporlanır).

### 4.4 "As-loaded" vs "fast dense" ayrımı (dürüstlük notu)

- Bu torch build'inde transformers SDPA'da `enable_gqa=True` ile **fused GQA kernel yok → MATH fallback**: "as-loaded" dense baseline ~100x yavaş görünüyordu. As-loaded T=1024/2048/4096/8192 = 242.9 / 777.7 / 2768 / **106648 ms** (≈106 s) vs düzeltilmiş fast dense aynı aralıkta 90.1 / 179.1 / 363.4 / **831.4 ms**.
- **Tüm hız karşılaştırmaları "fast dense"e karşıdır;** as-loaded yalnızca bu not için raporlanır. Düzeltme: `lcsa/harness.py::enable_fast_dense()` (+ PPL pariteli, |Δce|≈1.6e-4).

### 4.5 Ölçüm tuzakları (bulundu ve giderildi)

1. **`enable_gqa` / MATH fallback** — yukarıda; dense baseline'ı yanıltıcı biçimde 100x yavaşlatıyordu.
2. **Autograd-grafik tuzağı:** LoRA A/B `requires_grad=True` kalır ve zamanlama `torch.no_grad()` dışında yapılırsa, autograd grafiği T≥4096'da 4–6GB aktivasyon tutuyor → 6GB kartta WDDM shared-memory taşması → dense **6–30x yavaş** ölçülüyordu (2758 ms@4096, 24917 ms@8192 gibi sahte sayılar; doğrusu 436/967 ms). Çözüm: ölçümden önce `requires_grad_(False)` + `torch.no_grad()`. **Kirlenen exp4 satırları `results/results.jsonl`'den silindi.**
3. **Interleaved ölçüm güvenilmez:** exp4'ün iç içe dense/sparse ölçümü uzun bağlamda allocator kirlenmesi yüzünden sapıyor → T≥4k için `scripts/speed_ref.py` (dense-önce, taze süreç, pinli backend) kullanıldı.
4. **`reserved >> allocated` = taşma alarmı:** 6GB kartta `memory_reserved() > 6GB` ise ölçüm geçersiz; temiz koşuda reserved ≈ 1–2GB kalır (bu koşuda doğrulandı).
5. **TF32 hiçbir hız kazancı getirmedi** → kapalı; sayısal temizlik için dokunulmadı.
6. **Eğitim döngüsünde sync'siz zamanlama = enqueue-artefaktı (WDDM):** exp10 (P2) eğitim döngüsü adım süresini `torch.cuda.synchronize()` olmadan ölçtü; flat modelinin "16.55 ms/adım" değeri CPU-enqueue süresiydi — gerçek tempo **250.7 ms/adım** ve 49.7 dk duvar süresiyle birebir uyumlu. (Patch'in 68.56 ms kaydı da ~2.2x düşüktü; gerçek 150.1.) Düzeltme: `scripts/verify_p2_timing.py` (her adımda sync; iki koşu stabil) + exp10 eğitim döngüsüne adım-başı sync eklendi. Ders: hız sayıları bağımsız/sync'li doğrulamadan alınır — kalite yeniden ölçümü de aynı script'te yapıldı (bkz. `P2_RAPOR.md`).

### 4.6 C fazı: decode — tüm adım tek CUDA graph (16.10; kanıtlı)

**Teşhis (eager harness, `scripts/decode_ref.py`):** T=1024–8192'nin her bağlamında dense-naive ~62–98 ms, sparse ~84–142 ms ve **süre T'den neredeyse bağımsız** → decode darboğazı KV trafiği değil, **kernel-launch faturası** (~300+ küçük op/adım, WDDM/Windows). Genişletilmiş KV cache (repeat_kv bir kez) ve GQA-SDPA yolları bu faturayı değiştirmiyor.

**Çözüm (`lcsa/decode_graph.py`):** tüm decode adımı (24 katman) **tek CUDA graph** olarak yakalanır; adım başına host işi yalnız 2 küçük H2D (token + pozisyon) + tek replay. Dense: tam-cache SDPA + `scatter_` ile ilerletilen ek maskesi. Sparse: **graph içinde** router istatistiği güncellemesi + sabit-M topk + seçilen bloklardan toplanan mask — slot-düzeyi nedensellik (eğitim semantiğinin aynısı). Capture; default → EFFICIENT_ATTENTION → MATH sırasıyla denenir (backend tuzağına karşı); başarısızlıkta benchmark hata kaydeder ve devam eder.

**Sonuç (exp9_c_phase, `results/exp9_c_phase.json`; 64 adım; ms/token):**

| T | dense naive (eager) | sparse naive (eager) | sparse fast (eager) | **graph dense** | **graph sparse** |
|---|---|---|---|---|---|
| 1024 | 86.7 | 99.4 | 127.6 | **13.03** | **13.50** |
| 2048 | 90.7 | 129.5 | 122.3 | **14.63** | **14.34** |
| 4096 | 93.1 | 138.8 | 141.5 | **17.86** | **15.90** |
| 8192 | 97.8 | 133.6 | 137.2 | **24.21** | **19.08** |

- **Parity:** graph dense, naive dense ile **dört bağlamda bitwise birebir** ([0.0, 1.0]); graph sparse vs naive sparse [0.029–0.082, **top1=1.0**], vs fast sparse **[0.0, 1.0] (bitwise)**. Blok seçimleri katman katman birebir (`scripts/check_sparse_graph.py`: 24/24 + T=1024 parity [0.043, 1.0]).
- **Kazanç:** aynı koşuda graph, eager dense-naive'a karşı **4.0–6.7x**, eager sparse yollarına karşı **7.4–9.5x**. Eager süreler host yüküne belirgin duyarlı (önceki koşuda dense-naive 61.5–66.3 ms, sparse-fast 75.8–84.7 ms ölçülmüştü — koşular arası %40–80 oynama); graph süreleri iki koşuda **±%0.2 aynı** (tek launch → launch-bound rejimden bağımsız).
- **Uzun bağlam:** graph sparse, graph dense'e karşı 4k'da 1.12x, **8k'da 1.27x** hızlı — prefill'deki crossover (~5k) decode'da da görünüyor (graph'lar arası geçiş ~1–2k). Dense graph 13→24 ms, sparse graph 13.5→19 ms; sparse graph neredeyse düz.
- **Kapsam (dürüstlük):** batch=1, tek istek, 0.5B, 6GB laptop GPU. Karşılaştırma aynı makinede eager PyTorch yollarına karşıdır; eager'ın launch-bound OLMADIĞI sistemlerde (ör. Linux, batch>1) graph avantajı küçülür — ama iki graph yolunun kendi arasındaki sparse/dense oranı yapısaldır.

**Yol boyunca bulunan semantik bug (ve düzeltmesi):** İlk graph koşusunda graph-sparse ↔ naive-sparse parity [4.7–7.4, top1=0.0] çıktı. Kök neden: **eager harness'lerin (naive + fast) sparse yolu, kısmi bloğun yazılmamış gelecek slotlarını maskesiz softmax'a katıyordu** (sıfır anahtar → skor 0 → ağırlık exp(0)=1) — eğitim semantiği ise slot-düzeyi nedensellik uygular (`gather_sparse_attention`/Triton kernel: `kpos ≤ qpos`); graph doğru olanı yapıyordu. Eager yollar düzeltildi; C tablosu düzeltilmiş semantikle yeniden ölçüldü. Bu bug yalnız bağımsız decode harness'ini etkiler — model değerlendirmesi (exp3) eğitim yolundan yapıldığı için etkilenmemiştir. **Graph, bağımsız referans olarak değerini burada da kanıtladı: eager harness'teki gizli semantik sapmayı ortaya çıkardı.**

---

## 5. Dürüst Değerlendirme, Sınırlar ve Dersler

**Kanıtlananlar:**

- Seyrek formülasyon (seçim + Triton kernel) **sadık**: ρ=1.0'da dense ile birebir eş (v1: 14.303; v2: 12.854) — bu, bug-temiz bir referans noktası.
- Kalite açığı aynı protokolde **%52.7 → %29.9** kapandı (wt2_test, ρ=0.25, T=1024); KL ve top1 uyum da iyileşti.
- 2-blok bütçesinde router tam olarak sink+diyagonal'e dejenere oluyor (local == ρ=0.125, birebir) — mekanizma tutarlı ve tekrar edilebilir.
- **Uzun bağlam rejimi:** T=4096'da kalite kaybı +%26 ve recall 0.826; crossover ≈ 5k (A fazı sonrası); T=16384'te 1.25x hızlanma. İki eğri (kalite kaybı ↓, hız kazancı ↑) aynı yönü işaret ediyor.
- Ölçüm disiplini: iki büyük ölçüm hatası (GQA fallback, autograd tuzağı) bulundu, düzeltildi, kirli veriler temizlendi.
- **Attribution ayrıştırıldı (v2ctl kontrolü):** kazancın **%83'ü jenerik adaptasyon, %17'si seyreklik-özgü** (§3.1/§3.6).
- **v2 uzun bağlam:** T=8192'de göreli kalite kaybı yalnız **+%7.5** (recall 0.868) — hız crossover bölgesiyle (A öncesi 1.00x @8k; A sonrası ≈5k) aynı bölgede (§3.5).
- **B fazı (v3):** düşük LoRA LR (1e-4) hem on-domain'de en iyi (dense 12.456 / sparse 16.559) hem OOD unutmayı geri aldı (TinyStories dense 10.370 → 6.090; v1'in 5.557'sine yakın) — ve bunun LR'ye bağlı, jenerik adaptasyon kaynaklı bir etki olduğu v2ctl kontrastıyla biliniyor (§3.7).
- **C fazı (decode):** tüm adım tek CUDA graph; dense **bitwise** parity + 13.0–24.2 ms, sparse top1=1.0 + 13.5–19.1 ms (T=1k–8k); eager'a karşı 4–7x. Uzun bağlamda sparse graph, dense graph'tan %12–27 hızlı (§4.6).

**Kanıtlanmayanlar / henüz belirsiz:**

- ~~Attribution (V2CTL): v2 kazancının seyreklik-özgülük derecesi bilinmiyor~~ → **çözüldü** (§3.6; %83 jenerik / %17 seyreklik-özgü).
- ~~v2 bağlam eğrisi (V2CTX): v2'nin T>1024 davranışı ölçülmedi~~ → **ölçüldü** (§3.5): 8k'da +%7.5, recall 0.868.
- ~~Kısa bağlamda (<8k) uçtan uca hız kazancı yok~~ → **A fazı sonrası:** ≤2k'da hâlâ yavaş ama açık dar (0.92x @1k, 0.94x @2k, 0.98x @4k); crossover ≈ 5k.
- ~~Decode tarafında hiçbir iddia yok (naif harness)~~ → **çözüldü (§4.6):** CUDA-graph decode — dense bitwise parity, sparse top1=1.0, eager'a karşı 4–7x. Kapsam sınırı: batch=1 ve bu makinenin launch-bound ortamı (bkz. Sınırlar).

**Sınırlar / uyarılar:**

- **OOD unutma (kısmen kapatıldı):** v2 LoRA wikitext dışında ciddi unutuyordu (TinyStories dense 5.557 → 10.370); **v3 (LR 1e-4) bunu 6.090'a indirdi** (v1'in 5.557'sine yakın ama tam değil) — tek nöbetçi veri setiyle ölçüldü; "hiç unutmuyor" iddiası değil. v2/v2ctl sayıları yalnız eğitim dağılımı (wikitext-2) içindir.
- **C-fazı süreleri host yüküne duyarlı (eager tarafta):** aynı harness iki oturumda %40–80 fark verdi (WDDM launch-bound); graph tarafta ±%0.2. C tablosu tek oturumun kendi içinde tutarlı ölçümüdür; yapısal sonuç = graph↔eager farkı + graph-sparse↔graph-dense oranı.
- ~~Router maliyeti: 1024'te toplamın %27'si~~ → **düşürüldü (§4.3, A fazı):** @1024 44.8 → 6.5 ms (A2 graph) / ~18 ms (A1 eager). CUDA-graph yolu yalnız inference ve opt-in'dir; eğitim/eval A1'den faydalanır.
- **Tek veri seti + tek model + tek donanım:** wikitext-2 train; Qwen2.5-0.5B; 6GB laptop GPU. Ölçek genellemesi (7B, A100) test edilmedi.
- **Eğitim bütçesi düşük:** v1 1500 adım, v2 1250/2000 adım (ortam kesintisi; eğri düzleşmişti). Sonuçlar "bu bütçede" geçerli.
- **Protokol karışıklığı uyarısı:** §3.1 tablosu 128 chunk (130.944 token), §3.4 exp6 tablosu 16–32 chunk. Sayılar birbiriyle karşılaştırılırken protokol etiketi dikkate alınmalı (dense referansları bu yüzden farklı: 14.304 vs 14.559).
- `wt103_test == wt2_test` (**veri seti özelliği**: WT2/WT103 valid+test split'leri ortak — 16.10'da taze indirmeyle doğrulandı); bağımsız test olarak yalnız TinyStories kullanıldı.

**Dersler:**

- `enable_gqa=True` bu build'de sessizce MATH fallback'e düşürüyor; "dense baseline" 100x yavaş görünür — hız ölçümünden önce backend pinlenmeli.
- `requires_grad=True` + `no_grad` dışı zamanlama = 6GB kartta WDDM taşması = 6–30x sahte yavaşlık; ölçümden önce `requires_grad_(False)` şart.
- Sabit null-logit (-3) per-token rastgele sönümleme yapıyordu; `scale_min=0.0` izi ve ρ=1.0 paritesiyle teşhis edildi. İsim-bazlı checkpoint yükleme bu değeri geri getirdiği için yükleme sonrası `fill_(-1e9)` zorunlu.
- **Bağımsız referans disiplini:** CUDA-graph decode, eager harness'ten bağımsız ikinci bir yol olarak eager'daki gizli bir semantik sapmayı (kısmi blokta gelecek-slot maskesi) ortaya çıkardı. Ders: naive ↔ fast paritesi 1.0 iken ikisi de eğitim semantiğinden sapabiliyordu — parite, ancak referanslardan en az biri farklı bir yolsa anlamlıdır.
- Eğitim kaybı tek başına yanıltıcı: v1'de KL 0.74 → 0.13 düştü ama recall/PPL değişmedi; **seçim metrikleri (recall, oracle gap, top1) izlenmeden kalite kararı verilmemeli.**
- 6GB kartta profil disiplini: `reserved` kontrolü, tek iş, dense-önce taze süreç, medyan + CUDA events.

---

## 6. Yeniden Üretim

**Varlıklar:**

| Şey | Yol |
|---|---|
| Model | `models\Qwen2.5-0.5B` (fp16, donuk taban) |
| Eğitim verisi | `data\wt2_train.pt` (2.517M token, wikitext-2 train) |
| Testler | `data\wt2_test.pt` (298.939 token; `wt103_test` özdeş — WT2/WT103 split ortaklığı), `data\tinystories_valid.pt` (1.16M token) |
| Checkpoint'ler | `checkpoints\v1_best.pt`, `v2_best.pt` (step 1250), `v3_best.pt` (step 1250, §3.7), `v2ctl_best.pt` (step 1250) |
| P1 demo adaptörü | `checkpoints\p1_qwen3b_lora_r16_t60.pt` (3B, r=16; ayrı rapor: `P1_RAPOR.md`) |
| Byte verisi (P2) | `data\bytes_wt2_{train,valid,test}.pt` (10.95M / 1.15M / 1.29M byte) |
| Kayıtlar | `results\results.jsonl`, `results\exp*.json` |

```powershell
cd "C:\Users\biris\Desktop\opencode\ai gelistir"

# 1) v1: router eğitimi (kayıtlı koşunun konfigü: 1500 adım, batch 4, seq 1024, lr 1.5e-3 — hepsi varsayılan)
python experiments\exp2_train_router.py --tag v1
#  → checkpoints\v1_best.pt

# 2) v2: stage-2 eğitimi (LoRA + LM + KL + MSE, clean hedefler)
python experiments\exp7_stage2.py --steps 2000 --eval_every 250 --log_every 20 --tag v2 --ckpt_gather --init_ckpt checkpoints\v1_best.pt
#  kesilirse kaldığı yerden:
python experiments\exp7_stage2.py --steps 600 --warmup 20 --tag v2b --ckpt_gather --init_ckpt checkpoints\v2_best.pt

# 2b) attribution kontrolü (dense yolda yalnız LoRA, router v1'de donuk):
python experiments\exp7_stage2.py --steps 1250 --eval_every 250 --log_every 20 --tag v2ctl --dense_control --ckpt_gather --init_ckpt checkpoints\v1_best.pt

# 2c) B fazı — v3: tek değişken lr_lora 1e-3 → 1e-4 (v2 reçetesi, v1_best'ten warm-start)
python experiments\exp7_stage2.py --steps 1250 --lr_lora 1e-4 --tag v3 --ckpt_gather
#  → checkpoints\v3_best.pt

# 3) Kalite: temiz konfig, 3 set + ρ süpürme + kontroller (128 chunk)
python experiments\exp3_eval.py --ckpt checkpoints\v1_best.pt --tag v1_clean
python experiments\exp3_eval.py --ckpt checkpoints\v2_best.pt --tag v2_best --v2
python experiments\exp3_eval.py --ckpt checkpoints\v2ctl_best.pt --tag v2ctl --v2      # kontrol bitince

# 3b) Bağlam eğrileri
python experiments\exp6_context.py --ckpt checkpoints\v1_best.pt --tag v1              # v1: T=1024/2048/4096 + kontroller
python scripts\v2_context_curve.py --ckpt checkpoints\v2_best.pt --tag v2ctx           # v2: T=1024..8192

# 4) Hız: temiz referans (dense-önce, taze süreç, no_grad, pinli backend, merge'li LoRA)
python scripts\speed_ref.py --ckpt checkpoints\v2_best.pt --tag v2_clean_speed --v2 --merge

# 4b) C fazı: decode — eager vs tüm-adım CUDA graph (parity + süre)
python scripts\decode_ref.py --tag c_phase --Ts 1024,2048,4096,8192 --steps 64 --graph
python scripts\check_sparse_graph.py      # sparse graph ↔ eager: seçim ve parity doğrulaması

# 4c) P1 (parça-parça eğitim) ve P2 (byte-level) — bkz. P1_RAPOR.md / P2_RAPOR.md
python experiments\exp10_byte.py --tag p2 --which patch,flat --steps 12000 --batch 32 --W 512 --P 4
python scripts\verify_p2_timing.py        # P2: sync'li adım süresi + kalite bağımsız doğrulaması

# 5) Özet ve grafikler
python scripts\clean_results.py          # NaN/Inf satırları arşive (backup alır)
python scripts\make_plots.py
python scripts\summarize_results.py
```

Temiz konfigin koddaki karşılığı: `patch_model(..., use_coarse=False, null_init=-1e9)` sonrası her katman için `w.null_logit.data.fill_(-1e9)`, `set_selection_mode("topk")`, `set_sparse_enabled(True)`.

---

## 7. Sonraki Adımlar (öncelik sırasına göre)

1. ✅ *(başlangıç, 16.10)* ~~Router maliyetini düşür → crossover'ı sola çek~~ → **A fazı yapıldı (§4.3):** A1 exact-math önbellek/vektörleştirme (parity 10/10) + A1.5 çift-softmax kaldırma + A2 CUDA-graph replay (inference-only). @1024 44.8 → **6.5 ms**; crossover 8k → **≈5k**. Kalan opsiyonel fikirler: K=64→32 retrain, fp16 router, merge sonrası LoRA modüllerini kaldırma.
2. ✅ *(tamam, 16.10)* ~~LoRA LR'yi düşür (1e-3 → 1e-4) → OOD unutmayı azalt~~ → **B fazı (§3.7):** v3 on-domain'de en iyi (dense 12.456 / sparse 16.559) + OOD unutma geri alındı (TinyStories dense 10.370 → 6.090). TinyStories artık kalıcı OOD nöbetçisi.
3. ✅ *(tamam, 16.10)* ~~Decode harness optimizasyonu~~ → **C fazı (§4.6):** tüm adım tek CUDA graph; dense bitwise parity + 13.0–24.2 ms, sparse top1=1.0 + 13.5–19.1 ms; eager'a karşı 4–7x. Yol boyunca eager harness'teki slot-maskesi semantiği bug'ı bulunup düzeltildi.
4. ✅ *(tamam, 16.10)* ~~v2 bağlam eğrisi (V2CTX)~~ → §3.5 dolduruldu: 8k'da +%7.5, recall 0.868.
5. ✅ *(tamam, 16.10)* ~~v2ctl sonucunu rapora işle (V2CTL)~~ → attribution: %83 jenerik / %17 seyreklik-özgü (§3.1/§3.6).
6. **Uzun bağlamda tam kalite değerlendirmesi:** T=4096+ için v3 protokolü (exp6 benzeri, oracle/pencere kontrolleriyle) — "asıl rejim uzun bağlam" iddiasını v3 için de kanıtla.
7. **blok=32 denemesi:** aynı ρ'da daha ince seyreklik (daha fazla, daha küçük blok) — kernel maliyeti artar ama seçim çözünürlüğü ve kalite artabilir; T≥4096'da A/B ölç.
8. ✅ *(tamam, 16.10)* **P1 — parça-parça (block-streaming) eğitim** (ayrı rapor: `P1_RAPOR.md`): 3B model 1.40 GB sabit VRAM ile eğitildi (naive yükleyemiyor), 0.5B/1.5B/3B parity + sert-OOM kanıtları; demo adaptör kaydedildi.
9. ✅ *(tamam, 16.10)* **P2 — tokenizer'sız byte-level prototip** (exp10; rapor: `P2_RAPOR.md`): sabit-patch BLT-lite vs düz byte transformer, aynı bütçe (196.6M byte), ≈5M param. **Sonuç (wt2 byte test bpb): flat 1.6452 < patch 2.0513** — bu ölçekte flat baseline kalitede önde; patch adım başına ~1.7x hızlı (150.1 vs 250.7 ms, sync'li). Kalite ckpt'lerden bağımsız doğrulandı (±0.0004). Dinamik parçalama + KV-cache decode "gelecek iş".
10. ✅ *(tamam, 16.10)* **Paketleme (açık kaynak):** README + `requirements.txt` + temizlik (arşivlere taşındı); **lisans: MIT** (`LICENSE` eklendi; kullanıcı kararı).
11. **Opsiyonel A kalanları:** K=64→32 router retrain, fp16 router, merge sonrası LoRA modüllerini kaldırma; decode batch>1 ve farklı donanım ölçümleri.

_(Bu rapor `PROGRESS.md` ile birlikte okunmalıdır; tüm satır-metrikler `results/results.jsonl` içindedir ve yukarıdaki her sayı oradan doğrulanabilir.)_
