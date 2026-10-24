# P6 RAPOR — "Senkron Maliyeti": Parçalı / Senkronsuz Eğitimin Kalite Bedeli (Mini Deney)

> **Soru (kullanıcı fikri):** "Modeli 100 parçaya bölüp her GPU'ya atsak; GPU'lar arası
> transfer olmasa daha hızlı eğitmez mi?"
>
> Tek RTX 3050'de **gerçek ağ trafiği ölçülemez**. Bu deney onun yerine asıl kritik
> soruyu ölçer: **iletişimi (gradyan senkronunu) azaltmak/kaldırmak modelin kalitesini
> ne kadar düşürür?** Kod: `experiments/exp13_local_sgd.py`; kanıt: `results/e13_*.log`
> ve `results/results.jsonl` (`exp13_local_sgd` satırları); tarih: 18.10.2026.

## 1) Kurulum

- **Model:** mini LM, 17.31M param (8 katman, dim 384, ctx 512), ctrl8k tokenizer.
- **Veri:** `tok2_ctrl8k_train` (221M token) → **4 bitişik shard** (~55.3M token/işçi).
- **Bütçe (her kolda eşit):** 1000 optimizer adımı/işçi × 4 işçi × 32×512 = **65.5M token**.
- **Kollar:**
  1. `e13_full` — tek model; her adımda 4 shard'dan 1'er mikro-batch, gradyan ortalaması →
     **DDP+Adam'ın matematiksel eşdeğeri** (her adımda senkron).
  2. `e13_local250` — 4 replica; 250 adımda bir ağırlık ortalaması → **iletişim 250× az**.
  3. `e13_nosync` — 4 replica; **sıfır senkron**; sonunda bireysel + soup (ağırlık ortalaması).
- **Doğrulama:** selftest (birikim eşdeğerliği maks fark **3.95e-06**; senkron tam birebir)
  + CPU smoke ×3 + GPU smoke ✓. Ckpt'ler: `checkpoints/e13_*.pt` (nosync'ta 4 replica + soup).

## 2) Sonuçlar

**Final bpb (aynı token bütçesi):**

| Kol | final bpb | full'e göre | Not |
|---|---|---|---|
| `e13_full` (her adım senkron) | **0.6190** | — | referans |
| `e13_local250` (250× az iletişim) | 0.7117 | **+%15.0** | 4 sync; sonuncusu eğitim sonunda |
| `e13_nosync` (sıfır iletişim; 4 modelin ort.) | 0.7310 | **+%18.1** | en iyi replica 0.7285 |
| `e13_nosync` soup (tek modele indirme) | 0.8820 | **+%42.5** | farklı çözüm havzaları |

**Ara noktalar (bpb):**

| Kol | @250 | @500 | @750 | @1000 |
|---|---|---|---|---|
| full | 0.8596 | 0.7027 | 0.6442 | **0.6190** |
| local250 | 1.0155 | 0.8129 | 0.7394 | **0.7117** |
| nosync (ort.) | 1.0025 | 0.8426 | 0.7643 | **0.7310** |

**Sapma (replica'lar arası ortalama rel-L2; sync'ten önce ölçülür):**
- `local250`: 0.352 → 0.319 → 0.185 → **0.075** (ortalamalar tutuyor; LR düştükçe sapma düşüyor)
- `nosync`: 0.352 → 0.516 → 0.556 → **0.562** (monoton birikiyor)

Süreler (tek GPU'da sırayla): full 31.4 dk (1.87 s/adım — adım başına 4 mikro-batch),
local250 32.0 dk, nosync 32.2 dk.

## 3) Yorum (dürüst)

1. **Senkronu tamamen silmek "tek model"i kaybettiriyor:** 4 bağımsız replica'nın bireysel
   kalitesi ~0.73 (+%18); üstüne **sonradan ortalama (soup) felaket** (0.88, +%42) — çünkü
   uzun bağımsız eğitim modelleri **farklı çözüm havzalarına** taşıyor. "Önce ayrı eğitip
   sonra birleştiririz" bu ölçekte çalışmıyor.
2. **Seyrek senkron (250× az iletişim) bile görünür bedel getiriyor: +%15 bpb.** Sebep:
   senkron kol adım başına 4× büyük **etkin batch (128)** ile daha az gürültülü ilerliyor;
   replikalar 32'lik gürültülü adımlar atıp kopukluğu aralıklı ortalamayla kapatmaya çalışıyor.
3. **Sapma eğrisi mekanizmayı doğruluyor:** ara ara ortalamak (local250) replikaları "aynı
   havzada" tutuyor (sapma ≤0.35 ve düşüyor); hiç ortalamamak monoton sapma üretiyor (0.56)
   ve sonunda ortalama uyumsuz oluyor.
4. **"Transfer yok" = kazanç değil, kayıp:** iletişimi kesmek modeli N'e böler, hızlandırmaz.
   Gerçek dünyada iletişimi 250× azaltan yöntemler var (local SGD, DiLoCo tarzı); kazanç
   **iletişim maliyetinde**, bedel **kalitede**. Bizim ölçekte bedel büyük; büyük ölçek +
   büyük batch'te küçülebilir (açık soru). P1 bulgusuyla tutarlı: parçalama **bellek**
   kazandırır, hız için değil.

## 4) Sınırlar / sonraki adımlar

- (a) Küçük ölçek + küçük batch (32/işçi): bedel üst sınıra yakın olabilir.
- (b) Shard'lar homojen (TinyStories) — heterojen çok-işçili veride sapma daha büyük olur.
- (c) Tek GPU'da gerçek ağ trafiği / all-reduce süresi ölçülmedi (17M modelde zaten ihmal
  edilebilir düzeyde olurdu — kritik olan kalite bedeliydi).
- (d) Adam durumları replica-bazlı (yaygın pratik); "momentleri de ortala" varyantı denenebilir.
- İstenirse: `sync_every` ∈ {50, 100} ek kollarla eğri sıklaştırılabilir; `exp13` hazır (~30 dk/kol).
