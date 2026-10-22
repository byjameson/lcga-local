# P2 — Tokenizer'sız Byte-Level Prototip (BLT-lite, sabit patch): Deney Raporu

> **Sonuç çerçevesi (dürüst):** Aynı bütçe ve ≈5M parametreyle iki model eğitildi.
> Bu ölçekte **flat byte transformer kalitede önde** (wt2 byte test: **1.6452 bpb** vs
> patch 2.0513), **patch model ise adım başına ~1.7x hızlı** (150.1 vs 250.7 ms; sync'li
> doğrulama). Yani sabit-P=4 BLT-lite prototipi flat baseline'ı GEÇMEDİ; BLT'nin asıl
> kazanımları dinamik parçalama + ölçekle gelir (bkz. §4, "gelecek iş").
> **Ölçüm notu:** exp10 koşusunun kendi `step_med_ms` değeri flat için WDDM
> enqueue-artefaktıydı; kanonik süreler `scripts/verify_p2_timing.py` ile ölçüldü
> (`results/exp10_timing_verify.json`). Ayrıntı §3.

## 1) Tasarım ve bütçe

| | FlatByteLM (baseline) | BytePatchLM (BLT-lite) |
|---|---|---|
| Mimari | düz nedensel byte transformer | yerel encoder (2 katman, 64d) → 128 patch token üzerinde global transformer (6 katman, 256d) → yerel AR byte decoder |
| Parametre | **4.94M** | **5.02M** |
| Girdi | 512 byte | 512 byte = 128 × 4-byte sabit patch (P=4) |

- Ortak: dim 256 / 6 global katman / 8 head; hedef = sonraki byte; **hedef kümeleri
  özdeş** (pencerenin ilk byte'ı hariç — `lcsa/byte_lm.py`: flat `tgt = x[:,1:]`,
  patch `m[:,0,0] = 0`).
- Veri (wikitext-2 ham byte): train 10.95M / valid 1.15M / test 1.29M byte
  (`data/bytes_wt2_*.pt`).
- Bütçe: 12,000 adım × batch 32 × 512 byte = **196.6M byte görülen**; AdamW,
  lr 6e-4, warmup 200, cosine → 0.05×; seed 0; iki model **birebir aynı CLI**.
- Metrik: **bpb = NLL/byte / ln2**; valid 64 sabit pencere, test 128 sabit pencere
  (iki modelde özdeş pencereler).

## 2) Sonuçlar

| model | param | en iyi valid bpb | **test bpb** | adım (sync'li) | 12k adım temposu | örnek üretim (240B) |
|---|---|---|---|---|---|---|
| flat | 4.94M | 1.6582 | **1.6452** | 250.7 ms | ≈50 dk (duvar 49.7) | 1.5 s |
| patch | 5.02M | 2.0952 | 2.0513 | 150.1 ms | ≈30 dk | 1.1 s |

- Δ (patch − flat) = **+0.406 bpb** (+%24.7 bağıl NLL/byte).
- Eğriler doyuma yakın (son 2000 adımda ikisi de <0.03 iyileşti):
  flat 8.11 → 1.6582; patch 8.04 → 2.0952. Eğri grafiği: `results/plots/p2_valid_bpb.png`.
- Örnekler (temp 0.8, top-k 50; tam metinler JSONL'de) — ikisi de kelime-benzeri akıcı:
  - patch: `Eval offer which Hang @-@ Georgian stee and was land as Swamp and agreed on 191 September 28 , 2007 .`
  - flat: `Israel points have been divided on all times that land and white speaks at the chapters and became safe an insect .`

## 3) Ölçüm doğrulaması ve bulunan tuzak

`scripts/verify_p2_timing.py` (iki koşu; ckpt'ler diskten yeniden yüklenir, her adımda
`torch.cuda.synchronize()`):

- **Kalite yeniden ölçümü:** flat valid 1.6579 / test 1.6449; patch 2.0947 / 2.0509 →
  kayıtlarla **±0.0004** (CUDA redüksiyon non-determinizmi). Kalite iddiası bağımsız
  doğrulandı.
- **Süreler:** flat **250.7 ms/adım** (sync'siz 247.0), patch **150.1 ms/adım**
  (sync'siz 154.1); iki koşuda stabil. `eval_bpb(64 pencere)` ≈ 0.1 s → eval maliyeti
  ihmal edilebilir.
- ⚠️ **Tuzak:** exp10'un eğitim döngüsü sync'siz zamanlıyordu; flat için yazılan
  `16.55 ms/adım` ve `train_min 3.45` **CPU-enqueue** ölçümüydü. Gerçek duvar süresi
  (49.7 dk) 250.7 ms × 12,000 ile birebir uyumlu. Patch kaydı (68.56 ms) da kısmi
  sync'lere rağmen ~2.2x düşüktü. **Kanonik sayılar yalnız doğrulama koşusundan**;
  JSONL'deki ham satırlar tarihçe olarak duruyor + `exp10_byte_verify` düzeltme satırı
  eklendi (`results/exp10_timing_verify.json`). Eğitim döngüsü adım-başı sync ile
  düzeltildi (gelecek koşular).
- Gerçek eğitim verimi (196.6M byte bütçesi): flat ≈ 66 KB/s, patch ≈ 109 KB/s
  (fp32, tek 3050, eager; prototip ölçeği).

## 4) Yorum: ne öğrendik

1. **Sabit-patch BLT-lite bu ölçekte kazandırmadı.** Patch kalitede geride (+0.41 bpb
   test) ama adım başına ~1.7x hızlı: global transformer 128 patch token gördüğü için
   compute'u ~4x kısa dizide. Sonuç: "hızlı ama daha zayıf".
2. **Bu kıyas BLT'nin asıl rejimini temsil etmez (dürüst sınır):** BLT'nin literatürdeki
   kazancı entropy-tabanlı **dinamik** parçalama + büyük ölçekten gelir; sabit P=4 mini
   prototip o rejimi test etmez. Bu deney "tokenizer'sız byte-level eğitim hattı çalışıyor
   ve ölçülüyor" kanıtıdır; mimari iddiası taşımaz.
3. **Bütçe eşitliği ≠ compute eşitliği:** aynı görülen byte, aynı parametre mertebesi;
   mimari compute serbest — patch'ın hızı ve flat'ın kalitesi tam da bu takasın iki ucu.
4. **Ölçüm disiplini dersi:** eğitim döngüsünde de sync'siz zamanlama = uydurma riski
   (WDDM); P2 bunu yakalayıp düzeltti (RAPOR §4.5 trap #6).

## 5) Sınırlar

- Tek seed (0), tek koşu/model; eğri doyuma yakın ama LR takvimi sonu.
- Sabit valid/test pencereleri (ilk 64/128); train/valid/test byte düzeyinde ayrık.
- Çıkarım hızı iddiası yok: KV-cache'li decode / dinamik parçalama yazılmadı;
  `generate_*` naif (her byte için tam forward).
- bpb karşılaştırması byte düzeyinde; tokenizer'lı modellerle birebir kıyas değildir.
- Tüm süreler tek 3050'de, fp32 eager; başka donanımda mutlak değerler değişir.

## 6) Yeniden üretim

```bash
python experiments\exp10_byte.py --tag p2 --which patch,flat --steps 12000 --batch 32 --W 512 --P 4
python scripts\verify_p2_timing.py
python scripts\make_plots.py        # results/plots/p2_valid_bpb.png
```

Artefaktlar: `checkpoints/p2_patch.pt`, `checkpoints/p2_flat.pt`;
`results/results.jsonl` satırları (`exp10_byte` patch+flat, `exp10_byte_verify`);
`results/exp10_timing_verify.json`; flat koşu logu `results/p2_flat_run.log`.

Not: birleşik koşu patch'ten sonra düştüğü için flat, birebir aynı CLI ile ayrı
süreçte yeniden koşuldu (patch fazı etkilenmedi).
