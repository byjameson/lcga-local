# P1 — "Parça Parça" (Blok-Streaming) LoRA Eğitimi: Rapor

> Amaç: tek RTX 3050 Laptop 6GB ile, VRAM'e sığmayan modelleri LoRA ile
> eğitebilmek. Kod: `lcsa/stream_train.py` (eğitici) + `experiments/exp8_stream.py`
> (deney/CLI). Tüm sayılar `results/exp8_*.json` ve `results/results.jsonl`
> (`exp8_stream` satırları) içinden doğrulanabilir. Tarih: 16.10.2026.

---

## 1. Özet (TL;DR)

- **3B model (5.76GB fp16 ağırlık) 6GB kartta eğitildi:** streaming eğitici peak
  **1.40GB** VRAM kullandı; naive (tüm model GPU'da) yol aynı koşulda **OOM**.
- **1.5B (T=1024):** naive **10.89GB** rezerve etti — 6GB VRAM'i aşınca Windows
  WDDM paylaşımlı RAM'e taşıdı ve adım **13.5s**'ye çıktı; streaming **1.09GB**'da
  kaldı, adım **4.6s** (≈3x hızlı).
- **Sayısal doğruluk:** 0.5B ve 1.5B'de naive ile streaming eğitiminin loss
  eğrileri aynı (maks fark ~9e-4; fp16 chunk gürültüsü). Aynı batch'ler, aynı
  LoRA init (seed) — `results/exp8_*.json` içinde satır satır.
- **Sert OOM kanıtı:** süreç-başı VRAM tavanı (`torch.cuda.set_per_process_memory_fraction`)
  ile: 0.5B'de 2.1GiB tavanda naive OOM / stream 0.77GB ile eğitti; 3B'de aynı
  tavanda naive daha yüklenemedi / stream eğitti.
- **Doğru iddia:** "VRAM'e sığmayan modelleri eğitilebilir kılar ve sığmayan
  rejimde naive'e göre belirgin hızlıdır." Yanlış iddia olurdu: "her zaman
  daha hızlı" — VRAM'e sığan modellerde (0.5B) naive açık ara hızlı (0.3-0.6s vs
  1.1-1.9s/adım).
- **Gerçek eğitim demosu (3B, 60 adım):** sabit 1.40GB VRAM, held-out CE
  2.3302 → **2.2143** (−0.116), adapter diske kaydedildi (7.37M param).

---

## 2. Mekanizma

Bellek hiyerarşisi (her adımda):

- **GPU'da sabit:** `embed_tokens` + `rotary_emb` + `norm` + `lm_head`
  (tied ise embed ile aynı tensör) — 3B'de toplam ≈ 0.62GB.
- **CPU RAM'de sabit:** tüm transformer blokları (fp16, donuk) — 3B'de ≈ 5.76GB.
- **Akış:** forward'da blok sırayla GPU'ya kopyalanır → `no_grad` forward →
  CPU'ya geri; her bloğun **girdi gizli durumu (h_in, `B×T×d`)** GPU'da saklanır.
  Geri yayılım ters sırada: blok yeniden GPU'ya → **gradyanlı recompute**
  (blok-bazlı activation checkpointing) → `backward` → LoRA gradyanları CPU
  akümülatöre toplanır → blok CPU'ya döner.
- **Kayıp:** pozisyon-chunk'lı CE (`loss_chunk=128`); logit tensörü hiçbir zaman
  tam `T×vocab` olarak GPU'da durmaz (3B'de 1024×151936 fp32 ≈ 0.62GB tasarrufu).
- **Optimizer:** AdamW, yalnız LoRA A/B (fp32, ~2-7M param), CPU'da adım atar.

**Peak VRAM modeli:** `embed+head + 1 blok + 1 bloğun aktivasyonları + h_in
cache (L×T×d) + CE chunk` — **model derinliğinden bağımsız**. 3B (36 katman)
ölçümü 1.40GB; 0.5B (24 katman) 0.77GB — model büyüdükçe sabit kalır, sadece
blok boyutu ve d küçük oranda büyür.

**Doğruluk argümanı:** recompute backward, blok listesinin olağan autograd
backward'ıyla matematiksel olarak aynıdır (aynı modüller, aynı girdiler).
Bunun ampirik karşılığı Bölüm 3'teki parity tablosudur.

---

## 3. Kanıtlar

### 3.1 Sayısal parity (naive ↔ stream, aynı batch'ler, aynı seed)

| Model | T | naive loss (ilk 3 adım) | stream loss (ilk 3 adım) | maks fark |
|---|---|---|---|---|
| Qwen2.5-0.5B | 512 | 3.2777 / 2.9113 / 2.5149 | 3.2777 / 2.9106 / 2.5143 | 6.8e-4 |
| Qwen2.5-1.5B | 1024 | 2.1929 / 1.9365 / 2.6472 | 2.1929 / 1.9365 / 2.6481 | 9.1e-4 |

Farklar fp16 matmul + chunk sınırlarından; eğitim eğrileri pratikte birebir.
Aynı koşuların tekrarında kayıplar **birebir** aynı çıktı (deterministik akış).

### 3.2 VRAM ve adım süresi (naive vs stream)

| Koşu | Model | Tavan | naive | stream |
|---|---|---|---|---|
| `exp8_cap035_05b` | 0.5B, T=512 | 2.1GiB | **OOM** | peak **0.77GB**, 1.1-1.9 s/adım |
| `exp8_15b_parity` | 1.5B, T=1024 | — (6GB) | peak **10.89GB** (WDDM taşması), 12.9-13.6 s/adım | peak **1.09GB**, 3.9-4.8 s/adım |
| `exp8_3b_cap035` | 3B, T=1024 | 2.1GiB | **OOM** (yükleyemedi) | peak **1.40GB**, 7.3-8.7 s/adım |

Ek: `exp8_smoke` (0.5B, T=512, tavansız): naive 0.26-0.64 s/adım (peak 3.64GB),
stream 1.1-1.5 s/adım (0.77GB) → **sığan rejimde naive hızlı** (dürüst sınır).
Grafik: `plots/p1_streaming.png`.

### 3.3 3B'de gerçek eğitim demosu (60 adım, held-out CE, adapter kaydı)

`exp8_3b_train60` — Qwen2.5-3B, T=1024, LoRA r=16 (yalnız attention, 7.37M
param), lr=2e-4, clip=1.0, 60 adım (≈7.6 s/adım, toplam ≈7.5 dk):

| Ölçüm | Değer |
|---|---|
| peak VRAM (tüm koşu boyunca sabit) | **1.40GB** |
| host RSS (CPU ağırlıklar dahil) | 7.03GB |
| held-out CE (wt2_test, 8×1024, streaming eval) önce → sonra | 2.3302 → **2.2143** (−0.1159) |
| adapter | `checkpoints/p1_qwen3b_lora_r16_t60.pt` (288 tensör = 36 katman × 4 proj × A/B) |

60 adımda yalnız attention LoRA'sı güncellendi (taban donuk); held-out CE'nin
düşmesi eğitimin gerçekten öğrendiğini gösteriyor (demo amaçlı kısa koşu —
kalite iddiası değildir). Adım kayıpları batch=1 gürültüsüyle 1.5–2.7 arası
salınır; eğilim hafif aşağı. Sayılar `results/exp8_exp8_3b_train60.json`
içinde satır satır durur.

---

## 4. Sınırlar ve dürüst notlar

1. **Windows WDDM tuzağı:** 6GB VRAM'i aşan rezervasyon OOM vermek yerine
   paylaşımlı RAM'e taşınır. Bu yüzden "sığmaz" kanıtını süreç-başı **sert
   allocator tavanı** ile verdik (`--mem_fraction`/`--vram_gb`); 1.5B naive'in
   tavansız "çalışması" 10.89GB rezerve + 13.5 s/adım ile taşma rejimidir.
2. **Her zaman hızlı değil:** VRAM'e sığan modellerde naive daha hızlıdır
   (0.5B: 0.3-0.6s vs 1.1-1.9s). Streaming'in kazancı sığmayan modelde
   *eğitilebilirlik* + naive'in taşma yavaşlamasına karşı hızdır.
3. **PCIe trafiği:** 3B'de adım başına ~12-25GB H2D/D2H kopya (2 geçiş × 5.76GB
   + ara buffer'lar) → adım süresinin baskın kısmı transfer. Pinned memory,
   blok penceresi, 4-bit CPU ağırlık gibi optimizasyonlar yapılmadı (sonraki iş).
4. **Prototip kapsamı:** yalnız Qwen2/Qwen2.5 aile düzeni (`model.model.layers` +
   `rotary_emb`), LoRA-only (taban donuk), fp16, tek GPU, batch=1, CPU AdamW.
   Full fine-tune değil; grad clipping opsiyonel (`--clip`), LR takvimi yok.
5. **Host RAM şartı:** modelin fp16 CPU kopyası (3B: RSS ≈ 7.5GB; 1.5B: ~4GB;
   0.5B: ~1.8GB). 34GB RAM'li bu makinede rahat; 8-16GB'lık makinelerde 3B sınırda.
6. **Sayısal tolerans:** parity maks farkı ~1e-3 fp16'dandır; bitwise iddia
   yok. (Router C tarafında bitwise gate vardı; eğitim yolu için chunk'lı CE
   farkı kabul edilen tek farktır.)

---

## 5. Kullanım

```powershell
# Parity doğrulaması (naive vs stream, aynı batch/seed)
python experiments/exp8_stream.py --model models/Qwen2.5-0.5B --mode both --steps 3 --seq 512 --tag parity

# Sert VRAM tavanında kanıt (naive OOM beklenir, stream eğitir)
python experiments/exp8_stream.py --model models/Qwen2.5-3B --mode both --steps 3 --seq 1024 --mem_fraction 0.35 --tag 3b_cap

# Gerçek eğitim + held-out CE + adapter kaydı
python experiments/exp8_stream.py --model models/Qwen2.5-3B --mode stream --steps 60 --seq 1024 `
    --lr 2e-4 --clip 1.0 --eval_chunks 8 --tag 3b_train60 --save_lora checkpoints/p1_qwen3b_lora_r16_t60.pt
```

Adapter'ı yükleyip birleştirme (inference'ta sıfır ek maliyet):

```python
import torch
from transformers import AutoModelForCausalLM
from lcsa.lora import apply_lora, set_lora_state_dict, merge_lora

m = AutoModelForCausalLM.from_pretrained("models/Qwen2.5-3B", dtype=torch.float16)
apply_lora(m, r_attn=16, r_mlp=0)
ckpt = torch.load("checkpoints/p1_qwen3b_lora_r16_t60.pt", weights_only=False)
set_lora_state_dict(m, ckpt["lora"])
merge_lora(m)   # base ağırlıklara katla -> normal dense model gibi kullan
```

---

## 6. Sonraki adım fikirleri (opsiyonel)

- Transfer optimizasyonu: pinned-memory ağırlıklar + `non_blocking` kopyalar;
  2-3 blokluk sabit pencere (ping-pong) ile transfer/compute örtüşmesi.
- CPU'da 4/8-bit taban ağırlık (RAM yarıya iner; 7B'ye kapı açılır).
- Optimizer/kimin hedef: yalnız son N katmanı GPU'da tutmak gibi melez rejimler.
- Aynı altyapıyla "streaming generate" (eğitimdeki forward'ı decode'a uyarlama).
