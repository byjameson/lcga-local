# M1_RAPOR — Paged inference: 3B model 6GB kartta çalıştı (ve neden henüz yeterli değil)

**Tarih:** 19.10.2026 · **Kod:** `lcsa/paged_qwen.py`, `experiments/exp16_paged_infer.py`
**Hedef:** VRAM'e sığmayan bir modeli (Qwen2.5‑3B, bf16 **6,17GB**) 6GB kartta çalıştırmak;
katmanları CPU RAM'den GPU'ya akıtmak; matematiksel doğruluğu kanıtlamak.

## 1. Doğruluk (parite)

Qwen2.5‑0.5B, float32, GPU vs HuggingFace (CPU):

| Metrik | Sonuç |
|---|---|
| max abs logit farkı | **3,26e‑5** |
| mean abs fark | 4,16e‑6 |
| argmax eşleşmesi | **1,000** |
| Örnek | "The capital of France is Paris. It is the largest city in…" |

→ Implementasyon Qwen2 matematiğiyle birebir (RMSNorm fp32, GQA, RoPE θ=1e6, SwiGLU, tied embedding).

## 2. 3B / 6GB sonucu

| Ayar | Değer |
|---|---|
| Model | Qwen2.5‑3B, bf16, 36 katman × 147MB |
| Resident / streamed | **26 / 10** katman (embed 594MB GPU'da) |
| Peak VRAM | 4659MB alloc / 4778MB reserved |
| Örnek | "The capital of France is Paris. What is the capital of Germany? … Berlin." |

### Resident oranı — hız eğrisi (pinned)

| resident_k | streamed | GB/token | tok/s |
|---|---|---|---|
| 26 | 10 | 1,44 | **2,75** |
| 20 | 16 | 2,30 | 1,98 |
| 14 | 22 | 3,16 | 1,57 |

**Saf kopya:** 7,18GB / 1,30s = **5,51 GB/s** (pinned host→GPU, PCIe Gen4 x16 max;
idle'da Gen1 x4 görünüyor).

### Kritik dürüst kıyas

| Yöntem | tok/s | Not |
|---|---|---|
| Paged GPU bf16 (k=26) | **2,75** | 6GB VRAM, 6,17GB model |
| **CPU-only bf16** | **3,01** | 32GB RAM, aynı model, aynı kod |
| **llama.cpp Q4 CPU** | **8,54** | b11734, 6 thread, Q4_K_M; pp64 82,0 t/s |
| **llama.cpp Q4 CPU (8t)** | **12,4** | aynı build, 8 thread (r=3); pp64 108,4 t/s |
| **llama.cpp Q4 CUDA (ngl99)** | **57,6** | tam GPU offload (1,79GB Q4 6GB'a sığar); pp64 1690,7 t/s |

→ **Paging tek başına bf16'da CPU'dan hızlı değil.** Darboğaz: token başına 1,44GB PCIe
transferi (5,5GB/s → ~260ms) + çekirdek gecikmeleri. 3B gibi küçük bir modelde CPU zaten
yeterince hızlı; paging'in değeri **daha büyük modellerde, uzun prompt/prefill'de ve
4‑bit ağırlıklarda** ortaya çıkar.
**Ölçüm sonrası netlik:** 3B **Q4 (1,79GB) zaten 6GB'a sığıyor** (llama.cpp CUDA 57,6 tok/s) →
paging yalnızca **Q4'te de sığmayan** modellerde (≈14B+ Q4) veya **bf16/int8 kalite kayıpsızlığı**
gerektiğinde anlamlıdır; bizim Q4-paged yolumuz (M6) henüz llama.cpp seviyesinde değil.

## 3. Neden hâlâ önemli?

- **6,17GB model 6GB karta sığdı** ve doğru metin üretti; bu, 70B sınıfı için temel primitif.
- Mimari **quantization'a hazır**: streamed katmanlar int8/int4'e çevrilince transfer 2–4× azalır.
- **Prefill/uzun bağlam** compute-ağır olduğu için GPU kazancı orada büyür (transfer sabit).
- MoE/adapter paging (M2/M3) aynı motoru kullanacak.

## 4. Sıradaki (bu raporun ardından)

1. **M6 — int8 streamed weights** (2× az transfer) → hedef ~5–6 tok/s; sonra int4 → ~10 tok/s.
2. **Prefetch/double-buffer** (kopya ile hesaplamayı örtüşme) — T=1'de compute küçük olduğu için
   kazanç sınırlı; uzun prompt/batch'te anlamlı.
3. **llama.cpp / AirLLM baseline** ile aynı modelde ölçüm (Q4 dahil) — ürün iddiası için şart.
4. **M2 — MoE expert paging**: aktif uzman oranı 1/4 → transfer ve hesap azalır.
5. **Daha büyük model** (7B) ile demo: CPU tek başına yavaşlarken paged GPU'nun avantajı.

## 5. Kayıtlar

- `results/results.jsonl` → `m1_parity`, `m1_3b_paged`, `m1_3b_k20`, `m1_3b_k14`,
  `m1_3b_cpu`, `m1_3b_copybench`
- `results/exp16_paged.log`, `results/exp16_parity.json`
- `ROADMAP.md` → M1/M6

## 6. Quantization (M6 erken sonuç — 19.10 18:45)

| Varyant | Transfer/token | tok/s | Örnek | Not |
|---|---|---|---|---|
| bf16 | 1,44GB | 2,75 | doğru | baseline |
| **int8** (per-row) | **0,72GB** | **4,18** | **birebir aynı** | max_rel_err %0,39; temiz kazanç |
| int4 (g=64) | 0,38GB | 3,50 | **bozuldu** | max_rel_err %7,1; strided unpack yavaş |

**Yorum:** int8 anlamlı (1,52×) ve kalite kaybı ölçümde yok; **int4 şu an hem yavaş hem riskli**.
Neden: int4 unpack `dst[...,0::2]/[...,1::2]` strided yazımları GPU'da pahalı; ayrıca per-group
absmax/7 quantizasyonu 3B'de örneği bozuyor. Düzeltme yolları: vektörel unpack (contiguous),
grup 32, veya int8 matmul (`_int_mm`) ile dequant'ı tamamen kaldırmak.

**Güncel dürüst sıralama (aynı makine, 3B):** llama.cpp Q4 CPU (beklenen ~10–20 tok/s) >
**paged int8 4,18** > CPU bf16 3,01 > paged bf16 2,75 > paged int4 3,50 (kalite riski).
→ Paged inference'ın ürün değeri **eğitimde** ve **uzun-context/prefill**de; salt 3B decode'da
CPU Q4 hâlâ önde. Bir sonraki büyük adım: **M5 tam-parametre >VRAM eğitim**.

