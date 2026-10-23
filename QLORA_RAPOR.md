# QLoRA Raporu — 3B model, 6 GB kart, NF4 + LoRA (23.10.2026)

**Etiket:** `exp49_qlora` · **Durum:** ✅ 3B PoC çalışıyor · **Amaç:** tam-parametre
streaming'in yanında pratik bir "adaptör fine-tune" yolu göstermek.

## 1) Ne yaptık

`experiments/exp49_qlora_3b.py`:

- **Taban:** `Qwen/Qwen2.5-3B-Instruct` (`qwen-research` lisansı), **NF4 4-bit** +
  double-quant, compute dtype **bf16**; taban dondurulmuş.
- **Adapter:** LoRA `r=16`, `alpha=32`, `dropout=0.05`, hedef modüller
  `q/k/v/o_proj` + `gate/up/down_proj`; eğitilebilir parametre **29,93M / 1,729B = %1,73**.
- **Bellek teknikleri:** gradient checkpointing, `paged_adamw8bit` optimizer.
- **Veri/ölçüm:** WikiText-2 next-token loss; `exp26`/M5 ile aynı token akışı
  (`data/wt2_train.pt`), 2 chunk `wt2_test` eval.

## 2) Sonuçlar (seq 512, batch 1, lr 2e-4)

| Metrik | Değer |
|---|---|
| Yükleme | 12,6 s |
| Eğitilebilir parametre | 29,93M (%1,73) |
| Adım süresi (medyan) | **2,56 s** |
| Peak VRAM | **4,25 GB** |
| Host RSS | 1,83 GB |
| Eval CE (önce → sonra) | **2,5154 → 2,3188 (−0,1966)**, 10 adım |
| Optimizer | `paged_adamw8bit` |

Kısa smoke'lar: 3 adım/seq 256 → peak 3,55 GB, CE 2,6600 → 2,5810 (−0,079);
5 adım/seq 512 → peak 4,25 GB, CE 2,5154 → 2,3794 (−0,136).
Ham satırlar: `results/results.jsonl` (`exp=exp49_qlora`), log: `results/exp49_qlora.log`.
Adapter çıktıları (yerel, git dışı): `checkpoints/qlora_qwen3b_r16_t10/`.

## 3) M5 ile kıyas (aynı 3B, farklı hedef)

| | M5 tam-parametre streaming | **QLoRA (bu rapor)** |
|---|---|---|
| Güncellenen | 3,09B (%100) | 29,9M (%1,73) |
| 3B SGD adım | 5,2–6,6 s | **2,56 s** (seq 512) |
| Peak VRAM | 2,4–4,9 GB | **4,25 GB** |
| Ne için? | Tabandan eğitim/pretrain denemesi | Adaptör fine-tune |
| 7B'ye ölçek | 3B'de kalıyor (RAM) | 7B için ön-kantize model/ek RAM gerekir |

İkisi rakip değil: M5 "VRAM'den büyük modeli tam-parametre güncelle" sorusunu,
QLoRA "6 GB'da daha büyük bir tabanı adapte et" sorusunu yanıtlıyor.

## 4) Dürüst sınırlar

- **3B PoC.** 7B bf16 ~15,2 GB; bu makinede boşta ~11,4 GB RAM olduğu için
  yüklenemiyor. 7B için ön-kantize (NF4/AWQ/GGUF) model ya da daha fazla RAM gerekir.
- **Kısa koşu, tek tohum.** 10 adım, WikiText-2 next-token CE; talimat kalitesi,
  genel benchmark veya uzun eğri iddiası yok.
- **Lisans:** Qwen2.5-3B `qwen-research`; üretilen LoRA adaptörü de bu lisansın
  şartlarına tabidir. Şu an HF'ye yüklenmedi.
- **Optimizer:** `paged_adamw8bit` Windows'ta çalıştı; farklı bitsandbytes
  sürümlerinde fallback `torch.optim.AdamW`'ye düşer.

## 5) Çalıştırma

```powershell
pip install -r requirements-qlora.txt
python experiments/exp49_qlora_3b.py --steps 10 --seq 512 --eval_chunks 2 \
    --save checkpoints/qlora_qwen3b_r16_t10 --tag qlora3b_seq512_t10
```

## 6) Merge + paged inference (exp50)

`experiments/exp50_qlora_merge.py` adaptörü bf16 tabana merge edip
`checkpoints/qwen3b_qlora_merged` (6,17 GB, 4 shard) olarak kaydetti; ardından
`exp16_paged_infer.py` ile **int8 paged** olarak 6 GB'da servis edildi:

| | değer |
|---|---|
| Merge (CPU, bf16) | yükle 14,1 s + merge 13,8 s + kaydet 19,1 s, host RSS 7,28 GB |
| Paged inference | resident 26/36 katman, 10 streamed int8, peak **4731 MB**, **2,74 tok/s** |
| Taban (aynı ayar) | peak 4731 MB, **3,03 tok/s** |
| Örnek | "What is the capital of France? The capital of France is Paris. …" |

Bu, tüm hattın çalıştığını gösterir: **NF4 QLoRA eğitimi → merge → int8 paged
servis**. Aynı 32 token'lık örnekte taban ve merge'lı model benzer ve tutarlı
cevap verdi; bu küçük bir duman testidir, kalite benchmark'ı değildir.

## 7) Kısa SFT — Dolly-15k (exp51)

`experiments/exp51_qlora_sft_3b.py`: aynı NF4+LoRA hattı, bu kez **talimat
verisiyle** (`databricks/databricks-dolly-15k`, Apache-2.0; tek tohum). Qwen
sohbet şablonu, kullanıcı öneki maskelenir.

| Metrik | 20 adım (smoke) | 200 adım (ana) |
|---|---|---|
| Train / eval satır | 128 / 8 | 512 / 32 |
| Peak VRAM | 3,90 GB | 3,90 GB |
| Adım süresi (medyan) | 1,95 s | **1,74 s** |
| Host RSS | 2,04 GB | 2,04 GB |
| Eval loss (önce → sonra) | 1,9528 → 1,2480 (−0,705) | **2,4405 → 1,9040 (−0,537)** |
| Örnek | Paris; kısa tavsiye | Paris; üç somut dil öğrenme maddesi; tek cümlelik egzersiz özeti |

Not: iki koşunun eval seti farklı (8 vs 32 satır), bu yüzden mutlak seviyeleri
değil her koşunun kendi düşüşünü kıyaslamak gerekir. 200 adımlık koşu
**353 s (~5,9 dk)**, tek seed; hâlâ benchmark değil, ama talimat formatını
öğrendiğini ve 6 GB'da kararlı çalıştığını gösteriyor.

**SFT adapter'ını merge + paged servis:** `exp50` ile `qlora_qwen3b_sft_t200`
bf16 tabana merge edildi (`checkpoints/qwen3b_sft_merged`, 6,17 GB; merge 19,9 s,
host RSS 7,47 GB) ve `exp16` ile int8 paged servis edildi: **2,84 tok/s**, peak
**4733 MB**. Aynı sohbet-şablonlu istemde örnek:

- SFT-merged: "1. Practice speaking the language as much as possible. 2. Listen
  to the language as much as possible. 3. Read the language as much as possible."
- Taban 3B-Instruct: daha uzun bir giriş + madde listesi (64 token sınırında
  kesildi).

Aynı ayar/istem ile tek örnek; kalite benchmark'ı değil, hattın ürün yolunda
çalıştığını gösteren bir duman testi.

## 8) 7B denemesi — 6 GB'a sığmıyor (dürüst sonuç)

Qwen2.5-7B-Instruct indirildi (Apache-2.0, 14,2 GB bf16, 4 shard) ve aynı
NF4+LoRA hattı denendi:

| Ayar | Load | Peak alloc | Adım | Not |
|---|---|---|---|---|
| 7B, seq 256, 2 adım | 37,4 s | **8,69 GB** | 4,3–5,2 s | 6 GB'ı aşıyor → WDDM shared memory'ye taşıyor |
| 7B, seq 128, 2 adım | 41,3 s | **8,42 GB** | 4,7–8,1 s | Kısa bağlamda bile 6 GB'a sığmıyor |
| 7B, `device_map=auto` + CPU offload | — | — | — | `accelerate`/`bitsandbytes` meta-tensor hatası (`Tensor.item() cannot be called on meta tensors`) |

- 7B 4-bit taban + bf16 compute + aktivasyonlar bu kartta **~8,4–8,7 GB** tepe
  yapıyor; yani çalışıyor gibi görünse de gerçekte **6 GB VRAM'e sığmıyor**,
  paylaşımlı belleğe taşıyor. Bu yüzden 7B QLoRA'yı "6 GB'da çalıştı" diye
  raporlamıyoruz.
- Eğitilebilir parametre 40,37M (%0,92); eval CE 2 adımda −0,04…−0,06 oynadı
  (anlamlı bir eğri için çok kısa, üstelik spill).
- CPU offload denemesi bu sürümlerde (transformers 5.19 / bitsandbytes 0.50.2 /
  accelerate 1.15) meta-tensor hatasıyla düştü; düzeltmek için farklı sürüm,
  ön-kantize (AWQ/GPTQ) model veya daha fazla VRAM gerekir.

**Sonuç (HF in-memory yolu):** HF'nin tüm-model 4-bit QLoRA'sı bu kartta 7B'de
taşıyor. Ama kendi blok-streaming motorumuzla 7B **sığıyor** — §9.

## 9) Kendi streaming QLoRA'mız — 7B, 6 GB'da sığıyor (exp52)

`lcsa/stream_quant_lora.py` + `experiments/exp52_stream_qlora_7b.py`: 7B tabanı
**NF4 4-bit CPU'da** tutar, aynı anda **tek decoder bloğunu** GPU'ya alır, LoRA
gradyanlarını CPU'da biriktirir. HF'nin tüm-model QLoRA'sının aksine GPU'da tüm
model + compute geçici tamponları yok; bu yüzden 6 GB'a sığıyor.

| Metrik | Değer |
|---|---|
| Model | Qwen2.5-7B-Instruct, NF4 4-bit CPU'da |
| GPU'da aynı anda | 1 blok + embed/head + aktivasyonlar |
| LoRA | r=8, q/k/v/o, **5,05M (%0,116)** eğitilebilir |
| Peak VRAM | **4,90 GB** |
| Adım süresi | **5,9 s** (medyan, seq 128) |
| Host RSS | ~4,9 GB |
| Eval CE (önce → sonra) | **2,2304 → 2,1907 (−0,0397)**, 10 adım |
| Load | ~115 s (ilk 4-bit CPU quantization) |

Karşılaştırma:

| | HF in-memory QLoRA (exp49) | **Bizim streaming QLoRA (exp52)** |
|---|---|---|
| 7B peak VRAM | 8,4–8,7 GB (spill) | **4,90 GB** |
| 6 GB'a sığar mı? | ✗ | ✓ |
| Adım | 4,3–8,1 s | 5,9 s |
| LoRA params | 40,4M (q/k/v/o+mlp) | 5,05M (q/k/v/o, r=8) |

Bu, projeye eklenen **yeni bir yetenek**: HF QLoRA'nın başaramadığı 7B eğitimini
kendi blok-streaming motorumuz 6 GB'da yapıyor.

**Uzun SFT (exp53):** `experiments/exp53_stream_qlora_sft_7b.py`, Dolly-15k,
512 train / 32 eval, 100 adım, seq 256, aynı streaming QLoRA hattı:

| Metrik | Değer |
|---|---|
| Peak VRAM | **5,78 GB** |
| Adım süresi (medyan) | **5,99 s** |
| Duvar | 602 s (~10 dk) |
| Host RSS | 5,14 GB |
| Eval loss (önce → sonra) | **2,6850 → 1,8710 (−0,8139)** |
| Eğitilebilir LoRA | 5,05M (%0,116) |

Yani 7B, 6 GB VRAM'in altında **talimat verisiyle** eğitilebiliyor. 100 adım
hâlâ kısa bir PoC; daha uzun koşu ve mlp hedefleri bir sonraki adım.

## 10) Sıradaki

- 7B streaming QLoRA'da daha uzun SFT + geniş LoRA hedefleri (mlp).
- 13B için aynı hattı denemek (int4 CPU taban ~7 GB; host RAM sınırı).
- Merge edilmiş modellerin `serve.py`/`llm.py` ürün yoluna bağlanması.

