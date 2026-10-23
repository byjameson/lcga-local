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
seq 256, streaming QLoRA:

| Koşu | Train/eval | LoRA | Peak | Adım | Eval loss (önce → sonra) |
|---|---|---|---|---|---|
| 100 adım, attn | 512/32 | 5,05M | 5,78 GB | 5,99 s | 2,6850 → 1,8710 (−0,814) |
| **200 adım, all (attn+mlp)** | **768/32** | **20,19M** | **5,66 GB** | **5,25 s** | **2,4782 → 1,5670 (−0,911)** |
| **300 adım, all (attn+mlp)** | **1024/32** | **20,19M** | **5,52 GB** | **5,55 s** | **2,6185 → 1,7634 (−0,855)** |

- 200 adımlık koşu **1.052 s (~17,5 dk)**, 300 adımlık koşu **1.684 s (~28 dk)**;
  host RSS ~5,4 GB.
- Üç koşunun eval seti farklı (farklı train_size), bu yüzden mutlak seviyeler
  değil her koşunun kendi düşüşü kıyaslanmalı; hepsi hâlâ tek seed PoC.
- Yani 7B, 6 GB VRAM'in altında **talimat verisiyle** eğitilebiliyor; mlp LoRA
  hedefleriyle 200–300 adımda eval loss **−0,86…−0,91**.

## 10) 14B — CPU-embedding düzeltmesiyle 6 GB'a sığıyor

Qwen2.5-14B-Instruct (Apache-2.0, ~28 GB bf16) üzerinde iki deneme yapıldı.
**13B diye bir Qwen2.5 sürümü yok; en yakını 14B.**

### 10a. İlk deneme (embedding GPU'da) — taşıyor

| Metrik | Değer |
|---|---|
| Peak VRAM | **7,01 GB** → 6 GB'ı aşıyor, WDDM spill |
| Adım süresi | **70,2 s** (seq 128, 1 adım) |
| Host RSS | 9,31 GB |
| Load | 112,7 s |

Sebep: 14B'de `embed_tokens` + `lm_head` bf16 olarak GPU'da ~3 GB tutuyordu;
48 katman + aktivasyonlarla birlikte 6 GB aşılıyordu.

### 10b. CPU-embedding düzeltmesi — sığıyor

`StreamingQuantLoRATrainer` artık `embed_tokens`'ı **CPU'da** tutar; yalnızca
toplanan aktivasyon (seq 128 × 5120 bf16 ≈ 1,3 MB) GPU'ya gider. Bu, GPU'dan
~1,5 GB'ı kaldırır:

| Metrik | Değer |
|---|---|
| Model | Qwen2.5-14B-Instruct, NF4 4-bit CPU'da |
| Katman | 48 |
| Eğitilebilir LoRA (r=8, q/k/v/o) | 12,58M (%0,154) |
| **Peak VRAM** | **3,90 GB** ✅ (6 GB'ın altında) |
| **Adım süresi** | **8,5 s** (seq 128; eski 70,2 s) |
| Host RSS | ~11,6 GB |
| Load | ~94 s |

- Yani parça parça (blok blok) eğitimi doğru kurgulayınca **14B de 6 GB'da
  eğitilebiliyor**: bloklar CPU'dan akar, embedding CPU'da kalır, sadece LoRA
  gradyanları birikir.
- 3 adımlık kısa koşuda WikiText eval CE 2,0143 → 2,0145 (fiilî düz); bu bir
  **sığdırma/yetenek** sonucu, yakınsama iddiası değil.

### 10c. 20 adım — öğrenme başlıyor

Aynı hat 20 adım çalıştırıldı (seq 128, eval 2 chunk):

| Metrik | Değer |
|---|---|
| **Eval CE** | **2,3270 → 2,2215 (−0,1055)** ✅ |
| Peak VRAM | **3,90 GB** (sabit; adım başına değişmiyor) |
| Adım süresi (medyan) | **8,63 s** |
| Host RSS | 11,63 GB (kararlı) |
| Load | 91,5 s |

- 20 adımda eval CE **−0,105** düştü: 14B gerçekten öğreniyor, sadece "sığıyor"
  değil. Eğitim loss'u 2,92 → 1,63 aralığında dalgalı (küçük r, kısa koşu).
- Ayrıntı: `results/results.jsonl` (`qlora14b_cpuembed_t20`), checkpoint
  `checkpoints/stream_qlora_14b_cpuembed_t20`.
- Dürüst sınır: host RSS ~11,6 GB; bu makinede boş RAM sınırında. Ayrıca `lm_head`
  hâlâ GPU'da; onu da CPU'ya/chunk'lı hesaba almak VRAM'i daha da düşürebilir.

### 10d. 14B talimat SFT (Dolly-15k) — çalışıyor

Aynı streaming QLoRA hattı (`exp53`, `--targets attn`) 14B üzerinde 200 adım
talimat eğitimi (seq 128, 512 train / 32 eval satır):

| Metrik | Değer |
|---|---|
| **Eval loss** | **2,7531 → 1,4771 (−1,2760)** ✅ |
| Peak VRAM | **4,14 GB** (kararlı) |
| Adım süresi (medyan) | **8,77 s** |
| Duvar süresi | 1.778 s (~30 dk) |
| Host RSS | 10,19 GB |
| Eğitilebilir LoRA | 12,58M (%0,154) |

- Eğitim loss'u 8,88 → 0,49 aralığında düştü; eval loss **−1,28** ile 14B'nin
  Dolly talimat verisiyle gerçekten öğrendiğini gösteriyor.
- Yani **6 GB VRAM'li bir laptop GPU'sunda 14B talimat eğitimi** yapılabiliyor.
- Checkpoint: `checkpoints/stream_qlora_14b_sft` (adapter, git dışı).

## 11) Sıradaki

- 14B için SFT (Dolly-15k) + eval; bellek daha da düşsün diye `lm_head`'i de
  CPU/chunk'lı hesaba almak.
- 7B streaming QLoRA'da daha uzun SFT + geniş LoRA hedefleri (mlp).
- Merge edilmiş modellerin `serve.py`/`llm.py` ürün yoluna bağlanması.

## 12) Yeniden üretme

```bash
pip install -r requirements-qlora.txt          # bitsandbytes 0.50.2, peft 0.21.2

# 3B: HF in-memory QLoRA PoC
python experiments/exp49_qlora_3b.py --steps 10 --seq 512 --eval_chunks 2 --tag qlora3b_seq512_t10

# 7B: kendi blok-streaming QLoRA'mız (base 4-bit CPU'da, tek blok GPU'da)
python experiments/exp52_stream_qlora_7b.py --steps 10 --seq 128 --eval_chunks 1 --tag qlora7b_stream

# 7B: Dolly-15k talimat SFT
python experiments/exp53_stream_qlora_sft_7b.py --steps 100 --seq 256 --tag stream_sft7b

# 14B: CPU-embedding düzeltmesiyle 6GB'da (taban + SFT)
python experiments/exp52_stream_qlora_7b.py --model models/Qwen2.5-14B-Instruct \
    --steps 20 --seq 128 --eval_chunks 2 --tag qlora14b_cpuembed_t20
python experiments/exp53_stream_qlora_sft_7b.py --model models/Qwen2.5-14B-Instruct \
    --steps 200 --seq 128 --targets attn --tag sft14b_attn_200
```

Not: `exp52`/`exp53` dosya adlarında "7b" geçse de `--model` ile herhangi bir
Qwen2.5 tabanı verilebilir; hat model boyutundan bağımsızdır.

## 13) Disk'ten akış (mmap) — RAM sınırını kaldırmak

`lcsa/disk_quant_store.py` + `experiments/exp54_disk_qlora.py`: 4-bit taban artık
host RAM'de tutulmuyor. bnb-4bit safetensors dosyaları **mmap** ile okunuyor —
`safetensors` CPU'da sıfır kopya görünüm verir, sayfalar OS tarafından gerektikçe
SSD'den gelir ve baskı altında boşaltılır. Her decoder bloğu istendiğinde
doğrudan GPU'ya kurulur (`Params4bit.from_prequantized` ile NF4 quant state
yeniden takılır), işi bitince `meta`'ya döner. Böylece model boyutu sınırı
"RAM'e sığan 4-bit" değil, **disk + adım süresi** olur.

### Doğrulama — RAM yolu ile birebir parite (3B)

| | RAM (exp52) | Disk (exp54) |
|---|---|---|
| eval CE (önce) | 2,73626 | **2,73626** |
| 1. adım loss | 3,347981 | **3,347981** |
| Eğitilebilir LoRA | 3.686.400 | 3.686.400 |
| Peak VRAM | 1,60 GB | 1,98 GB |
| Host RSS | 2,96 GB | 3,57 GB |

Özdeş sonuç → mmap + quant_state loader **doğru**.

### Yol boyunca çıkan ve düzeltilen gerçek hatalar
1. `base_model_of` peft'siz modellerde iç `Qwen2Model`'i döndürüyordu.
2. CPU-embedding optimizasyonu **tied** modelleri kırıyordu (3B/0.5B) → tied'de
   embedding ve head GPU'da kalır.
3. PyTorch `meta` parametreye `.data =` atamayı reddeder → `Parameter` nesnesi
   değiştirilir.
4. transformers 5.x'te `config.rope_theta` taşındı → `rotary_emb` kendi
   constructor'ıyla yeniden kurulur.
5. İskelette dondurma + fp32 cast eksikti (peft `prepare_model_for_kbit_training`
   yapar) → parite 0,0008 sapıyordu; hizalandı.

`--shared-dtype as-stored` ile embedding/head fp32'ye şişirilmeden checkpoint
dtype'ında (bf16) kalabilir — büyük modeller için şart.

## 13b) Autotune — her GPU'ya uyum (`lcsa/autotune.py`)

Açık kaynakta herkesin VRAM'i farklı olacağı için motor sabit bir varsayım
yapmıyor; plan **checkpoint metadata'sından** (ağırlık okumadan, ms cinsinden)
ve makinenin boş VRAM/RAM'inden çıkarılıyor:

```bash
python -m lcsa.autotune --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit --seq 128
```

Bu makinede (RTX 3050 6 GB, 32B checkpoint) çıktı:

```
strategy=disk-stream  gpu=NVIDIA GeForce RTX 3050 6GB Laptop GPU
VRAM total=6.4GB free=5.4GB budget=4.2GB
model=19.2GB  block=258MB  layers=64
blocks_on_gpu=6 (1.51GB)  head_on_gpu=True (1.56GB)  loss_chunk=64
note: 4-bit base (19.2GB) does not fit comfortably in RAM: stream from disk
```

Karar mantığı:
- Blok boyutu **gerçek** (4-bit saklanan hâli), tahmin değil.
- Bütçe = boş VRAM × 0,78; ayrıca **0,6 GB rezerv** (CUDA allocator/fragmentasyon;
  WDDM taşması yavaşlamadan çok daha kötü).
- 4-bit head kullanılırsa **dequant workspace**'i (`hidden × vocab × 2`) hesaba katılır.
- Strateji: tüm model sığıyorsa `gpu-full`; 4-bit taban RAM'e sığıyorsa
  `ram-stream` (exp52 yolu); sığmıyorsa `disk-stream` (exp54).
- `exp54 --blocks-on-gpu auto` (varsayılan) bu planı uygular: **büyük kartta tüm
  bloklar resident kalır ve hiç streaming olmaz**; küçük kartta pencere daralır.

### 13c) Ölçüme dayalı autotune (`--blocks-on-gpu adaptive`, varsayılan)

Statik tahmin yetmiyor (allocator/fragmentasyon karttan karta değişir), bu yüzden
varsayılan **ölçüm** ve ölçüm **periyodik** (`--adapt-every N`, varsayılan 10):

1. Her zaman **1 blokla** başlar (en güvenli).
2. Her N adımda o adımın `max_memory_reserved` değerini okur.
3. Yumuşak hedefe (`toplam VRAM × 0,80`) kadar büyütür — ama **nominal blok
   boyutuyla değil, ölçülen marjinal maliyetle**: son *farklı* pencerelerdeki iki
   ölçümden `Δpeak / Δblok` hesaplanır. Böylece hedefe yaklaşırken adımlar küçülür.
4. Sert sınırı (`× 0,88`) aşarsa küçültür ve `empty_cache()` ile allocator'ı boşaltır.

32B üzerinde ölçülen yörünge (`--adapt-every 5`, 12 adım):

| Adım | Pencere | Peak | Karar |
|---|---:|---:|---|
| 1 | 1 | 3,18 GB | marjinal nominal (258 MB) → 1→**8** |
| 5 | 8 | 4,64 GB | **ölçülen marjinal 213 MB/blok** → 8→**10** |
| 10+ | 10 | 5,10 GB | pay kalmadı → **sabit** |

Sonuç: **10 blok**, peak **5,10 GB** (yumuşak 5,15 / sert 5,63 GB), adım
**11,5 → 9,3 s**. Başka bir koşuda (26 adım, aynı ayarlar) 1→8→9'da durdu
(peak 4,92 GB): yani sonuç kartın o anki durumuna göre değişiyor ama **salınım
yok ve hedef aşılmıyor**.

24 GB'lık bir kartta aynı döngü tüm 64 bloğu resident yapar ve streaming biter.

Hedefler elle de verilebilir: `--vram-soft-gb` / `--vram-hard-gb` (ör. VRAM'in bir
kısmını ekrana/başka uygulamaya bırakmak için).

**Bulunan ve düzeltilen kusur (salınım).** Büyütme yalnızca yumuşak hedefe bakıyordu;
`sert < yumuşak` verildiğinde döngü salınıyordu — ölçüldü: `1→8→5→9→6→5→…`
(her büyütme sert sınırı aşıyor, her aşım küçültüyor). Düzeltme: **büyütme de sert
sınırla kapatılıyor** (tahmini peak ≤ hard). Aynı ayarla (`--vram-hard-gb 4.2`)
düzeltmeden sonra yörünge tek yönlü: **1→5→6**, peak **4,19 GB** (sınır 4,20),
adım 9,6–9,8 s, sabit. Küçültme kodu yine güvenlik ağı olarak duruyor ve ilk koşuda
gerçekten çalıştı (4,63 GB → 5 blok).

### 13d) Statik plan (`--blocks-on-gpu auto`)

Tahminî plan (yukarıdaki `plan_for`) hâlâ mevcut ve hızlı: bu makinede 32B için
6 blok / peak 4,14 GB / 10,5–11,2 s ölçüldü. Fark: statik plan VRAM'i tahmin
eder, `adaptive` **ölçer**. Not: aynı koşuda 1. adım loss'u birebir aynı
(2,903609) çıkıyor, sonraki adımlar LoRA'nın rastgele başlangıcı yüzünden farklı.

## 14) 32B — disk'ten akışla 6 GB VRAM'de (ana sonuç)

Qwen2.5-32B-Instruct (unsloth **bnb-4bit**, 64 katman, 17,9 GB safetensors;
dört shard'ın SHA256'sı doğrulandı) hiç RAM'e yüklenmeden, mmap'ten akıtılarak
eğitildi.

| Metrik | Değer |
|---|---|
| **Peak VRAM** | **3,11 GB** (6 GB'ın altında) |
| Adım süresi (medyan) | **10,95 s** (seq 128) |
| Eval CE | **2,3942 → 2,3637 (−0,0305)** / 10 adım |
| Eğitilebilir LoRA (r=8) | 16,78M |
| İskelet yükleme | 9,7 s (**RSS 2,44 GB** — 17,9 GB'ın tamamı yüklenmiyor) |
| Host RSS (adım sırasında) | 19,4 GB — ağırlıkların **dosya sayfası önbelleği** |

- `--shared-dtype as-stored`: embedding bf16 olarak **CPU'da mmap view**, lm_head
  bf16 **GPU'da** (1,56 GB); norm/bias da bf16.
- İlk adım soğuk cache ile **24,8 s**; 10 adımda cache ısınınca **11 s**.
  Adım başına disk trafiği ≈ 2 × 16 GB (forward + geri-recompute).
- İlginç: 32B'nin peak'i **14B'den düşük** (3,11 vs 3,90 GB). Sebep: 14B'de
  `lm_head` 4-bit'ti ve bnb'nin 4-bit matmul'ü dequantize workspace'i ayırıyordu;
  32B'nin head'i doğal bf16 olduğu için o workspace yok.
- **RAM hakkında dürüst not:** ağırlıklar başlangıçta RAM'e alınmıyor (RSS 2,44 GB)
  ama adım sırasında OS okunan dosya sayfalarını önbellekte tuttuğu için RSS
  19,4 GB'a çıkıyor. Windows'un commit sayacı da ~20 GB gösterdi; bu sayfaların
  ne kadarının anında geri alınabilir olduğunu `EmptyWorkingSet` ile bu makinede
  kanıtlayamadım (sayaç setleri kapalı). Yani: **16 GB RAM'li bir makinede de
  çalışır ama sayfa değişimi artar ve adım süresi uzar**; 6 GB VRAM sınırı ise
  aynen korunur. Asıl kanıtlanan şey şu: model hiçbir anda bütün olarak RAM'e
  yüklenmiyor, bloklar tek tek materialize ediliyor.

### 14b. 32B talimat SFT (Dolly-15k) — asıl sonuç

Aynı disk-akışlı hat, `--sft` modunda 100 adım (seq 128, 256 train / 8 eval):

| Metrik | Değer |
|---|---|
| **Eval loss** | **3,2600 → 1,7849 (−1,4751)** |
| **Peak VRAM** | **3,21 GB** |
| Adım süresi (medyan) | **11,0 s** |
| Duvar süresi | 1.194 s (~20 dk) |
| Eğitilebilir LoRA | 16,78M |

- Yani **32B, 6 GB VRAM'li bir laptop GPU'sunda talimat eğitimi görüyor** ve
  20 dakikada eval loss'u **−1,48** düşürüyor. 14B'nin 200 adımlık SFT'si −1,28
  idi; 32B daha az VRAM'de (3,21 vs 4,14 GB) daha büyük düşüş verdi.
- Dürüst kıyas notu: eval setleri farklı (32B: 256 train/8 eval, 14B: 512/32),
  bu yüzden mutlak seviyeler değil **kendi içindeki düşüş** kıyaslanmalı.
- Adapter: `checkpoints/disk_qlora_32b_sft` (git dışı).

