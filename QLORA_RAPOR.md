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

## 6) Sıradaki

- 7B için ön-kantize model (NF4/AWQ) veya RAM boşaltma planı.
- Gerçek talimat verisiyle (SFT) kısa bir kalite kontrolü; M5/LoRA ile aynı sette kıyas.
- LoRA adapter'ını birleştirip (`merge_and_unload`) paged inference ile servis etme denemesi.
