# P4 RAPOR — Döngülü Derinlik (LoopLM): "Uzun Düşünme" Mini Deneyleri

> Amaç: aynı parametreyi birden çok kez kullanmak (ağırlık-paylaşımlı döngü) hem **parametre
> verimliliği** hem de **test-zamanı derinliği** ("daha çok döngü = daha çok düşünme")
> sağlar mı? Kod: `lcsa/mini_lm.py` (`loops`), `experiments/exp11_own_lm.py`
> (`--loops`, `--loops_rand`, `--loop_sweep`); kanıt: `results/p4_*.log` + `results/results.jsonl`.
> Ayar: ctrl8k tokenizer, 6000 adım × 32×512, aynı eval pencereleri (n_eval=128). Tarih: 18.10.2026.

## P4.1 — Sabit döngü (L=2) vs düz katmanlar

| Model | Param | Adım-başı hesap | bpb | Süre |
|---|---|---|---|---|
| flat8 (a2_ctrl8k, referans) | 17.3M | 8 blok-geçişi | **0.5612** | — |
| **loop2×4** | 10.2M | 8 blok-geçişi | **0.5761** | 48.2 dk |
| flat4 (kontrol) | 10.2M | 4 blok-geçişi | 0.5889 | 28.7 dk |

- **Eşit paramda döngü kazandı:** loop2×4 − flat4 = **−0.0128 bpb (%2.2 iyi)**.
- **Eşit adım-başı hesapta:** loop2×4, flat8'in **%2.7 gerisinde** ama **%41 az parametreyle**.
- → Döngü bu ölçekte bir **parametre verimliliği** aracı.

**Çıkarım sweep'i (L=1/2/4/8):** Eğitim L=2 sabit → tüm eval noktalarında en iyi L=2.
@6000: L=1 **0.7789** (+%35), L=2 0.5761, L=4 0.6797 (+%18), L=8 1.0116 (+%76).
→ **Sabit-döngü eğitimi modeli derinliğe kilitliyor; fazladan çıkarım döngüsü BOZUYOR.**

## P4.2 — Rastgeleleştirilmiş döngü: L ~ U{1..4} (`p4_looprand14`)

Aynı bütçe; eğitimde her adımda L rastgele (hist: 1494/1516/1479/1511 ≈ düzgün).
Not: L=4 adımları 6 GB VRAM'i aştı (vram 7.08G → sistem RAM'ine taşma); koşu 153 dk
sürdü — yavaş ama sağlıklı. (CPU smoke + GPU smoke ✓.)

| Çıkarım L | looprand14 bpb | loop2x4 (sabit L=2) bpb |
|---|---|---|
| L=1 | **0.6029** | 0.7789 |
| L=2 | **0.5840** | 0.5761 |
| L=4 | **0.5852** | 0.6797 |
| L=8 (eğitim aralığı dışı) | **0.6066** | 1.0116 |

- **Derinlik kilidi AÇILDI:** L=1..4 neredeyse düz (L=2↔L=4 farkı %0.2); L=8 bile +%3.9'da
  kalıyor → **"anytime" derinlik yeteneği** kazanıldı (değişken hesapla çalışabilme).
- **Ama kalite kazancı yok:** fazla döngü kaliteyi artırmıyor (L=4 ≈ L=2, L=8 kötü).
  "Daha çok düşünme = daha iyi" bu ölçekte **DOĞRULANMADI**.
- Sabit-L=2'ye karşı **kalite vergisi ~+%1.4** (0.5840 vs 0.5761); karşılığında derinlik
  esnekliği. L=1 hızlı-mod (0.6029) ise aynı hesaptaki **flat4'ün gerisinde** (0.5889) →
  döngünün "hızlı çıkarım" vaadi bu ölçekte zayıf.

## Verdict (dürüst)

1. **Parametre verimliliği: EVET.** Döngü, eşit paramda düzden iyi; eşit hesapta %41 az
   parametreyle flat8'in %2.7 içinde.
2. **Test-zamanı "uzun düşünme": HAYIR (bu ölçekte, 2 deney).** Ne sabit ne rastgele eğitim
   ekstra döngüden kalite kazandı; rastgele eğitim sadece hasarı sıfırladı.
3. **AnyTime çıkarım: EVET (rastgele eğitimle).** L=1..4 arası ~%3 bantta — esnek bütçe için
   kullanışlı; ama "bedava zeka" değil.
4. Literatürle tutarlı: rastgele-derinlik eğitimi "anytime" davranış verir; akıl yürütme
   kazancı genelde ek ölçek/veri/eğitim şartına bağlı çıkar.

## Sınırlar / sonraki fikirler

- (a) L=4 VRAM taşması koşuyu yavaşlattı (gradyan-biriktirme ile düzeltilebilir).
- (b) 6000 adım sınırlı; L=4 vs L=2 farkı kapanma trendinde (@500 +0.0037 → @6000 +0.0012) —
  çok daha uzun eğitimde "L=4 küçük kazanç" ihtimali teorik olarak açık (test edilmedi).
- (c) Sonraki adaylar: **uyarlanabilir derinlik** (kolay token L=1 / zor token L=4),
  **MoE** (çıkarım hızı), P3 taban modelinin güçlendirilmesi (yürüyor).

## Dosyalar
- ckpt: `p4_loop2x4_best.pt` · `p4_flat4_best.pt` · `p4_looprand14_best.pt` (+ `_ckpt.pt` resume noktaları)
- log: `results\p4_loop2x4.log` · `p4_flat4.log` · `p4_looprand14.log`; JSONL: `p4_*` satırları
- Plan: `P4_PLAN.md` · İlgili: `P6_RAPOR.md` (senkron), `P5_RAPOR.md` (tokenizer)
