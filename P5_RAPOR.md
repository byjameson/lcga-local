# P5 RAPOR — "Tokenizer 2" Arenası (Aşama 1 + 2 + 2b; karar: klasik kazandı)

**Tarih:** 16–17.10.2026 · **Ortam:** RTX 3050 Laptop 6GB, Windows 11 · **Metrik:** tokenizer-bağımsız **bpb** (bit/byte; val_loss nats × n_tok / (ln2 × n_bytes))

## 1) Sorular
1. Klasik (GPT-2 tarzı regex ön-bölmeli) BPE mi, **süper-kelime** (regex'siz; mergeler boşluk/noktalama sınırını aşabilir) BPE mi daha iyi sıkıştırır? → Aşama 1
2. Sıkıştırma üstünlüğü **dil modeli kalitesine** dönüşüyor mu? → Aşama 2 (eşit-adım MiniLM)
3. Vocab büyüklüğü (4k/8k/16k) etkisi?

Veri: TinyStories train ilk 1M hikâye (+ valid[:3000]); tüm kollar **aynı metin**; valid_bytes = 2.493.514.

## 2) Aşama 1 — sıkıştırma & hız (CPU)
Kaynak: `results/tokenizer_arena_v1.json` + `.log` (held-out valid 3000 hikâye).

| aday | byte/token | encode MB/s | tokenizer eğitimi |
|---|---|---|---|
| kontrol (`ts_bpe8k`) | 4.158 | 14.7 | — |
| `wb_4k` | 3.988 | 16.3 | 30s |
| `wb_8k` | 4.158 | 16.5 | 29s |
| `wb_16k` | 4.193 | 15.9 | 29s |
| `sw_4k` | 5.258 | 6.3 | 798s |
| `sw_8k` | 6.334 | 5.3 | 1159s |
| `sw_16k` | 7.474 | 4.8 | 1457s |

Bulgular: süper-kelime aynı vocab'da **%32–78 daha iyi sıkıştırıyor** (8k: +%52, 16k: +%78) → aynı 512-token bağlam 1.5–1.8x metin; aynı metin için ~1.8x daha az forward. Bedeli: tokenizer eğitimi ~40–50x, encode ~3x yavaş (satır-içi kullanımda önemsiz). Vocab verimi: wb doygun (4k→16k yalnız +%5), sw canlı (+%42). Doğrulama: `ts_bpe8k` ≡ `bpe_wb_8k` (md5 birebir).

## 3) Aşama 2 (1x) — eşit-adım MiniLM (GPU)
Protokol: 4 kol; her kol **6000 adım × batch 32 × ctx 512 = 98.3M token**; aynı ayar (lr 1e-3 cosine, warmup 300, seed 1337); MiniLM tied-emb (~17.3M param 8k / ~20.5M param 16k); kanıt: `results/results.jsonl` (`a2_*` satırları), `results/a2_*.log`.

| kol | **bpb** | val | ppl | duvar | eğitimde gördüğü metin |
|---|---|---|---|---|---|
| `a2_ctrl8k` (klasik 8k) | **0.5612** | 1.6093 | 5.00 | 48.0 dk | ~400 MB |
| `a2_sw16k` (süper-kelime 16k) | 0.5769 | 2.9621 | 19.34 | 56.9 dk | ~722 MB |
| `a2_wb16k` (klasik 16k) | 0.5776 | 1.6706 | 5.32 | 47.8 dk | ~404 MB |
| `a2_sw8k` (süper-kelime 8k) | 0.5890 | 2.5663 | 13.02 | 48.5 dk | ~613 MB |

Sonuçlar:
1. **Klasik 8k kazandı.** Süper-kelimeler hem daha iyi sıkıştırıyor hem adım başına %52–78 daha fazla metin görüyor — buna rağmen bpb'de geride kaldı (en iyi sw16k +%2.8).
2. **sw16k yakınsamadan bitti:** son 2500 adımda ~−0.009 bpb/500 (diğerleri ~−0.004/500) → "daha uzun bütçe" sorusu açık; **aşama 2b (2x) bu yüzden koşuyor.**
3. **Vocab etkisi zıt yönlü:** klasikte 8k→16k +%2.9 kötüleşti; süper-kelimede 8k→16k −%2.1 iyileşti.
4. Örnek metinler tabloyla tutarlı (ctrl8k en akıcı; örnekler log'larda `[sample ...]` satırlarında); sw16k'nın 160-token örneği ~1.2KB metin üretiyor (ctrl ~0.66KB) — token verimliliği somut, kaliteye (bpb) henüz yansımadı.

## 4) Aşama 2b (2x) — ana kollar bitti
Sıfırdan **12.000 adım** (197M token/kol); aynı ayar; tag `a3_*`.

| kol | bpb | val | duvar |
|---|---|---|---|
| `a3_ctrl8k` (klasik 8k) | **0.5264** | 1.5095 | 96.0 dk |
| `a3_sw16k` (süper-kelime 16k) | 0.5422 | 2.7838 | 113.4 dk |
| `a3_wb16k` (kelime-bazlı 16k) | 0.5445 | 1.5747 | 95.0 dk* |
| `a3_sw8k` (süper-kelime 8k) | 0.5529 | 2.4088 | 96.0 dk |

\* wb16k, ckpt step 2000'den resume edilerek tamamlandı (17.10 gecesi kesilmişti).

**Sonuç (18.10, tablo tamamlandı):** 2x bütçede de klasik kazandı; fark **%3.0** (1x'te %2.8 idi — kapanmıyor, stabil). **Tam tablo:** `a3_ctrl8k` **0.5264** < `a3_sw16k` 0.5422 < `a3_wb16k` 0.5445 < `a3_sw8k` 0.5529 — klasik 8k en iyi, süper-kelime 8k en kötü (1x sıralamasıyla tutarlı). İki ana kol da 1x finaline göre ~%6 iyileşti ama sıralama değişmedi; sw16k adım başına 1.8x fazla metin görmesine rağmen (1.44GB vs 0.80GB) bpb'de %3 geride → "süper-kelime daha çok bütçe istiyor" hipotezi bu ölçekte **doğrulanmadı**. Karar: **klasik (ts_bpe8k ailesi) kalıyor**; süper-kelime = hız modu adayı.

## 5) Şimdiki çıkarım
- Eldeki tokenizer (`ts_bpe8k` = klasik wb_8k) bu reçetede **en iyi**; değiştirmek için henüz kanıt yok.
- Süper-kelime şimdilik "aynı metni daha az token'la işle → hızlı inference" adayı (bağlam başına 1.5–1.8x metin) ama küçük bir kalite bedeli riski var (2b netleştirecek).
- Sıradaki adaylar: BLT-proper entropy patching; süper-kelime için epoch/bütçe analizi; vocab 32k.
