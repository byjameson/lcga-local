# P4 PLAN (taslak v0) — Az veriyle zekâ · uzun düşünme · hızlı inference

**Durum:** 17.10.2026'da yazıldı (P5 aşama-2b koşarken taslak). Tokenizer'la ilgili satırlar 2b sonucuyla güncellenebilir.
**Kısıt:** RTX 3050 Laptop 6GB — plan bu bütçeye göre kademelendirildi.

## Kullanıcı hedefinin ölçülebilir vekilleri
- **Zekâ/verimlilik:** aynı bütçede bpb; sentetik görev doğruluğu (aritmetik, çok adımlı hikâye soruları); "az veri eğrisi" (N hikâye → doğruluk).
- **Uzun düşünme:** hesap ↔ doğruluk eğrisi (düşünme adımı arttıkça doğruluk artıyor mu?).
- **Hız:** token/s, ms/adım, "metin başına forward sayısı".

## Kaldıraçlar

### A. Veri verimliliği
- **A1 — Tokenizasyon (P5):** bpb/hız tradeoff'u; tokenizer seçimi 2b sonucuyla netleşecek.
- **A2 — Sentetik akıl-yürütme karışımı:** TinyStories'e ek, "adım adım çözüm" formatlı küçük ve temiz setler (kendi üretecimizle: basit aritmetik, çok adımlı hikâye soruları, düzenli/kod-benzeri diziler). Az veri + yüksek sinyal.
- **A3 — Eğitim reçetesi:** yüksek-loss pencere madenciliği; tekrar yerine seçim; kolay→zor curriculum.

### B. Uzun düşünme (test-zamanı hesap / TTC)
- **B1 — LoopLM / döngülü derinlik:** aynı bloğu N kez uygula; parametre sabit, hesap artar. Mini ölçek testi: eşit param bütçesinde {8 katman düz} vs {2×4 döngülü}.
- **B2 — Uyarlanabilir hesap:** durma sinyali ("bitti mi?"); kolay soruda 1 döngü, zorda N. Ortalama maliyeti düşük tutar.
- **B3 — Düşünme verisi:** adım adım çözümlerle eğitim; "düşünme uzunluğu ↔ doğruluk" eğrisi ölçülür.

### C. Hızlı inference
- **C1 — MoE (stretch):** küçük MoE (ör. 4 uzman, top-1); aktif param sabit, toplam kapasite ~2x. 6GB'ta küçük-matmul overhead riski var — ölçmeden iddia yok.
- **C2 — Mimari/quantization:** GQA-benzeri KV küçültme; int8/int4 denemeleri; decode batch>1 (P2 dersleri).
- **C3 — Token hızı:** süper-kelime (P5) veya draft-model spec-decode (deneysel).

## Kademeli deney planı (öneri)
1. **P4.0:** 2b bitince karar: tokenizer (ctrl8k vs sw16k) — bpb + metin/forward tradeoff.
2. **P4.1 — LoopLM mini testi** ✅ (18.10 sonuçları): 10.2M param — loop2x4 **0.5761** vs flat4 **0.5889** (eşit paramda döngü kazandı, ~%2.2); flat8 referansı 0.5612 (17.3M, aynı-adım, aynı-adım-başı-hesap) → döngü **%41 az paramla %2.7 içinde**. Sweep: eğitim L=2 → tüm noktalarda en iyi L=2; @6000: L=1 0.779 / L=4 0.680 / L=8 1.012 → **sabit-döngü eğitimi derinliğe kilitler; fazla çıkarım döngüsü bozar.** → **P4.2 ✅ (18.10):** `p4_looprand14` (L~U{1..4}) bpb **0.5840** (+%1.4 vergi vs sabit-L2); sweep: L1 **0.6029** / L2 **0.5840** / L4 **0.5852** / L8 0.6066 → **kilit açıldı ("anytime" L=1..4 ~%3 bantta) ama ekstra döngü kazancı YOK; "bedava uzun düşünme" bu ölçekte doğrulanmadı.** Rapor: `P4_RAPOR.md`. Sıradaki adaylar: uyarlanabilir derinlik / MoE / taban model (P3 resume yürüyor).
3. **P4.2 — Sentetik akıl yürütme + uyarlanabilir durma:** üreteç + eğitim + hesap↔doğruluk eğrisi.
4. **P4.3 — MoE stretch:** top-1 4 uzman; aktif param sabit; token/s ve bpb.

Her adım: `results/results.jsonl` + `PROGRESS.md`; negatif sonuç da dürüstçe yazılır.

## Riskler / dürüst notlar
- 6GB'ta MoE hız kazancı garanti değil (küçük matmul overhead'i) — benchmark şart.
- Döngülü derinlik eğitimi kararsız olabilir (gradient büyümesi); warmup / kısmi dondurma denenebilir.
- "Az veriyle kat kat zekâ" ucu açık bir hedef; vekil metriklerle ölçülecek ve iddia öncesi bağımsız doğrulama yapılacak.
