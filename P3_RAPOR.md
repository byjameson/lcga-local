# P3 Raporu — Kendi Mini-LM (17.31M) · 18.000 adım — BİTTİ

**Tarih:** 19.10.2026 13:37 · **Etiket:** `p3` · **Durum:** tamamlandı (yaşayan eğri, kesintili 5 oturum)
**Veri:** `ts` (TinyStories token akışı, BPE-8k; train 466.76M tok) · **Görülen token:** 442.37M (≈0,95 epoch)
**Donanım:** RTX 4050 Laptop 6GB, bf16 autocast

## 1) Sonuç

| metrik | değer |
|---|---|
| val_loss (eval@18000) | **1.3890** |
| ppl | **4.011** |
| **bpb** (byte-normalize) | **0.4892** |
| parametre | 17.308.032 (tied embedding dahil) |
| adım süresi | 654–655 ms (batch 48×512 = 24.576 tok/adım) |
| throughput | 37.373 tok/s (son oturum) |
| VRAM | 5,90 GB sabit |
| duvar | son oturum 87,7 dk; **toplam ≈ 4,1 sa** (parçalı) |

## 2) Karşılaştırma — a2/a3 kontrol koşuları (aynı tokenizer ailesi)

| koşu | adım | token | val_loss | bpb | süre |
|---|---|---|---|---|---|
| a2_ctrl8k | 6.000 | 98,3M | 1.6093 | 0.5612 | 48 dk |
| a3_ctrl8k | 12.000 | 196,6M | 1.5095 | 0.5264 | 96 dk |
| **P3 (bu koşu)** | **18.000** | **442,4M** | **1.3890** | **0.4892** | 87,7 dk (son oturum) |

- bpb kazanımı: a3'e göre **−0,0372 (−7,1%)**; a2'ye göre −0,0720 (−12,8%).
- **Akış doğrulaması:** `tok2_ctrl8k_valid` = `ts_valid`'in birebir öneki (ilk 100 token ve offset-0 eşleşmesi test edildi) → üç koşu da aynı metin akışından değerlendirildi. Pencereler farklı aralıklardan çekildi (seed 999) → ±küçük örneklem gürültüsü; büyük farkların yorumunu etkilemez.
- Verim: P3 37.373 tok/s ≈ a3'ün 34.133 tok/s'sinden hızlı (batch 48 + chunk'lı lm_head fix).

## 3) Eval eğrisi (val_loss)

| adım | 0 | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | 7000 | 8000 | 9000 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 9.0663 | 2.0768 | 1.8484 | 1.7135 | 1.6807 | 1.6182 | 1.5846 | 1.5600 | 1.5338 | 1.5135 |

| adım | 10000 | 11000 | 12000 | 13000 | 14000 | 15000 | 16000 | 17000 | 18000 |
|---|---|---|---|---|---|---|---|---|---|
| val | 1.4900 | 1.4867 | 1.4714 | 1.4437 | 1.4283 | 1.4164 | 1.4022 | 1.3939 | **1.3890** |

- İniş son ana kadar sürdü (son 500 adım: 1.3905 → 1.3890) ama eğim belirgin şekilde azaldı: 18k civarında **doygunluk başlangıcı**. 24k adımda muhtemelen yalnızca birkaç binde bir iyileşme daha gelir.

## 4) Üretim örnekleri (final)

Sıcaklık 0,8 (top-k 100):
> "Once upon a time, there was a young bear named Bob. He was walking in the woods one day when he saw a big, huge, impressive tree. Bob was so excited that he ran towards it and started to jump and jump as high as he could. Suddenly, he heard a loud noise. It was coming from the tree. Bob was so scar…"

Sıcaklık 0,6:
> "Once upon a time, there was a little girl named Lily. She loved to play outside in the garden with her mom. One day, while they were playing, Lily's mom noticed that her phone was missing. She asked Lily if she had seen it, but Lily shook her head no. Lily's mom tried to find the phone, but she co…"

(İki örnek de dilbilgisel ve tematik olarak tutarlı; anlatı kapanışı zayıf — bu boyutta beklenen.)

## 5) Mühendislik notları (bu fazın kalıcı kazanımları)

- **Perf fix (kritik):** İlk oturum (16.10) WDDM taşmasıyla çöküyordu (lm_head tam matmul ~0,4G bf16 + backward). `MiniLM.loss` artık lm_head'i de chunk'lı → matematik birebir (smoke: Δ=4,8e-7), 655 ms/adım @5,90G — orijinalden hızlı ve stabil.
- **Resume zinciri:** ckpt = model + opt + step + evals (atomik); resume'da `del ck` + `empty_cache()` (RAM şişmesi fix). Her 1000 adımda `p3_ckpt.pt`; sabit yedek `p3_ckpt_s3000.pt` P8 için alındı.
- **Kesintisiz iş akışı:** WMI ile oturumdan bağımsız koşu; çoklu duraklat/devam döngüsü kayıpsız işledi (3000→8000→10000→18000 tüm ckpt'ler doğrulandı).

## 6) Sınırlar / dürüst çerçeve

- bpb, `ts_valid`'in **decode edilmiş gerçek baytıyla** (19.218.350 B) hesaplandı; a2/a3'ün bpb'si kendi diliminden — normalizasyon farkı ≈%1 mertebesinde, sonuçları değiştirmez.
- Val pencereleri üç koşuda farklı örnekler (aynı dağılım) → ±0,005–0,01 val_loss gürültüsü varsayımı.
- **(19.10 ek — MoE deneyinde ölçüldü):** pencere-SETİ sistematik farkı bu tahminden büyük çıktı: aynı model a2-pencerelerinde 1.6606 ↔ ts-pencerelerinde 1.6264 (Δ≈0,034). Farklı pencere setleri arası val/bpb kıyaslarında bu mertebeye kadar set etkisi olabilir; P3–a3 farkı (0,12) bundan büyük olduğundan ana sonuç etkilenmez, ancak raporlar arası küçük farklar yorumlanırken akılda tutulmalı.
- "Kendi LM'i" kalıp/tekrar eğilimli; anlatı seviyesi hâlâ mini ölçek (17007 parametre/katman ölçeğinde dürüst beklenti).

## 7) Sonraki

- **P8** — küme-farkındalıklı parçalı fine-tuning (related vs random; s3000'den). *Not: ilk deneme protokol hatasıyla (adım-başına küme değişimi) diverge etti; düzeltme + probe doğrulaması ile yeniden koşuluyor — bkz. P8 raporu.*
- **MoE-lite (exp15)** — E=4 top-1, eşit toplam param, ~1/4 aktif MLP; zincirde bekliyor.
