# P8 Raporu — İlgili Parametre Kümeleri × Parçalı Eğitim

**Tarih:** 19.10.2026 · **v1 (abort):** 13:38–13:46 · **v2:** 13:57–14:55
**Etiketler:** `p8_rel`, `p8_rnd` (+`exp14_probe` ×3, `corr`) · **Kod:** `experiments/exp14_p8_clusters.py`
**Soru (kullanıcı fikri):** "İlgili" parametre kümelerini gradyan etkileşiminden bulup kümeleri fazlarla sırayla eğitmek — tam eğitime göre ne kaybettirir? İlgililik, rastgele kümeye göre avantaj sağlıyor mu?

## 1) Kümeleme (corr — s3000'den)

- 18 birim: `tok`; 8× (attn: n1+qkv+proj); 8× (mlp: n2+w12+w3); final `norm`.
- Yöntem: her birime `-eps·ĝ` adımı → `s_vu = ||Δg_v|| / (eps·||g_v||)` (6 batch ort.), `W=(C+Cᵀ)/2`, dengeli min-cut (brute force; param-frac 0.35–0.65).
- **Bulunan kümeler:** `C1 = {L1..L6 .mlp}` (6 birim, %40.9 param) — 6 orta-katman MLP'si birlikte hareket ediyor; `C0 = kalan 12 birim` (tok + tüm attention'lar + L0/L7 MLP + norm; %59.1 param).
- Kesim skoru: related **21.19** vs aynı profilli rastgele etiketleme **33.98** → betimleyici (min vs tek çekiliş; tek başına kanıt değil).
- Rastgele kontrol kümesi (rnd): aynı 12/6 profilinde rastgele (frac0 = 0.546).

## 2) Protokol (v2 — düzeltilmiş)

- s3000 ckpt'ten (model + Adam durumu aynen); batch 48×512; 3000 adım → global 3000→6000; LR = P3 takvimi (≈9,5e-4'ten iner), warm Adam.
- **`--phase 300`:** aktif küme her 300 adımda değişir (10 faz: C0→C1→…); diğer küme `requires_grad=False` (Adam None-grad'ı atlar).
- **Kontrol = P3'ün kendi 3000→6000 eğrisi.** Aynı ckpt, aynı tohum (1337) ve aynı veri üretim yolu → **birebir aynı batch dizisi** (her iki koşucuda da adım başına tek `get_batch`, `gen=seed(1337)`; ayrıca doğrulandı). Tek fark: donuklaştırma programı.
- Eval: 128 pencere, seed 999 → üç koşuda **aynı pencereler**.

## 3) v1 olayı ve probe'lar (şeffaflık)

- v1: kod `K`'yı küme sayısı (2) alıp küme her ADIMDA değişiyordu + tam lr → **diverge**: 1.7141 → eval@3500 **3.2427** (train loss 1.93→3.25, tırmanış). İzleme ile erken yakalandı; zincir durduruldu; kanıt yedeklendi (`p8v1_rel_aborted.log`, `p8v1_rel_best_aborted.pt`).
- 200-adım probe (s3000, aynı lr, üç rejim):

| rejim | 0→200 adım val | sonuç |
|---|---|---|
| `full` (tüm param) | 1.7667 → 1.7522 | STABİL |
| `step_alt` (v1 hatası) | 1.7667 → **3.0504** | EXPLODE — mekanizma birebir yeniden üretildi |
| `phase300` (v2) | 1.7667 → **1.7236** | STABİL (en iyi erken iyileşme) |

- Yorum: diverge mekanizması = adım-başına küme değişimi (her parametre iki adımda bir, momentumlu, tam lr) → kararsız; fazlı rejimde sorun yok. exp14'e `--phase` eklendi (5-adım GPU smoke PASS).

## 4) Ana sonuç (v2)

| gstep | kontrol (P3, tam eğitim) | **rel (fazlı)** | rnd (fazlı) |
|---|---|---|---|
| 3000 (start) | 1.7135 | 1.7141 | 1.7141 |
| 3500 | 1.6871 | **1.6404** | 1.6505 |
| 4000 | 1.6807 | 1.6482 | 1.6577 |
| 4500 | 1.6435 | 1.6408 | 1.6280 |
| 5000 | 1.6182 | 1.6272 | 1.6165 |
| 5500 | 1.6005 | 1.6174 | 1.6087 |
| **6000** | 1.5846 | **1.5801** | 1.5920 |
| ppl @6000 | 4.877 | **4.855** | 4.913 |

- **rel ≈ kontrol ve kıl payı önde** (−0.0045 ≈ −%0.28 loss; ppl −0.022): "yarım parametreli fazlı eğitim, tam eğitimin 3000-adım sonucunu yakaladı (hatta küçük farkla geçti)."
- **rnd, kontrolün gerisinde** (+0.0074) ve **rel'den belirgin geride**: rel−rnd = **−0.0119** (ppl 4.855 vs 4.913, ≈ −%1.2).
- Erken evre (3500–4000): iki fazlı kol da kontrolün önünde — faz-1'de C0 (tok+attention) eğitmek hızlı kazanç sağlıyor; kontrol 5500'de öne geçiyor, rel son fazda (C1) tekrar öne geçiyor.
- **Maliyet:** adım 618 ms (rel) / 602 ms (rnd) vs 655 ms (kontrol); **C1 (sadece-MLP) fazlarında 529 ms (−%19)** — donuk attention geri-yayılımı kısaldığı için. 3000 adım: rel 28.8 dk, rnd 29.8 dk.

### 4b) Doğrulama — seed 1338 (bağımsız eğitim tohumu)

Aynı protokol, yalnız eğitim tohumu 1338 (`p8_rel_s1338`, `p8_rnd_s1338`; 15:47–16:47):

| gstep | rel_s1338 | rnd_s1338 |
|---|---|---|
| 3000 (start) | 1.7141 | 1.7141 |
| 3500 | 1.6313 | 1.6400 |
| 4000 | 1.6243 | 1.6319 |
| 4500 | 1.6295 | 1.6167 |
| 5000 | 1.6245 | 1.6111 |
| 5500 | 1.6149 | 1.6046 |
| **6000** | **1.5773** | 1.5861 |

- **Replikasyon ✓:** rel<rnd iki bağımsız tohumda da (Δ = −0,0119 / −0,0088); rel kontrolün (1,5846) iki tohumda da önünde (−0,0045 / −0,0073). rnd↔kontrol: +0,0074 / +0,0015.
- **Ortak desen (post-hoc, 2/2 tutarlı):** son fazda (C1) rel ≈−0,037 kazanırken rnd ≈−0,017…−0,019 kazandı — ilgili çekirdek (6 orta-MLP) blok halinde eğitildiğinde son kazanç belirgin biçimde büyük.
- Süreler tutarlı: 617,6 / 601,9 ms (s1337: 618 / 602).

### 4c) İkinci rastgele-partition (P9a — partition-robustluk)

Aynı protokol/seed (1337), yalnız **rnd partition'ı yeniden çekildi**. 60 çekiliş taranınca kesim skorlarının 21,6–35,0 arasında değiştiği görüldü (rel = 21,19) → kontrol, **orijinal rnd ile kesim profili eşleşmiş** ikinci bir çekilişle yapıldı (seed 2036: frac0 0,546, cut 33,99 ≈ 33,98).

| @6000 | rel_s1337 | rnd_s1337 (cut 33,98) | **rnd2_s1337 (cut 33,99)** |
|---|---|---|---|
| final | **1.5801** | 1.5920 | **1.5924** |
| rel farkı | — | +0,0119 | **+0,0123** |
| ppl | 4,855 | 4,913 | 4,915 |

- **İki farklı rastgele partition aynı sonucu verdi** (1,5920 vs 1,5924; Δ=0,0004) — üstelik erken eğri de neredeyse birebir (3500: 1,6505 vs 1,6506). Yani sonuç "şanslı/şansız çekiliş" değil, **partition'ın yapısal profili** (kesim skoru) tarafından belirleniyor.
- rel avantajı böylece **2 eğitim tohumu × 2 rnd partition** üzerinde tutarlı: −0,0119 / −0,0123 (s1337) ve −0,0088 (s1338, rnd1).
- Not: seed-2024 çekilişi yanlışlıkla rel'e çok benzer çıktığı (cut 21,58; 6 orta-MLP'nin 5'i aynı kümede) için **kullanılmadı** — seçim, eğitim sonucu görülmeden yalnız yapısal kritere göre yapıldı.

## 5) Yorum ve sınırlar (dürüst)

- **Doğrulama tamam (seed 1338):** rel<rnd iki bağımsız eğitim tohumunda da tutarlı (Δ = −0,0119 / −0,0088) ve rel her iki tohumda kontrolün önünde (−0,0045 / −0,0073) → ilgililik avantajı tohum gürültüsü değil, **replike edildi**.
- Rastgele-partition güvenilirliği **P9a ile kapatıldı**: ikinci, profil-eşleşmeli çekiliş ilkini birebir tekrarladı (1,5924 vs 1,5920). Farklar hâlâ küçük (~%0,8 loss) — pratik anlamlılık iddiası bu ölçekle sınırlı.
- Kontrol karşılaştırması batch-eşleşmeli (aynı tohum + veri sırası) → bu yönüyle güçlü.
- Kesim skoru (21.19 vs 33.98) betimleyici; istatistiksel test değil.
- `--phase 300` seçimi probe'a dayanıyor (faz geçiş stabilitesi doğrulandı); 150/600 duyarlılığı test edilmedi.
- Bu bir "aynı adım sayısı" karşılaştırması — "aynı hesap" değil: fazlı kollar adım başına daha az hesap harcadı, yani avantaj onlarda.

## 6) Sonraki

- ~~Seed-1338 doğrulaması~~ ✅ tamamlandı ve **olumlu** (bkz. §4b). Yöntem başarı hanesine geçti (BASARILAR §8.2).
- ~~Yeni-partition rnd~~ ✅ P9a tamam (bkz. §4c): ikinci çekiliş ilkini birebir tekrarladı.
- **Küme-fazlı paralel eğitim** (P6 harness'ı): fazda yalnızca aktif kümeyi senkronize et → iletişim trafiği vs kaliteyi doğrudan ölç (P8 bulgusunun en değerli mühendislik uzantısı).
- MoE-lite sonuçları ayrı rapor (`MoE_RAPOR.md` — yazılacak).
- Olası ileri adımlar: faz uzunluğu/küme sayısı taraması; donuk küme disk/CPU offload ile VRAM tasarrufu senaryosu.
