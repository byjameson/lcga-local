# MoE-Lite Raporu (exp15) — E=4 top-1, eşit toplam parametre

**Tarih:** 19.10.2026 14:56–15:47 (50,7 dk) · **Etiket:** `moe4` · **Kod:** `lcsa/mini_lm.py` (`MoESwiGLU`) + `experiments/exp11_own_lm.py --moe_experts 4`
**Soru:** Token başına uzmana yönlendirme (parametre-düzeyi router), eşit TOPLAM parametrede dense'i geçer mi; eşit AKTİF parametrede ne kazandırır?

## 1) Tasarım

- Her MLP yerine **4 uzmanlı top-1** MoE: `y = p_top(x) · Uzman_top(x)` (Switch stili; router gradyanı `p_top` üzerinden akar), uzman gizli boyu 256 (=1024/4) → **eşit toplam param**.
- Yük-dengeleme aux: `α·E·Σ f_e·P_e`, α=0.01 (loss'a eklenir).
- Toplam 17,32M param (dense 17,31M); **aktif/token ≈ 10,25M** (MLP'nin ~1/4'ü).
- Protokol: a2 ile birebir — batch 32×512, 6.000 adım (98,3M token), warmup 300, LR 1e-3 cosine, `ts` verisi, n_eval 128.

## 2) Adil karşılaştırma (aynı 128 pencere, aynı yöntem)

> Tüm satırlar aynı pencerelerle yeniden değerlendirildi; kontrol koşuları **4 ondalığa kadar birebir** üretildi (a2: 1.6093, flat4: 1.6888, loop2x4: 1.6521) → karşılaştırma güvenilir.

| koşu | toplam param | aktif/token | val_loss | ppl | **bpb** | süre |
|---|---|---|---|---|---|---|
| a2_ctrl8k (dense 8 katman) | 17,31M | 17,31M | 1.6093 | 4.999 | **0.5612** | 48 dk |
| p4_loop2x4 (döngülü) | 10,2M | 10,2M | 1.6521 | 5.218 | 0.5761 | 48 dk |
| **moe4 (bu deney)** | **17,32M** | **≈10,25M** | **1.6606** | **5.262** | **0.5791** | 50,7 dk |
| p4_flat4 (dense 4 katman) | 10,2M | 10,2M | 1.6888 | 5.413 | 0.5889 | 29 dk |

## 3) Bulgular

- **Toplam-param eşitliğinde dense'i geçemedi:** 0.5791 vs 0.5612 (%3,2 rel. kötü). Beklenen yön — MoE'nin aktif hesabı ~%40 az.
- **Eşit-aktif (≈10,2M) karşılaştırmasında flat4'ü geçti** (0.5791 vs 0.5889; %1,7 rel. iyi); ancak **loop2x4'ün hafif gerisinde** kaldı (0.5761; %0,5). Yani bu ölçekte "10M sınıfının" en verimlisi hâlâ loop2x4 (döngülü derinlik), MoE onu geçemedi.
- **Router sağlığı ✅:** aux çalıştı — kullanım başlangıçtaki çarpık dağılımdan (ör. L2'de bir uzman 0.75) eğitim sonunda **tüm katmanlarda ~0.25/0.25/0.25/0.25**'e oturdu; temsilci çökmesi yok.
- **Hız:** 506 ms/adım (dense 480, loop 482). MLP backward'ının %75'i kesilmesine rağmen kazanç ~%5-8'de kaldı: 8 katman attention + Python uzman-dispatch döngüsü + router. Dispatch optimizasyonu (gruplu/toplu matmul) yapılmadı.
- Üretim örnekleri akıcı (bkz. `results/moe4.log`); örnek üretimi yavaş (7-8 s vs dense ~1,3 s) — Python döngüsü etkisi.

## 4) Sınırlar / dürüst çerçeve

- **Tek seed** (1337); başka tohumla doğrulama yapılmadı.
- Uzmanlaşmanın içeriği artık **incelendi** (§5): net token-tipi uzmanlaşması var (noktalama/zamir/frekans), pozisyon uzmanlaşması yok.
- **Pencere-seti farkı ölçüldü:** aynı model a2-pencerelerinde 1.6606 ↔ ts-pencerelerinde 1.6264 → setler arası sistematik fark ≈0.034. Bu, farklı pencere setlerinden gelen koşular arasındaki kıyaslarda (±0.03 mertebesine kadar) dikkate alınmalı (bu raporun tablosu tek-set içinde olduğu için etkilenmez).
- bpb dönüşümünde global bayt/token oranı kullanılıyor (a2 ailesiyle aynı formül).
- "Eşit adım" karşılaştırması; "eşit FLOP" değil (MoE'nin aktif hesabı az, duvar süresi ~eşit).

## 5) Uzmanlaşma analizi (moe4_best.pt, 24.576 token)

Yöntem: 48 valid penceresi (seed 999) üzerinde her katmanın top-1 ataması toplandı; token özellikleriyle NMI, token-içi atama entropisi ve **karıştırma (shuffle) taban çizgisi** hesaplandı (`scripts/moe_routing_analysis.py`; ham çıktı `results/moe_routing.json`).

- **Kullanım dengeli:** tüm katmanlarda 0,19–0,28 (aux çalışıyor; çökme yok).
- **Yönlendirme token-tipi odaklı, pozisyon değil:** pozisyon NMI ≈ 0,0007 (taban 0,0001); buna karşılık noktalama 0,074, kelime-başı 0,052, frekans kovası 0,048, uzun token 0,027 (taban ~0,0001).
- **Token-içi entropi** (0 = hep aynı uzman, 1 = rastgele): L0 **0,221** · L1 0,298 · L2–L5 0,59–0,65 · L6 0,405 · L7 0,307 (shuffle ~0,99) → özellikle erken/geç katmanlarda neredeyse **deterministik token→uzman eşlemesi**.
- **Router güveni derinlikle artıyor:** ortalama top-1 olasılığı L0 0,36 → L7 0,85; p<0,6 oranı L0 %97 → L7 %12. Erken katman router'ları "zayıf ama tutarlı" (aynı token hep aynı uzman, düşük marj), geç katmanlar "güvenli leksem uzmanı"; orta katmanlar (L2–L5) düşük güven + en yüksek bağlam duyarlılığı (entropi ~0,6).
- **Uzmanlar yorumlanabilir "leksem uzmanları" (L6):** E3 = özne zamirleri (" He" %97, " She" %98, " he" %80, " They" %76); E2 = virgül/bağlaç (" ," %100, " with" %89, " and" %74); E1 = " to" %81, "\n" %75; E0 = " very" %92, " was" %80, " so" %76.
- **Frekans/uzunluk ayrışması (L4):** nadir (<50) ve 50–500 tokenlar → E1 (%51/%44); uzun tokenlar → E1 %53; sık tokenlar → E2/E3 (%30/%31); noktalama → E3 %63.
- **Katmanlar arası ortak yapı zayıf:** ortalama ofdiyagonal NMI **0,033** (en yüksek L1–L6 ve L5–L6 = 0,12) → her katman kendi bölümlemesini kuruyor.
- **Yorum (dürüst):** Bu ölçekte/bütçede router, "bağlam-duyarlı bir token-sınıfı hash'i" gibi davranıyor; pozisyon veya daha üst düzey sözdizimsel yapı keşfedilmiyor. Bu, kalite sonucuyla tutarlı: MoE kapasiteyi token-tipi bölümlemesiyle kullanıyor, anlamsal uzmanlaşma henüz yok. Daha uzun eğitim/daha büyük veri ile değişebilir (test edilmedi).
- Sınırlar: 24.576 token örneklem; NMI'ler mutlak olarak küçük ama shuffle tabanına göre net; `p_expert` örneklem-içi koşullu frekanstır.

## 6) Sonraki (öneriler)

1. ~~Uzmanlaşma analizi~~ ✅ yapıldı (§5).
2. E taraması (E=2/8) + uzman boyu varyasyonu; dispatch'i gruplu matmul ile hızlandırma.
3. loop2x4 ile birleşim: "döngülü + MoE" (10M sınıfında yeni SOTA adayı).
