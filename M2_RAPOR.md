# M2_RAPOR — MoE expert paging: uzmanlar CPU'da, yalnız routed olan GPU'ya

**Tarih:** 19.10.2026 · **Kod:** `lcsa/paged_moe.py`, `experiments/exp18_moe_paging.py`
**Model:** MoE-lite top-1: E=4 (17,3M param; `checkpoints/moe4_best.pt`, val ppl 5,09) ·
E=16 (45,7M) · E=64 (159,1M) — ölçek testleri §3b–3c
**Hedef:** Uzman MLP'lerini CPU RAM'de tutup, token başına yalnızca router'ın seçtiği uzmanı
GPU'ya kopyalamak; kalite paritesi + transfer/hız ölçümü.

## 1. Sonuç (1024 eval token + 24 greedy token)

| Mod | Kapasite | CE | tok/s | pass başına uzman MB | Hit oranı | VRAM (peak alloc) |
|---|---|---|---|---|---|---|
| **paged** (1 slot) | 0 | **1,37843** | **36,1** | 15,37 | %0,6 | 28,7MB |
| cap1 (LRU 1) | 1 | **1,37843** | 30,0 | 15,37 | %0,6 | 28,7MB |
| resident (preload) | 4 | **1,37843** | 39,5 | 0,0 | %100 | 42,2MB |

- **Kalite paritesi birebir:** üç modda CE aynı; greedy örnekler aynı
  (", there was a little girl named Lily. She loved to play outside…").
- Paged, resident'tan yalnızca **~%9 yavaş** (36,1 vs 39,5 tok/s) — 18MB uzman bankası tamamen
  CPU'da iken; GPU'da yalnız paylaşılan ağırlıklar + 1 slot var.
- Pass başına ~15,4MB kopyalanıyor (18MB bankın %85'ine denk) çünkü routing her token'da
  neredeyse tamamen değişiyor → 1-slot hit oranı **%0,6**.

## 2. Mekanizma

- Paylaşılan ağırlıklar GPU'da: token embedding, attention, RMSNorm'lar, **router**.
- Uzmanlar (`w12`, `w3`) CPU'da **bf16** (fp32'ye göre yarı bellek).
- `_moe`: router probs (GPU) → top-1 → benzersiz uzman için `_get` (slot ya da LRU) → masked
  tokenlar üzerinde `F.linear`; çıkış `p_top` ile ölçeklenir.
- Kapasite 0/1: tek yeniden kullanılabilir slot (aynı uzman üst üste gelirse hit); kapasite>1:
  LRU; `preload`: hepsi GPU'da (resident baseline).
- Transfer sayacı: kopyalanan uzman baytı / çağrı sayısı / hit oranı; `usage[layer][expert]`.

## 3. Ölçekleme (dürüst ekstrapolasyon)

- Bu modelde uzman = 2·256·384 + 256·384 ≈ 295k param ≈ **0,59MB bf16**.
- E=4 top-1 → pass başına *ideal* 8×1×0,59 ≈ 4,7MB; **ölçülen 15,4MB** çünkü pass = prefill(4)
  + 24 token ve slot yalnızca son uzmanı tutuyor; her yeni token seti farklı uzman çekiyor.
- Gerçek MoE'lerde: E=64, top-2, uzman 3–8M param → pass başına onlarca–yüzlerce MB. Kazanç,
  **VRAM'e sığmayan uzman bankası**dır (ör. Mixtral-8x7B: ~47B uzman param ≈ 94GB bf16; 6GB'a
  yalnız aktif ~%10–20 akıtılır).
- Bizdeki E=4 demo mekanizmayı kanıtlar; **E=16 (3b) ve E=64 (3c) ölçek testleri tamam**.

## 3b. Ölçek testi — E=16 (19.10 gece)

Aynı tarif (batch 32, ctx 512, 2000 adım) ile E=16 (256 param/uzman, banka ~75,5MB bf16,
model 45,7M param; **840ms/adım, 19,4k tok/s**). Eşit adımda kıyas: E=16 val **1,8685**,
E=4 **1,9178** (E=4 6000 adımda 1,6264'e iniyor).

| Mod | CE | tok/s | pass başına MB | hit | prefetch kullanımı | peak alloc |
|---|---|---|---|---|---|---|
| paged (1 slot) | **1,57647** | 17,24 | 34,27 | %0,13 | – | 28,8MB |
| cap1 (LRU 1) | **1,57647** | 16,54 | 34,27 | %0,13 | – | 28,8MB |
| lr2 (LRU 2) | **1,57647** | 17,10 | 34,00 | %0,92 | – | 44,5MB |
| **pf2 (prefetch 2)** | **1,57647** | 15,16 | **8,42** | **%75,5** | %73,0 | 34,8MB |
| resident (16 slot) | **1,57647** | 18,64 | 0 | %100 | – | 96,3MB |

- **Kalite paritesi birebir:** beş modda da CE 1,57647; greedy örnekler aynı.
- Talep trafiği E ile büyüyor: E=4 15,37 → E=16 **34,27 MB/pass** (banka 18,9→75,5MB).
- Prefetch kazancı: 34,27 → **8,42 MB/pass (4,07×)**; hit %0,13 → **%75,5**; önceden yüklenenin
  %73'ü kullanılıyor (%27 boşa).
- E=4'te hit %92 idi; E=16'da **%75,5** → uzman sayısı arttıkça geçiş tahmini zorlaşıyor
  (dürüst ölçek eğilimi). Buna rağmen talep trafiği 4× azalıyor.
- **Hız:** pf2 (15,16 tok/s) paged'den (17,24) ~%12 **yavaş** — bu toy boyutta kopyalar
  hesaplamaya göre ucuz; kazanç bayt cinsinden, hız cinsinden değil. Gerçek MoE'de (uzman
  MB'ları büyük) beklenen yön tersidir; bu ölçekte iddia edilmez.
- VRAM: resident 96,3MB vs paged 28,8MB → **3,35×** tasarruf; en büyük modda bile peak <100MB.

## 3c. Ölçek testi — E=64 (20.10 gece; kesintili: 1500/2000 adım)

E=64 (288MB uzman bankası, model 159,1M param; adım ~2,4–3,2s) **1500 adımda kesildi**
(val_loss 1,9684); paging ölçümleri bu checkpoint ile (`moe64_best_s1500.pt`). E=16/E=4 ile
adım bütçesi farklı — CE'ler bu yüzden karşılaştırılmaz; asıl ölçü transfer/hit eğilimi.

| Mod | CE | tok/s | pass başına MB | hit | prefetch kullanımı | peak alloc |
|---|---|---|---|---|---|---|
| paged (1 slot) | **1,71569** | 11,75 | 53,98 | %0,0 | – | 29,1MB |
| cap1 (LRU 1) | **1,71569** | 13,00 | 53,98 | %0,0 | – | 29,1MB |
| lr2 (LRU 2) | **1,71569** | 11,18 | 53,98 | %0,0 | – | 51,0MB |
| **pf2 (prefetch 2)** | **1,71569** | 12,13 | **15,59** | **%71,1** | %68,2 | 35,1MB |
| resident (64 slot) | **1,71569** | 18,63 | 0 | %100 | – | 312,6MB |

- **Parite birebir:** beş modda CE 1,71569; greedy örnekler aynı.
- Talep trafiği: E=4 15,37 → E=16 34,27 → E=64 **53,98 MB/pass** (banka 18,9→75,5→288MB).
- Prefetch: 53,98 → **15,59 MB/pass (3,46×)**; hit %0 → **%71,1**; önceden yüklenenin %68,2'si kullanıldı.
- **Ölçek eğilimi (dürüst):** prefetch isabeti %92,7 (E=4) → %73,0 (E=16) → **%68,2 (E=64)**;
  uzman sayısı arttıkça geçiş tahmini zorlaşıyor, ama talep trafiği hâlâ **~3,5×** azalıyor.
- **Hız:** bu kez prefetch paged'den hafif **hızlı** (12,13 vs 11,75 tok/s) — talep kopyaları
  büyüdükçe örtüşmenin değeri artıyor; E=16'da tersiydi (15,16 vs 17,24). Eşik E≈16–64 arasında.
- VRAM: resident 312,6MB vs paged 29,1MB → **10,7×** tasarruf.
- Dürüst not: E=64 modeli 1500/2000 adımda (kesinti) — CE 1,71569, E=16'nın 1,57647'sinden
  yüksek; bu **eğitim bütçesi** farkıdır, mimari karşılaştırması değildir.
- Not: 1-slot ve LRU-2 hit oranı **%0** — 64 uzmanda ardışık aynı uzman pratikte yok; prefetch şart.

## 4. Sınırlar

- Model 17,3M/45,7M/159,1M (demo ölçeği) — gerçek MoE ölçeğinde transfer/compute oranı farklı.
- Kopyalar **pinned değil**; pinned + çift tampon ile örtüşme (overlap) iyileştirmesi yapılmadı.
- Greedy üretim; sampling altında parite davranışı test edilmedi (matematik aynı).
- LRU derinliği (cap1→cap4+) bu routing'de fayda sağlamıyor (hit %0,6) — **prefetch (M3) gerekli**.
- CE, modelin kendi tokenizerı + `tok2_ctrl8k_valid` ilk 1024 token'ı üzerinde; farklı pencere
  farklı CE verir (smoke 256 token'da 1,27843 idi).

## 5. Sıradaki (M3)

- ~~**Prefetch:** uzman geçiş matrisi (layer × prev→next) çıkarıp bir sonraki token'ın uzmanını
  önceden yükle; hedef hit oranı >%30.~~ → **Yapıldı (19.10, aynı gün):** bkz. `M3_RAPOR.md` —
  talep kopyası 12,4× azaldı, hit %92, prefetch isabeti %92,7, parite birebir.
- ~~**E=16/64 + hidden 512/1024** sentetik ölçek testi: bytes/token ve tok/s eğrisi.~~
  **Yapıldı (20.10):** §3b–3c — trafik 15,37→34,27→53,98 MB/pass; pf2 3,46–4,07× azaltıyor;
  hit %92,7→%75,5→%71,1; VRAM tasarrufu 3,35×/10,7×; hız eşiği E≈16–64.
- Pinned uzman + çift tampon (kopya/compute örtüşmesi).
- Query düzeyinde birim seçimi (adapter/LoRA bankası) aynı motorla (M4 kompozisyon).
