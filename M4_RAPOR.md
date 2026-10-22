# M4_RAPOR — Çok-birimli scheduler: sığanları paralel, sığmayanları sıralı

**Tarih:** 19.10.2026 · **Kod:** `lcsa/scheduler.py`, `experiments/exp20_scheduler.py`
**Hedef:** Kullanıcı vizyonunun çekirdeği — VRAM'e **sığan** bağımsız birimleri paralel,
**sığmayanları** sıralı çalıştır; çıktıları birleştir. Dürüst ölçüm: hız, VRAM, çıktı paritesi.

## 1. Mekanizma

- **Unit:** (model yolu, VRAM tahmini `est_gb`, prompt, üretim uzunluğu).
- **`plan(units, budget)`:** greedy first-fit — bütçeye sığan ardışık birimler tek grup
  (paralel); taşan birim yeni grup (sıralı).
- **`run_group`:** gruptaki birimler önce yüklenir (sıralı, bellek güvenli), sonra her birim
  **kendi CUDA stream'i + thread'i** ile eşzamanlı üretir, sonra boşaltılır.
- Bütçe = kullanıcı override'ı (küçük kart simülasyonu) veya boş VRAM.
- `merge_outputs(units, policy)`: `concat` (etiketli birleştirme, varsayılan) veya **`best`**
  — birimlerin ürettiği devamın **ortalama token logprob**'u (güven vekili) en yüksek olan
  birimin metni döndürülür; skor yoksa concat'a düşer.

## 2. Sonuçlar (Qwen2.5-0.5B + Qwen2.5-1.5B, 24 token, greedy)

| Senaryo | Bütçe | Plan | Yükleme | **Üretim** | Toplam | Üretim hızlanması | Peak alloc |
|---|---|---|---|---|---|---|---|
| Referans (tek tek) | – | – | 12,97s (soğuk disk) | **4,01s** | 18,66s | 1,00× | ~3,0GB |
| **par** | 5,0GB | `[[0.5B, 1.5B]]` | 7,24s | **2,31s** | **10,86s** | **1,74×** | 3931MB (reserved 4212) |
| **seq** | 2,5GB | `[[0.5B], [1.5B]]` | 6,33s | 3,53s | 11,47s | 1,14× | 2996MB |

- **Çıktı paritesi:** iki senaryoda da metinler referansla **birebir** (greedy) → paralel
  yürütme sonucu değiştirmiyor.
- **Üretim hızlanması 1,74×** (4,01 → 2,31s): iki küçük model tek GPU'da stream'lerle örtüşüyor.
- **Sıralı fallback çalışıyor:** bütçe 2,5GB'a düşünce planlayıcı birimleri ayırdı; peak 3,0GB.
- Not: `toplam` sürelerde disk/önbellek etkisi var (referans yüklemesi soğuktu); **üretim**
  kazancı temiz ölçüttür (modeller GPU'dayken de aynıdır).

## 2b. Cevap birleştirme politikası — `best` (19.10 akşam)

`--merge best` + `--prompt_both` ile iki birim **aynı** prompt'u yanıtlar; satır başına
`units.*.score` (devamın ortalama logprob'u), `best_pick`, `merged_best` kaydedilir.

| Demo | 0.5B çıktısı | 1.5B çıktısı | skorlar | seçim |
|---|---|---|---|---|
| Aritmetik ("What is 17 times 23?") | " 391." | " 391." | −0,1671 / **−0,1333** | **1.5B** (aynı metin) |
| Hikâye ("Once upon a time in a small village") | " lived a young man named Jack…" | ", there was a wise old owl…" | — / — | **1.5B** (metinler farklı) |

- Skor farkı beklendiği yönde: 1.5B devamına daha yüksek güven → politika onu seçti.
- Hikâye demo: iki birim **farklı** metin üretti; `merged_best` 1.5B'nin devamı
  (greedy, seed yok; iki senaryoda da aynı seçim, çıktılar referansla birebir).
- Dürüst not: logprob-güven **doğruluk garantisi değil**; aynı çıktıda seçim bilgi taşımaz
  (aritmetik demo), farklı çıktıda ise yalnız bir sezgisel. Oylama/rerank hâlâ yok.

## 2c. Model havuzu + LRU (19.10 gece)

`lcsa/pool.py`: `ModelPool(budget)` yüklü modelleri GPU'da tutar; yeni bir model bütçeye
sığmazsa **en eski kullanılan GPU sakini** host RAM'e taşınır (evict) ve gerektiğinde
**diskten değil RAM'den** geri yüklenir (reload). İstek betiği: [0.5B+1.5B] → [0.5B+1.5B] → [1.5B].

| Senaryo | r1 | r2 | r3 | Toplam | Not |
|---|---|---|---|---|---|
| Havuzsuz (her istekte yükle/boşalt) | 11,38s | 8,13s | 5,65s | **25,16s** | her istekte disk yüklemesi |
| Havuz (5,0GB bütçe) | 6,52s | 1,63s | 1,02s | **9,17s (2,74×)** | r2: **load 0,00s**; iki model sakin |
| Dar bütçe (2,5GB) | 2,05s | 5,39s | 1,74s | **9,18s** | r2: 0,5B evict (0,38s) → 1,5B yükle; r3: 1,5B evict (0,68s) → 0,5B **reload 0,35s** (disk yüklemesi 1,40s) |

- İlk sürümde bütçe yalnız *disk* yüklemesinde uygulanıyordu; reload bütçeyi aşabiliyordu
  (tight r3'te resident 4,4GB > 2,5GB). **v2**: reload öncesi de evict. Satır etiketleri:
  `m4_pool_*` (v1, hatalı) ve `m4_pool_v2_*` (düzeltilmiş) — v1 satırları log'da duruyor.
- Dürüst notlar: evict edilen model host RAM'de kalır (üretimde düşürülmeli); evict+reload
  bedava değil; `est_gb` beyan edilir (ölçülmez); fragmentasyon modellenmez.

## 3. Dürüst sınırlar

- İki küçük model demo; **havuz (2c)** istek başına yükleme maliyetini kaldırdı ama havuz
  kapasitesi/bütçe elle veriliyor ve evict edilenler host RAM'de tutuluyor.
- `est_gb` elle veriliyor; otomatik tahmin (ağırlık + KV + aktivasyon + fragmentasyon) yok.
- "Paralel" kazanç yalnız VRAM'e birlikte sığan birimlerde; 6GB kartta iki büyük model yine
  sığmaz → planlayıcı doğru kararı verir ama kazanç yok (3B sınıfı için paged motor şart).
- Birleştirme: **skora dayalı seçim** var (`merge best`), ama oylama/rerank yok; seçim
  keyfî çıktıda yanlış olabilir. Sorguya göre birim seçimi yok.
- Kuyruk/çoklu istek/öncelik yok; tek seferlik batch.

## 4. Sıradaki

- ~~**Model havuzu + LRU**~~ ✅ (2c) — kalan: havuz bütçesinin otomatik yönetimi, evict edilenin
  host RAM'den tamamen düşürülmesi, çoklu-istek kuyruğu.
- **Otomatik bütçe:** parametre sayısından est_gb + KV hesabı; fragmentasyon payı; gerçek
  non-fitting senaryoyu otomatik saptama.
- **Sorgu yönlendirme:** sorguya göre birim seçimi (adapter/compute branch); aynı soruya
  çok-birimli cevap → oylama/seçim politikası.
- **Paged entegrasyon:** sığmayan birimler (3B bf16) paged motorla sıralı çalışsın (M1 ile).
