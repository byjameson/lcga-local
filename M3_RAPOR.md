# M3_RAPOR — Uzman prefetch: geçiş matrisiyle bir sonraki uzmanı önceden yükle

**Tarih:** 19.10.2026 · **Kod:** `lcsa/paged_moe.py` (`prefetch` modu), `experiments/exp18_moe_paging.py`
**Amaç:** M2'de uzman isabet oranı %0,6'ydı; token başına talep kopyası (~0,6MB × katman) kritik yolda
stall yaratıyordu. Geçiş istatistiğiyle bir sonraki uzmanı önceden kopyalayıp stall'ı gizlemek.

## 1. Yöntem

- Katman başına **2 uzman slotu**; CPU uzman tensorları **pinned** → H2D kopyalar `non_blocking=True`.
- `trans[katman][önceki, sonraki]`: aynı katmanda ardışık iki uzman çağrısı arasındaki geçiş sayacı
  (token→token geçişlerine karşılık gelir).
- Uzman `e` hizmete girdiğinde `p = argmax P(· | e)` hesaplanır; boş olan slota ön-kopyalanır ve
  `pending[katman] = p` işaretlenir. Sonraki istek `p` ise **prefetch hit** (kopya yok).
- Isınma: 128 eval tokenı üzerinde ileri geçiş (sayaçlar dolar) → istatistikler sıfırlanır → üretim.

## 2. Sonuçlar (1024 eval tokenı + 24 greedy token; iki bağımsız koşu)

| Mod | CE | Talep MB/pass | Hit | Prefetch kopya | Prefetch isabet | tok/s (koşu 1 / 2) |
|---|---|---|---|---|---|---|
| paged (1 slot) | 1,37843 | 15,368 | %0,58 | – | – | 20,9 / 38,2 |
| cap1 (LRU 1) | 1,37843 | 15,368 | %0,58 | – | – | 26,9 / 37,2 |
| lr2 (LRU 2) | 1,37843 | 14,670 | %5,09 | – | – | 19,3 / 38,3 |
| **pf2 (prefetch)** | 1,37843 | **1,238** | **%92,0** | 626 | **%92,7** | 26,3 / 36,5 |
| resident | 1,37843 | 0 | %100 | – | – | 28,7 / 39,4 |

- **Parite:** tüm modlarda CE birebir; greedy örnekler karakter karakter aynı
  (", there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a").
- **Talep kopyası 12,4× azaldı** (15,368 → 1,238 MB/pass): kritik yolda kopya neredeyse yok.
  Toplam trafik ≈ aynı kalır (prefetch kopyaları dahil ~15MB/pass) — kazanç **gecikme gizleme**.
- **Geçiş öngörülebilirliği %92,7:** top-1 routing'in ardışık-token geçişi oldukça yapılı.
  LRU-2'nin %5,1'de kalması, öngörünün basit "son uzmanı tut" olmadığını gösteriyor.
- **tok/s:** 18MB uzman bankalı bu mikro modelde host/Python gürültüsü baskın (aynı mod iki koşuda
  20,9→38,2); pf2, resident bandında ve paged'den kötü değil.

## 3. Sınırlar

- Tek pencere ölçümü (TinyStories valid ilk 1024 token + "Once upon a time"); farklı dağılımda isabet düşebilir.
- Prefetch hataları %7,3 → boşa kopya; E büyüdükçe boşa kopya maliyeti artar (önlem: güven eşiği).
- Kopyalar aynı stream'de: gerçek çakışma (copy/compute overlap) için ayrı stream + event gerekli.
- **Sorgu planlayıcı** kısmı henüz eksik: şu an sabit ısınma penceresi; prompt ön-geçişinden kalıcı
  tahmin tablosu çıkarımı sırada.

## 4. Sıradaki

- Ayrı CUDA stream + event: kopya/compute gerçek overlap; pinned çift tampon.
- E=16/64 + hidden 512 sentetik ölçek testi: isabet ve MB/token eğrileri.
- Prompt profilinden katman-bazlı kalıcı tahmin (planlayıcı) + güven eşiğiyle seçici prefetch.
