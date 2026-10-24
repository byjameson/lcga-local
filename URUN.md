# LCGA Local — 6 GB'de çalışan yerel sohbet asistanı

LCGA Local, **tek komutla açılan** bir yerel yapay zekâ asistanıdır. API, sunucu
veya web uygulaması değildir: terminalde çalışır, **çalışırken internet
istemez** ve modeller bilgisayarınızdan çıkmaz (ilk model indirmesi hariç).

```powershell
python llm.py
```

veya Windows'ta `sohbet.cmd` dosyasına çift tıklayın.

## Neden var?

RTX 3050 Laptop **6 GB** VRAM'e bf16 bir 4B model sığmaz (≈8 GB). LCGA Local,
Qwen3-4B-Instruct-2507'yi **int8 olarak GPU'da tutup her katmanı sırayla bf16'ya
açarak** çalıştırır:

- 36 katmanın tamamı int8 olarak GPU'da: **3465 MB**
- embed + tek bf16 dequant slotu + KV cache: toplam **~4.5 GB**
- ölçülen üretim hızı: **~3.4 tok/s** (kısa cevaplar 1–5 s); ilk açılışta
  int8 önbelleği yoksa **~18.8 s**, önbellek oluştuktan sonra ölçülen süreç
  başlangıcı **~2.4 s**
- ölçülen peak VRAM: **4513 MB**, yükleme sonrası boşta **~0.59 GB**

Bu, "6 GB'de 4B" iddiasının ölçülmüş hâlidir. 3B/1.5B modelleri de vardır;
4B yoksa veya istenmezse otomatik olarak daha küçük modele düşer.

## Kurulum

1. Python 3.11 ve CUDA'lı PyTorch (depo kökündeki `requirements.txt`).
2. 4B modelini indirin (~8.0 GB, devam ettirilebilir):

   ```powershell
   python scripts\download_qwen3_4b.py
   ```

3. Sohbeti başlatın:

   ```powershell
   python llm.py
   ```

En az bir Instruct modeli gerekir (4B/3B/1.5B'den biri); hiçbiri yoksa
`llm.py` açılmaz ve `[hata] hicbir Instruct modeli bulunamadi` yazar. Elinizde
hangi model varsa onu kullanır (`chat4b` → `chat3b` → `chat` sırasıyla). İlk
istekte model yüklenir, ekranda `[model yukleniyor, ilk seferde ~20 sn, sonra ~3 sn]`
görürsünüz. İlk 4B açılışı int8 ağırlıkları yeniden hesaplar ve **~3.4 GB**
boyutunda bir önbellek yazar; sonraki süreçler bu dosyayı doğrulayıp nicemlemeyi
atlar. Önbellek `cache/` altında oluşur; disk alanı yetersizse veya
`LCGA_NO_QUANT_CACHE=1` ayarlıysa ürün önbelleksiz yolu kullanır.

## Kullanım

```text
Sen: Fransa'nın başkenti neresi?
Asistan: Fransa'nın başkenti Paris'tir.
```

Komutlar:

| Komut | İş |
|---|---|
| `/yardim` | kısa yardım |
| `/model` | kullanılabilir modelleri listeler |
| `/model chat4b` | 4B'yi açıkça seçer (`chat3b`, `chat`, `deep`, `fast` de olur) |
| `/temizle` | konuşma geçmişini siler |
| `/durum` | son cevabın rota/süre/VRAM özeti |
| `/cik` | çıkış |

Varsayılan mod **`quality`**: genel sorular ve kod → `chat4b`; yalnızca kısa
nezaket cümleleri → 1.5B `chat`. 4B yoksa 3B, o da yoksa 1.5B kullanılır.
`Ctrl+C` cevabı keser; geçmiş silinmez.

### Yerel RAG (deneysel, kaynaklı)

Varsayılan ürün yolu, `data/rag/world_facts.jsonl` içindeki **326 Wikipedia
özeti / 4.793.376 bayt (4,57 MiB)** üzerinde **CPU-only BM25** çalıştırır. Tam
korpusta indeks yükleme ölçülen **1,55 s** (ikinci ölçüm 2,14 s); GPU
kullanmaz, runtime'da internet yoktur. Kaynak yalnızca sorguyla yeterli kelime
örtüşmesi varsa prompt'a eklenir; selamlama, liste, kod, yaratıcı/mesaj ve
"tam olarak şunu yaz" isteklerinde devreye girmez. Kesin olgu sorularında
**tek kaynak**, geniş sorularda yalnızca konuyu bağımsız destekleyen **ikinci
kaynak** kullanılır; `/durum` kaynak başlıklarını gösterir; `--no-rag` ile
kapatılır. Bu genel bir web araması değildir ve yalnızca korpustaki sayfaları
kapsar; kapsam dışı sorularda normal 4B yolu kullanılır.

**Ölçüm (`exp48`, 50 kapsanan + 20 kontrol, baseline/RAG, `max_new=64`):**
CPU retrieval tanısı **50/50** kapsanan soruda beklenen olguyu bağlamda buldu;
**50/50** sorguda tek kaynak seçildi; **20/20** kontrolde RAG devreye girmedi.
Kapsanan sorularda GPU keyword isabeti **42/50 baseline → 49/50 RAG**; yeni 30
soruda **24/30 → 30/30**. RAG cevaplarının **50/50**'si manuel okumada olgusal
olarak doğru; atıf **50/50**, RAG kullanımı **50/50**. Tek keyword kaçağı
`tr-ocean`'dır: cevap doğru Türkçe ad olan **Büyük Okyanus**'tur, beklenen
literal “Pasifik” geçmez. Kontrollerde RAG ve atıf **0/20**; baseline/RAG
çıktıları aynıydı. Strict exact **4/36 vs 0/36**; RAG atıf eklediği için bu
metrik anlamlı değildir. Ortalama süre **5,82/5,85 s**, ortalama üretim
**5,74/5,33 s**, peak **4517,8/4547,7 MB**. Bu 50 sorunun tamamında kaynak
başlıkları doğrudan konuyla ilgiliydi; önceki `tr-nile`, `tr-whale`, `tr-nl`
ve `tr-ocean` gibi kaynak karışmaları bu sette düzeldi.

**Sınır:** 326 sayfa genel bilgi benchmark'ı değildir. Sıralama bu dar sette
tek kaynağa indirildi, ancak kapsam dışı veya çok benzer sayfalarda hata
yapabilir; geniş sorularda en fazla iki kaynak kullanılır. `[1]` biçimi tek
başına doğruluk kanıtı değildir. Ham GPU cevapları
`results/exp48_rag_eval.json`, kaynak denetimi
`results/exp48_retrieval_check.txt`, inceleme notları
`results/exp48_manual_review.md` içindedir.

Tek seferlik soru:

```powershell
python llm.py --stats "Türkiye'nin başkenti neresi?"
```

## Modeller

| Birim | Model | Bellek | Hız (ölçülen) | Ne zaman |
|---|---|---|---|---|
| `chat4b` | Qwen3-4B-Instruct-2507, int8 GPU-resident | ~4.5 GB | ~3.4 tok/s | Varsayılan kalite |
| `chat3b` | Qwen2.5-3B-Instruct, int8 paged | ~4.75 GB | ~2.7–3.0 tok/s | 4B yoksa / `/model chat3b` |
| `chat` | Qwen2.5-1.5B-Instruct, dense bf16 | ~3.3 GB | ~0.9 s kısa cevap | Nezaket, hızlı yedek |
| `deep` | Qwen2.5-3B + kendi WT2 checkpoint'i | ~4.7 GB | ~2.6–3.9 tok/s | Kod/akıl yürütme denemesi |
| `fast` | Qwen2.5-0.5B | ~1.1 GB | çok hızlı | Yedek |
| `balanced` | Qwen2.5-1.5B | ~3.3 GB | hızlı | Yedek |
| `alt` | SmolLM2-1.7B-Instruct | ~3.4 GB | ~0.8 s | Yalnızca açık seçim; Qwen dışı adaptör testi |

`chat4b` dışındaki paged birimler CPU→GPU katman akışı kullanır; `chat4b` ise
int8 ağırlıkları GPU'da tutup yalnızca tek bir bf16 katman slotunu yeniden
kullanır. Bu yüzden 4B, 3B'den daha hızlı olabilir.

## Kalite: ölçülen ve ölçülmeyen

Aşağıdaki sayılar `results/results.jsonl` içindeki deneylerden gelir. Hepsi
**küçük, anahtar-kelime tabanlı** setlerdir; insan hakemliği veya genel benchmark
değildir.

| Set | Model | Keyword | Exact | Ham liste formatı | Ort. üretim |
|---|---|---|---|---|---|
| 60 prob (`exp42`) | Qwen2.5-3B-Instruct | %88.5 | %92.3 | %12.5 | 2.28 s |
| 60 prob (`exp42`) | Qwen3-4B-Instruct-2507 | %92.3 | %84.6 | %12.5 | 5.65 s (paged) |
| 60 prob (`exp42 resident`) | Qwen3-4B-Instruct-2507 | %92.3 | %84.6 | %12.5 | **3.21 s / 2.72 tok/s** |
| 35 holdout (`exp42`) | Qwen2.5-3B-Instruct | %86.7 | %90.0 | %20.0 | 2.52 s |
| 35 holdout (`exp42`) | Qwen3-4B-Instruct-2507 | %83.3 | %90.0 | %40.0 | 7.06 s (paged) |
| 35 holdout (`exp42 resident`) | Qwen3-4B-Instruct-2507 | %83.3 | %90.0 | %40.0 | **3.96 s / 2.71 tok/s** |
| **60 prob ürün `quality`** (`exp43`) | 4B + 1.5B sosyal + liste düzeltme | **%94.2** | %84.6 | %87.5 | **3.23 s / 2.99 tok/s** |
| **35 holdout ürün `quality`** (`exp43`) | 4B + 1.5B sosyal + liste düzeltme | %83.3 | %90.0 | %100 | **3.97 s / 3.17 tok/s** |

`exp43` gerçek ürün yolunu ölçer: `Service.answer(..., mode="quality")`. 60
istekte rota **59× chat4b, 1× chat**, holdout'ta **34× chat4b, 1× chat**; peak
**4526 MB**. Liste skoru 60'ta %87.5 çıktı çünkü bir liste istemi 64 token
bütçesinde kesildi; holdout'ta beş liste de üç satıra düzgün çevrildi. CLI
varsayılanı 160 token olduğu için bu kesilme üründe daha seyrek görülür, yine de
garanti değildir.

**Int8 resident vs paged ablasyonu (`exp42 resident`):** Aynı 95 probda 4B'nin
36/36 katman int8 resident hali, 19 katmanı bf16 bırakan paged haline göre
keyword/exact/liste metriklerinde **birebir aynı** sonucu verdi (60:
%92.3/%84.6/%12.5; holdout: %83.3/%90.0/%40.0). Ortalama üretim 5.65 → **3.21 s**
(60) ve 7.06 → **3.96 s** (holdout); peak 4518 MB. Bu, bu 95 probda kuantizasyon
kaybı görülmediğini gösterir; genel kalite garantisi değildir. Bu koşularda
int8 cache kullanıldı (load 2.36/2.39 s).

Yorum:

- 60 + 35 holdout keyword oranlarında iki model **başa baş** (4B: %92.3 / %83.3;
  3B: %88.5 / %86.7). Ürün yolu da benzer: 4B `quality` 60'ta %94.2, holdout'ta
  %83.3 (ortalama ≈%88.8); 3B `quality` (exp40, offline) 60'ta %91.7, holdout'ta
  %85.7 (ortalama ≈%88.7). Fark ölçüm gürültüsü içinde. 4B Türkçe olgularda daha
  iyi (altın → Au, en büyük organ → deri, kalp → dört odacık), bazı
  talimat/çeviri işlerinde daha zayıf (su 212°F, “Köyde köpek var”).
- Ham liste formatı düşük görünür çünkü modeller listeyi satır içi veriyor.
  Ürünün `quality` yolu bunu deterministik olarak üç satıra çevirir; bu yüzden
  ham liste skoru ürün skoru değildir.
- 3B'nin bilinen hataları: Mısır → İskoçya/Kairo, en büyük çöl → Coca-Cola,
  kalpte 20–25 odacık, altın → O. 4B bunların bir kısmını düzeltir; Mısır'ı yine
  “Kairo” yazar (Kahire değil) ve Türkçe “en büyük okyanus” sorusunda Hint
  Okyanusu der. Yani **4B doğru cevap garantisi değildir**.
- 4B'nin ürün hızı, yukarıdaki paged ölçümden yüksektir: int8 GPU-resident
  modunda kısa ölçümde **~3.4 tok/s**, 60 prob uçtan uca ortalamasında
  **2.99 tok/s**; 60 probdaki 5.65 s, CPU paging ile alınmıştır. 3B paged'in
  aynı setteki 2.73 tok/s değerine göre decode hızı daha yüksektir. 4B daha
  uzun cevap verdiği için ortalama duvar süresi 3.23 s (3B: 2.28 s) olur;
  "daha hızlı" iddiası token/s için, duvar süresi için değil.

## Dürüst sınırlar

- **Bilgi doğruluğu:** 4B, eğitim verisinden bildiği kadarını söyler; güncel
  olayları, nadir olguları ve bazı Türkçe adları yanlış verebilir. Web araması ve
  araç kullanımı yoktur; yalnızca `data/rag/world_facts.jsonl` içindeki **326
  sayfalık** yerel kaynaklar kullanılabilir. Bu korpus sınırlıdır ve genel bilgi
  benchmark'ı değildir; `exp48` yalnızca 50 kapsanan olgu + 20 kapı kontrolü
  ölçer. Kaynak sıralaması bu sette 50/50 tek kaynak seçti; yine de kapsam
  dışında veya çok benzer sayfalarda hata yapabilir; atıf biçimi tek başına
  cevabın doğruluğunu kanıtlamaz.
- **Hafıza:** konuşma geçmişinin son 8 mesajı tutulur. 4B ile kısa probe:
  ad, tercih, önceki konu ve düzeltme **4/4** doğru; ilk olgu 8 mesajlık
  pencerenin dışına çıkarıldığında benzersiz sayı **0/1** hatırlandı. Bu,
  “son 8 mesaj” davranışını doğrular; genel hafıza iddiası değildir.
- **Hız:** ~3.4 tok/s, 6 GB laptop GPU'su için dürüst bir sayıdır; llama.cpp
  Q4 gibi bir çözümle yarışmaz. İnt8 önbelleği yoksa ilk yükleme ~18.8 s'dir;
  önbellek sonrası ölçülen süreç başlangıcı ~2.4 s'dir (soğuk disk okuması ayrı
  ölçülmedi).
- **VRAM:** model-only 4513 MB peak; RAG değerlendirmesinde 4558,7 MB, ürün
  RAG smoke'unda 4552 MB. Başka GPU uygulamaları açıkken OOM olabilir; model
  yüklüyken tarayıcı/oyun gibi VRAM kullanan uygulamaları kapatın.
- **Dil:** Üründe Türkçe ve İngilizce test edildi; Türkçe olgularda İngilizceye
  göre daha fazla hata görülür. Diğer diller modelin kendi yeteneğine bağlıdır,
  bu üründe ölçülmedi.
- **API değildir:** `serve.py` yalnızca geliştirici demosudur; auth/CORS/rate
  limit ürün hedefi değildir.

## Neyi kanıtlar, neyi kanıtlamaz?

Kanıtladığı:

- 4B bf16 6 GB'a sığmazken, int8 GPU-resident + on-the-fly dequant ile
  **6 GB'de çalıştığını** (peak 4513 MB, ~3.4 tok/s);
- 95 probda 4B resident-int8'in paged 4B ile aynı keyword/exact/liste
  metriklerini verdiğini (aynı ham çıktı protokolü);
- aynı servis üzerinden 1.5B/3B/4B modellerinin ölçülebilir biçimde
  karşılaştırılabildiğini;
- `quality` rotasının ve liste düzeltmesinin uçtan uca çalıştığını;
- dar `exp48` setinde yerel RAG'in 50 kapsanan olguda keyword isabetini
  **42/50 → 49/50**'ye, yeni 30 olguda **24/30 → 30/30**'a çıkardığını,
  50/50 cevabın manuel olarak doğru olduğunu ve 20 kontrolün hiçbirinde RAG'i
  tetiklemediğini.

Kanıtlamadığı:

- 4B'nin 3B'den **genel olarak** daha kaliteli olduğunu (95 probda başa baş);
- insan düzeyinde doğruluk veya benchmark liderliği;
- RAG'in genel bilgi araması olduğunu; 326 sayfa dışında kapsam yoktur;
- kaynak seçiminin her soruda temiz olduğunu (bu sette 50/50 tek kaynak;
  kapsam dışı/genel sorular için garanti değil);
- çok turlu güvenilir hafıza;
- llama.cpp/Ollama gibi optimize runtime'larla hız yarışı.

## Yeniden üretim

```powershell
python experiments\exp42_chat4b_quality.py --set exp26 --tag exp42_60
python experiments\exp42_chat4b_quality.py --set holdout --tag exp42_holdout
python experiments\exp42_chat4b_quality.py --units chat4b --set exp26 --tag exp42_resident60
python experiments\exp42_chat4b_quality.py --units chat4b --set holdout --tag exp42_resident_holdout
python experiments\exp43_quality4b_e2e.py --set exp26 --tag exp43_60
python experiments\exp43_quality4b_e2e.py --set holdout --tag exp43_holdout
python experiments\exp44_system_prompt_4b.py --tag exp44
python experiments\exp46_multiturn_4b.py
python scripts\smoke_qwen3_4b.py --resident-quant
python scripts\smoke_qwen3_4b.py --resident-quant --quant-cache cache\qwen3_4b_instruct_2507_int8_resident.pt
python scripts\check_rag_facts.py                  # CPU retrieval/kaynak tanisi
python experiments\exp48_rag_eval.py               # 50 kapsanan + 20 kontrol, GPU
python -m unittest discover -s tests -p "test_*service*.py" -v   # 53
python -m unittest tests.test_paged_qwen tests.test_paged_quant_cache -v  # 5
python -m unittest tests.test_rag -v               # 21
# toplam 79 test
```

Ayrıntılı araştırma raporları: `README.md`, `SERVICE_RAPOR.md`, `RAPOR.md`,
`PROGRESS.md`, `results/results.jsonl`.
