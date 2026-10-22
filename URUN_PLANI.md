# LCGA Local - urun degerlendirmesi

Tarih: 2026-10-20 (asagidaki 20.10 bolumu tarihsel)

## 2026-10-21 guncellemesi - urun yuzeyi degisti

- **Urun artik API/web degil:** tek komutla acilan yerel sohbet.
  `python llm.py` veya Windows'ta `sohbet.cmd`. Sunucu, auth, CORS, rate limit
  urun hedefi degil; `serve.py` yalnizca opsiyonel gelistirici demosu.
- **Model:** Qwen3-4B-Instruct-2507, 36 katman int8 GPU-resident + tek bf16
  dequant slotu. Olculen: **4513 MB peak, ~3.4 tok/s decode, 18.8 s ilk
  yukleme; int8 cache sonrasi olculen surec baslangici 2.4 s**. 4B bf16 ~8 GB
  oldugu icin bu, "6 GB'de 4B" iddiasinin ana kaniti.
- **Varsayılan mod `quality`:** genel/kod/liste -> `chat4b`; kisa nezaket ->
  1.5B `chat`; 4B yoksa `chat3b`, o da yoksa `chat`. 3B/1.5B yalnizca yedek
  veya acik secim.
- **Yerel RAG:** `llm.py` varsayilan olarak CPU-only BM25 ile
  `data/rag/world_facts.jsonl` (**326 dokuman / 4.793.376 bayt / 4,57 MiB**)
  uzerinden kesin olgularda tek, genis sorularda en fazla iki kaynak pasaji
  ekler; kaynak yalnizca yeterli kelime ortusmesi varsa devreye girer, cevap
  `[1]` ile isaretlenir. Indeks yukleme **1,55-2,14 s**; runtime'da internet
  yok; `--no-rag` ile kapatilir. `exp48` (50 kapsanan + 20 kontrol,
  baseline/RAG): retrieval **50/50**, tek kaynak **50/50**; keyword
  **42/50 -> 49/50** (yeni 30'da **24/30 -> 30/30**), manuel cevap **50/50**,
  atif **50/50**; kontrollerde RAG/atif **0/20**. Genel benchmark degil;
  kapsam disi/benzer sayfalarda kaynak hatasi olabilir.
- **Kalite (durust):** 60 prob urun yolu keyword **%94.2**, exact %84.6,
  liste-format %87.5, 3.23 s / 2.99 tok/s; 35 holdout **%83.3 / %90.0 / %100**,
  3.97 s / 3.17 tok/s (exp43). Ham 3B/4B karsilastirmasinda 60 + 35 holdout
  keyword oranlarinda iki model basa bas; 4B Turkce olgularda daha iyi, bazi
  talimat/cevirilerde daha zayif. Genel "4B daha kaliteli" iddiasi kurulmuyor.
- **Sinirlar:** `--no-rag` yolunda Mısır -> "Kairo" (Kahire degil), Turkce
  "en buyuk okyanus" -> Hint Okyanusu, su 212F gibi talimat hatalari; RAG bu
  dar `exp48` setinde 50/50 cevabi olgusal olarak duzeltti, genel garanti
  degil. Web aramasi yok; yerel RAG korpusu **326 sayfa** ile sinirli; kapsam
  disi/benzer sayfalarda kaynak hatasi olabilir; son 8 mesaj hafiza; ilk 4B
  yuklemesi 18.8 s (cache sonrasi 2.4 s). Testler **79**. Ayrinti: `URUN.md`.
- **Eski 20.10 degerlendirmesi** (API/orchestration odakli) asagida tarihsel
  olarak duruyor; urun yuzeyi ve varsayilan model degisti.

## Kisa hukum (20.10, tarihsel)

Bu proje **yeni bir model** degil. Degerli olabilecek sey, 6GB VRAM gibi
sert bir sinirda farkli boyutlardaki modelleri tek bir OpenAI uyumlu API
arkasinda, sorguya gore secip, VRAM'e sigmayani int8 paged calistiran
**yurutme/orchestration katmani**.

Su haliyle durust tanim: **teknik demo ve gelistirici araci**. "Piyasada
yenilikci urun" iddiasi icin asagidaki farklarin olculmesi ve rakamlarla
gosterilmesi gerekir.

## Neden ilgi cekebilir?

- 6GB laptop GPU'larda 3B modeller tam sigmaz; kullanicilar genelde tek bir
  kucuk modele mahkum olur.
- Mevcut yerel araclar (Ollama, LM Studio, llama.cpp) model calistirmaya
  odakli. Otomatik **kalite/hiz/VRAM butcesi** yonlendirmesi ve ayni endpoint
  altinda heterojen unit'ler daha az yaygin.
- OpenAI uyumlu olmasi, mevcut istemcilerin kodu degistirmeden denemesini
  saglar.
- Paged 3B, "6GB'de 3B kalite modu" gibi somut bir vaat uretir; ancak hiz
  bedeli acikca olculmelidir.

## Neden "yenilik" demek icin yeterli degil?

- Paged/offload inference yeni degil; llama.cpp ve benzeri projelerde var.
- Model routing de yeni degil; arastirma ve bazi urunlerde var.
- Kural tabanli router (keyword/uzunluk) bir urun farki degil; kolayca
  kopyalanir.
- 1.5B/3B kalite farki 60+35 probla sinirli; insan/LLM judge olmadan
  genel kalite iddiasi kurulamaz.
- Tek laptop, tek GPU, tek kullanici senaryosu ölçek iddiasini desteklemez.
- Memory template'i genel hafiza degil; sadece bes kalipli bir yardimci.

## Ar-Ge bulgulari piyasada ilgi gorur mu?

| Bulgu | Teknik deger | Pazar/ilgi durumu |
|---|---|---|
| LCGA: ogrenilmis blok-seyrek dikkat, 0.5B | Uzun baglamda %25 blokla ~1.25x hiz @16k; exact sparse kernel; CUDA graph decode | Arastirma ilgisi olabilir; **urun degil**. Kalite farki buyuk (PPL 16.56 vs 12.46), hiz avantaji ancak ~5k token sonrasi. Longformer/BigBird/Native Sparse gibi kalabalik bir alan; standart benchmark ve daha iyi kalite olmadan paper bile zor. |
| 3B full-parameter block-streaming egitim, 6GB | 3.09B parametre, 6.64 s/adim, peak 2.43GB; Adafactor ile CE 2.50 -> 2.32 | **Entegre/egitim meraklisi** icin ilgi cekici; ticari uretim icin yavas. QLoRA ile kiyaslandiginda pratik degil. Dusuk kaynakli laboratuvarlar/blog/HF demosu icin niche. |
| 3B int8 paged inference, 6GB | 4.18 tok/s; paged/dense CE parity | llama.cpp Q4 3B zaten 6GB'e sigiyor ve daha hizli (57.6 tok/s). Paging'in degeri **>VRAM modeller** ve egitimde; 3B tek basina urun farki degil. |
| MoE expert paging + transition prefetch | Sentetik testte 12.4x daha az demand copy, %92 hit | Arastirma olarak ilgi cekici; gercek buyuk MoE ve gercek workload ile dogrulanmadi. Urun icin erken. |
| Multi-unit scheduler + paged/dense gecis | Paralel 1.30x, LRU pool, VRAM planlama | **En uygulanabilir urun parcasi.** Tek basina moat degil; iyi bir orchestration katmani ve olcumle birlestirilirse degerli. |
| 3B Instruct paged kalite modu | **Ölçüldü:** 60 probda keyword %88,5 (chat %82,7), exact %92,3; holdout 35'te %86,7 (chat %73,3); strict liste varsayılan olarak 1.5B daha iyi; Türkçe numaralı talimatla 3B liste formatı düzeliyor; üretim ~3,0-3,5 s, peak ~4,4-4,8GB | Kullanıcıya doğrudan görünen tek bulgu. Kalite artışı var ama **tek başına ürün farkı değil**; yavaş. `quality` yalnızca açık seçimde. |

### Durust sonuc

- **Arastirma toplulugu** LCGA, full-parameter streaming ve MoE paging'i ilgi cekici bulabilir; ancak bunlar birbirinden bagimsiz, kucuk olcekli ve standart benchmark'lardan uzak.
- **Yatirimci/ana akim pazar** bu haliyle ilgilenmez; "6GB'de 3B" tek basina yeni degil.
- **Gelistirici/yerel AI toplulugu** icin en anlamli paket: bu parcalari tek bir **butce farkindalikli, OpenAI uyumlu yerel sunucuda** birlestirmek.
- Ilgi gormek icin gereken: disaridan tekrarlanabilir kurulum, 60+ prompt ve gercek kullanici senaryosu, llama.cpp/Ollama ile durust kiyas, hata ornekleri, model kartlari.

### 20.10 olcum guncellemesi (routing)

- **Hibrit `quality` olculdu:** TR liste -> 3B + numarali talimat; EN liste
  -> 1.5B + deterministik format duzeltme; sosyal -> 1.5B; diger -> 3B.
  60 probda **%91,7**, ortalama **2,87 s**, chat 5/chat3b 55; 35 problu
  holdout'ta **%85,7**, **3,60 s**, chat 3/chat3b 32. Bu, ayni satirlar uzerinde
  **offline politika simulasyonu**; yukleme/gecis haric, yeni uctan uca benchmark
  degil. Liste skoru **format-only**; manuel incelemede 12 cevabin 11'i dogru,
  1'i kismi (zeytin). Holdout hata analizinde kullanildigi icin **tanisal**,
  temiz genelleme degil. Onceki kural %88,3/%82,9 idi; 60'taki artisin bir kismi
  `maddede` cue duzeltmesi (86,7->88,3).
- **Format tuzagi (`exp39`):** 3B + numarali talimat 11 liste probunda format
  **11/11**, fakat Ingilizce "primary colors" isteminde tekrarli/bozuk metin
  uretti; bu yuzden yalnizca Turkce `listele` 3B'ye gidiyor. Format-only skor
  tek basina kalite kaniti degil.
- **Ogrenilmis router denemesi:** `exp36` hashed n-gram + lojistik regresyon
  holdout'ta **%82,9 / 3,03 s** aldi; eski kural router'la ayni kalite. Entegre
  edilmedi. **Qwen disi adaptor:** `exp37`/`alt` SmolLM2-1.7B'yi ayni servis
  uzerinden calistirir; holdout keyword %50, exact %20, liste-format %100 ama
  icerik hatali. Kalite iddiasi degil, tasinabilirlik kaniti; `auto`/`quality`
  bu birimi secmez.
- **Confidence cascade negatif:** `exp31`'de chat guveni dusukken 3B'ye
  yukseltme, 60 probda always-chat'i gecemedi; test yarisinda esik hic
  tetiklenmedi. Bu yuzden entegre edilmedi. Bu, "confidence ile otomatik
  yukseltme" fikrinin bu veriyle desteklenmedigini gosterir.
- **Eski `auto` zayif:** 24 probda keyword %81,8, exact %100 ama strict liste
  %0; 23/24 sorgu chat'e gidiyor. `auto` hiz icin makul, kalite icin degil.
- **Pazar yorumu:** Bu sonuclar "6GB'de routing" fikrinin **teknik olarak
  gosterilebilir** oldugunu kanitliyor; ancak fark tasarim setinde +13,4 puan ve
  ~1,6x uretim bedeliyle geliyor (yukleme/gecis haric). Gelistirici/lab demosu icin ilgi cekici; ana akim urun
  veya yatirimci iddiasi icin hala yetersiz. Bir sonraki esik, ayni sette
  insan/LLM judge ve 100+ bagimsiz prompt.



## En guclu urun konumu

**"6GB VRAM icin OpenAI uyumlu, butce farkindalikli yerel model sunucusu."**

Kullanici sunu secer:

- `budget_gb`: kullanilabilir VRAM
- `quality`: `fast` / `balanced` / `chat` / `chat3b` / `quality`
- `max_latency` veya `max_new_tokens`

Sistem de:

1. Sorguyu ve istenen modu siniflandirir.
2. Butceye sigan en kucuk modelle baslar; secili modda gorev ipucuna gore
   1.5B/3B secer (confidence cascade **degil**; `exp31` negatif).
3. 3B'yi int8 paged calistirir; iki paged motor ayni anda yuklenmez.
4. Cevabi, kullanilan unit'leri, latency'yi ve peak VRAM'i dondurur.

## Ilk surum icin minimum kanit

- **Yapildi:** ayni 60 soruda `chat` vs `chat3b` kalite + latency + peak
  (keyword %82,7 → %88,5; 1,114 → 3,025 s; 3248 → 4411MB).
- **Yapildi:** bagimsiz 35 problu holdout (keyword %73,3 → %86,7; strict liste
  %80 → %20); eski `quality` router 60'ta %88,3, holdout'ta %82,9 idi. Hibrit
  listeyle (`exp40`) 60'ta **%91,7 / 2,87 s**, holdout'ta **%85,7 / 3,60 s**
  (offline satir simulasyonu; yukleme haric). Liste skoru format-only; 12
  cevabin 11'i manuel olarak dogru, 1'i kismi. Holdout hata analizinde
  kullanildi, tanisal. `exp36` ogrenilmis router esit kaldi, entegre edilmedi.
- **Yapildi:** `always-chat` / `always-chat3b` / oracle ve cascade kiyasi;
  cascade negatif, entegre edilmedi. `exp39` 3B numarali talimatin format
  kazanci ile semantik riskini ayirdi (Ingilizce primary colors basarisiz).
- **Yapildi:** `exp37` Qwen disi SmolLM2 adaptor testi; `alt` birimi ayni
  Service/OpenAI ucundan acilir, varsayilan rotaya girmez. Holdout keyword %50,
  exact %20; kalite iddiasi degil.
- **Yapildi:** cold/warm TTFT, tok/s, model degistirme ve canli hibrit
  gecisleri (`exp41`, soguk: TR liste 11,98 s, EN liste 6,40 s, fact 3,67 s;
  isinmis stream EN liste 0,91 s); hata ornekleri ve sinirlar belgelendi.
- **Eksik:** insan/LLM judge, 100+ bagimsiz prompt, llama.cpp/Ollama ile ayni
  donanimda durust kiyas, tek komut kurulum ve gercek kullanici demosu.

## Yayin stratejisi

1. **Acik kaynak gelistirici araci**: GitHub + Hugging Face model/benchmark
   kartlari + tek komut demo.
2. **Teknik yazi**: "6GB'de butce farkindalikli yerel model yonlendirme" -
   gercek ölçümler ve basarisiz örneklerle.
3. **Paper degil**: mevcut kanit seviyesi hakemli yayin icin yetersiz. Paper
   icin coklu GPU, daha buyuk benchmark, alternatif sistemlerle karsilastirma
   ve daha guclu bir router gerekir.
4. **Urun kanali**: Product Hunt / Hacker News / yerel AI topluluklari ancak
   kurulum gercekten tek komut ve demo stabil oldugunda.

## Sonraki adimlar

- [x] `chat3b` indirmesini bitir, tek soruda latency/VRAM dogrula.
- [x] exp30: 1.5B chat vs 3B paged, 24 → 60 soru.
- [x] exp31: `auto` vs `always-chat` vs `always-chat3b` ve cascade kiyasi.
- [x] exp33/exp34: bagimsiz 35 promptta `quality` router dogrulamasi.
- [x] exp36: ogrenilmis router denemesi (esit, entegre edilmedi).
- [x] exp37/alt: Qwen disi dense model adaptor testi.
- [x] README'ye sinirlar, model/API ve olcum bolumu ekle.
- [ ] Insan/LLM judge ve 100+ bagimsiz prompt; hata analizi.
- [ ] Tek komut kurulum/indirme akisini netlestir.
- [ ] llama.cpp/Ollama ile ayni donanimda kalite/hiz kiyasi.
- [ ] Gercek kullanici demo senaryosu: 3-5 dakikalik, hatasiz akis.

## Durust vaat

**Simdi:** 6GB'de calisan, OpenAI uyumlu, **bes unit'li** (fast, balanced,
deep, chat, chat3b), paged 3B'yi ve olculmus `quality` router'i deneyebilen bir
yerel prototip.

**Hedef:** Ayni donanimda kalite/hiz/VRAM dengesini otomatik yoneten,
ölçülebilir ve kurulumu kolay bir gelistirici araci.

**Degil:** 30B'yi 6GB'de calistirmak, genel amaçli hafiza, ChatGPT kalitesi
veya kanitlanmis bir ticari basari.
