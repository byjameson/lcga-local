# exp47 — manuel/agent cevap incelemesi

Bu dosya `results/exp47_rag_eval.json` içindeki 60 ham cevabın, özellikle
kapsanan 20 sorunun, insan tarafından değil **agent tarafından** okunmasının
kaydıdır. Otomatik kelime ve exact skorlarından ayrıdır; genel bir insan
hakemliği veya benchmark değildir.

## Kapsanan sorular (20)

| # | id | Baseline | RAG | Not |
|---|---|---|---|---|
| 1 | tr-eg | yanlış: “Kairo” | doğru: Kahire | baseline Mısır adını İngilizceleştirdi |
| 2 | tr-tr | doğru: Ankara | doğru: Ankara | |
| 3 | tr-fr | doğru: Paris | doğru: Paris | |
| 4 | tr-jp | doğru: Tokyo | doğru: Tokyo | |
| 5 | tr-au | doğru: Canberra | doğru: Canberra | |
| 6 | tr-de | doğru: Berlin | doğru: Berlin | |
| 7 | tr-ca | doğru: Ottawa | doğru: Ottawa | |
| 8 | tr-it | doğru: Roma | doğru: Roma | |
| 9 | tr-ocean | yanlış: Hint Okyanusu | doğru: Büyük/Pasifik Okyanusu | baseline “en büyük”ü karıştırdı |
| 10 | tr-au-sym | doğru: Au | doğru: Au | |
| 11 | tr-water | doğru: 100 °C | doğru: 100 °C | RAG kaynaklarından biri alakasız |
| 12 | tr-heart | doğru: 4 odacık | doğru: dört odacık | RAG'de ikinci kaynak Köpek |
| 13 | en-eg | doğru: Cairo | doğru: Cairo | |
| 14 | en-au | doğru: Canberra | doğru: Canberra | |
| 15 | en-ca | doğru: Ottawa | doğru: Ottawa | |
| 16 | en-gold | doğru: Au | doğru: Au | |
| 17 | en-water | doğru: 100 °C | doğru: 100 °C | |
| 18 | en-heart | doğru: dört | doğru: dört | |
| 19 | en-ocean | doğru: Pasifik | doğru: Pasifik | |
| 20 | en-planet | doğru: Jüpiter | doğru: Jüpiter | RAG'de Mercury ikinci kaynak |

**Manuel sonuç: baseline 18/20, RAG 20/20.** Baseline'ın iki hatası `tr-eg`
ve `tr-ocean`. RAG bu iki olguyu düzeltti; bu 20 olgu dışında genel bir RAG
kazancı iddiası yoktur.

## Otomatik metriklerle ilişki

- Düzeltilmiş keyword: baseline **18/20**, RAG **20/20**. Eski alt-dize kontrolü
  `tr-ocean` baseline'ını yanlışlıkla geçiriyordu; artık `pasifik` kelime sınırı
  ile aranıyor. `au` da “because” gibi kelimelerin içinde sayılmıyor.
- Strict exact: baseline **1/15**, RAG **0/15**. RAG cevapları cümle ve `[1]`
  atıfı eklediği için bu metrik uygun değildir; yukarıdaki manuel okuma esas
  alınmalıdır.
- RAG atıfı kapsananlarda **20/20**, RAG kullanımı **20/20**. Kontrollerde RAG
  **0/10**, atıf **0/10**; kontrol baseline/RAG çıktıları aynıdır. Bu yalnızca
  kapının çalıştığını gösterir; kontrol cevaplarının semantiği değerlendirilmedi.

## Kaynak gürültüsü

RAG cevabı doğru olsa da ikinci kaynak bazen alakasız olabiliyor:

- `tr-water`: Kurşun, Su — Su pasajı 100 °C'yi verdi, Kurşun pasajı bağlamı
  kirletti.
- `tr-heart`: Kalp, Köpek — insan kalbi pasajı doğru cevabı verdi.
- `tr-au-sym`: Altın, Altın oran — sembol doğru.
- `en-gold`: Gold, Potassium — altın pasajı doğru.
- `en-planet`: Mercury (planet), Jupiter — Jüpiter pasajı doğru.

Bu nedenle kaynak başlığı/atıf biçimi tek başına doğruluk kanıtı değildir.

## Sınırlar

- Set yalnızca **20 kapsanan olgu + 10 kontrol**; genel bilgi benchmark'ı değil.
- `max_new=64`; uzun kontrol cevapları kesilebilir. Kontrollerde bazı yaratıcı
  cevaplar bu bütçede yarım kaldı; bu RAG kapısının doğruluğunu etkilemez.
- Ölçülen süreler tek koşuya aittir: kapsanan ortalama **4,99/5,26 s**,
  üretim **4,86/5,01 s**; kontrol ortalama **10,46/10,45 s** (uzun/64-token
  cevaplar). Peak **4517,6/4558,7 MB**.
- Ham çıktılar `results/exp47_rag_eval.json`; okunabilir döküm
  `results/exp47_review.txt`; retrieval tanısı
  `results/exp47_retrieval_check.txt`.
