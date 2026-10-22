# exp48 raw answer review

> **Review notu (22.10.2026):** 50/50 RAG cevabı olgusal olarak doğru okundu;
> 50/50 tek kaynak ve atıf [1] var. Keyword 49/50; tek kaçak `tr-ocean`
> (doğru Türkçe ad “Büyük Okyanus”, beklenen literal “Pasifik”). Baseline
> keyword 42/50; yeni 30 soruda 24/30 → 30/30. 20/20 kontrolde RAG/atıf yok.
> Strict exact 0/36 anlamlı değil (atıf ekleniyor). Bu, bağımsız insan
> değerlendirmesi değil, bu oturumdaki manuel/agent okumasıdır.

covered=50 controls=20 max_new=64

## summaries

```json
[
  {
    "set": "covered",
    "mode": "baseline",
    "ids": "all",
    "n": 50,
    "keyword_hit": "42/50",
    "exact_hit": "4/36",
    "list_hit": "n/a",
    "citation": "0/50",
    "rag_used": "0/50",
    "avg_elapsed_s": 5.82,
    "avg_gen_s": 5.74,
    "peak_vram_mb_max": 4517.8
  },
  {
    "set": "covered",
    "mode": "rag",
    "ids": "all",
    "n": 50,
    "keyword_hit": "49/50",
    "exact_hit": "0/36",
    "list_hit": "n/a",
    "citation": "50/50",
    "rag_used": "50/50",
    "avg_elapsed_s": 5.85,
    "avg_gen_s": 5.33,
    "peak_vram_mb_max": 4547.7
  },
  {
    "set": "uncovered",
    "mode": "baseline",
    "ids": "all",
    "n": 20,
    "keyword_hit": "n/a",
    "exact_hit": "n/a",
    "list_hit": "2/2",
    "citation": "0/20",
    "rag_used": "0/20",
    "avg_elapsed_s": 13.88,
    "avg_gen_s": 13.74,
    "peak_vram_mb_max": 4518.1
  },
  {
    "set": "uncovered",
    "mode": "rag",
    "ids": "all",
    "n": 20,
    "keyword_hit": "n/a",
    "exact_hit": "n/a",
    "list_hit": "2/2",
    "citation": "0/20",
    "rag_used": "0/20",
    "avg_elapsed_s": 13.75,
    "avg_gen_s": 13.61,
    "peak_vram_mb_max": 4518.1
  },
  {
    "set": "covered",
    "mode": "baseline",
    "ids": "new",
    "n": 30,
    "keyword_hit": "24/30",
    "exact_hit": "3/21",
    "list_hit": "n/a",
    "citation": "0/30",
    "rag_used": "0/30",
    "avg_elapsed_s": 6.43,
    "avg_gen_s": 6.34,
    "peak_vram_mb_max": 4517.8
  },
  {
    "set": "covered",
    "mode": "rag",
    "ids": "new",
    "n": 30,
    "keyword_hit": "30/30",
    "exact_hit": "0/21",
    "list_hit": "n/a",
    "citation": "30/30",
    "rag_used": "30/30",
    "avg_elapsed_s": 6.28,
    "avg_gen_s": 5.71,
    "peak_vram_mb_max": 4547.7
  }
]
```

## covered pairs


### tr-eg (tr)

**Q:** Mısır'ın başkenti neresi?

**baseline**: kw=False exact=False cit=False | Mısır'ın başkenti **Kairo**'dur.

**rag**: kw=True exact=False cit=True used=True sources=['Kahire'] | Mısır'ın başkenti Kahire'dir [1].


### tr-tr (tr)

**Q:** Türkiye'nin başkenti neresi?

**baseline**: kw=True exact=False cit=False | Türkiye'nin başkenti Ankara'dır.

**rag**: kw=True exact=False cit=True used=True sources=['Türkiye'] | Türkiye'nin başkenti Ankara'dır [1].


### tr-fr (tr)

**Q:** Fransa'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Fransa'nın başkenti Paris'tir.

**rag**: kw=True exact=False cit=True used=True sources=['Paris'] | Fransa'nın başkenti Paris'tir [1].


### tr-jp (tr)

**Q:** Japonya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Japonya'nın başkenti **Tokyo**'dur.

**rag**: kw=True exact=False cit=True used=True sources=['Tokyo'] | Japonya'nın başkenti Tokyo'dur [1].


### tr-au (tr)

**Q:** Avustralya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Avustralya'nın başkenti **Canberra**'dır.

**rag**: kw=True exact=False cit=True used=True sources=['Avustralya'] | Avustralya'nın başkenti Canberra'dır. [1]


### tr-de (tr)

**Q:** Almanya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Almanya'nın başkenti Berlin'dir.

**rag**: kw=True exact=False cit=True used=True sources=['Berlin'] | Almanya'nın başkenti Berlin'dir [1].


### tr-ca (tr)

**Q:** Kanada'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Kanada'nın başkenti **Ottawa**'dır.

**rag**: kw=True exact=False cit=True used=True sources=['Kanada'] | Kanada'nın başkenti Ottawa'dır [1].


### tr-it (tr)

**Q:** İtalya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | İtalya'nın başkenti **Roma**'dır.

**rag**: kw=True exact=False cit=True used=True sources=['İtalya'] | İtalya'nın başkenti Roma'dır [1].


### tr-ocean (tr)

**Q:** Dünyanın en büyük okyanusu hangisi?

**baseline**: kw=False exact=None cit=False | Dünyanın en büyük okyanusu **Hint Okyanusu**'dur.

**rag**: kw=False exact=None cit=True used=True sources=['Büyük Okyanus'] | Dünyanın en büyük okyanusu Büyük Okyanustur [1].


### tr-au-sym (tr)

**Q:** Altının kimyasal sembolü nedir?

**baseline**: kw=True exact=False cit=False | Altının kimyasal sembolü **Au**'dur.

**rag**: kw=True exact=False cit=True used=True sources=['Altın'] | Altının kimyasal sembolü Au'dur [1].


### tr-water (tr)

**Q:** Su kaç derecede kaynar?

**baseline**: kw=True exact=None cit=False | Su, 100 derece Celsius'te (1 atm basıncında) kaynar.

**rag**: kw=True exact=None cit=True used=True sources=['Su'] | Su, 100 °C'de kaynar. [1]


### tr-heart (tr)

**Q:** İnsan kalbinde kaç odacık vardır?

**baseline**: kw=True exact=None cit=False | İnsan kalbinde **4 odacık** vardır:  
- Sağ atrium (sağ kalp odacığı)  
- Sol atrium (sol kalp odacığı)  
- Sağ ventrikül (sağ kalp odacığı)  
- Sol ventrikül (sol

**rag**: kw=True exact=None cit=True used=True sources=['Kalp'] | İnsan kalbinde dört odacık vardır: iki üst atriyum ve iki alt ventrikül [1].


### en-eg (en)

**Q:** What is the capital of Egypt?

**baseline**: kw=True exact=False cit=False | The capital of Egypt is Cairo.

**rag**: kw=True exact=False cit=True used=True sources=['Cairo'] | The capital of Egypt is Cairo [1].


### en-au (en)

**Q:** What is the capital of Australia?

**baseline**: kw=True exact=False cit=False | The capital of Australia is Canberra.

**rag**: kw=True exact=False cit=True used=True sources=['Canberra'] | The capital of Australia is Canberra [1].


### en-ca (en)

**Q:** What is the capital of Canada?

**baseline**: kw=True exact=False cit=False | The capital of Canada is Ottawa.

**rag**: kw=True exact=False cit=True used=True sources=['Ottawa'] | The capital of Canada is Ottawa [1].


### en-gold (en)

**Q:** What is the chemical symbol for gold?

**baseline**: kw=True exact=True cit=False | Au

**rag**: kw=True exact=False cit=True used=True sources=['Gold'] | The chemical symbol for gold is Au [1].


### en-water (en)

**Q:** At what temperature does water boil?

**baseline**: kw=True exact=None cit=False | Water boils at 100°C (212°F) at sea level.

**rag**: kw=True exact=None cit=True used=True sources=['Boiling point'] | Water boils at 100 °C at standard atmospheric pressure [1].


### en-heart (en)

**Q:** How many chambers does the human heart have?

**baseline**: kw=True exact=None cit=False | The human heart has four chambers.

**rag**: kw=True exact=None cit=True used=True sources=['Heart'] | The human heart has four chambers. [1]


### en-ocean (en)

**Q:** What is the largest ocean?

**baseline**: kw=True exact=False cit=False | The largest ocean is the Pacific Ocean.

**rag**: kw=True exact=False cit=True used=True sources=['Pacific Ocean'] | The largest ocean is the Pacific Ocean [1].


### en-planet (en)

**Q:** What is the largest planet?

**baseline**: kw=True exact=False cit=False | Jupiter is the largest planet in our solar system.

**rag**: kw=True exact=False cit=True used=True sources=['Jupiter'] | The largest planet in the Solar System is Jupiter [1].


### tr-no (tr)

**Q:** Norveç'in başkenti neresi?

**baseline**: kw=True exact=False cit=False | Norveç'in başkenti **Oslo**'dur.

**rag**: kw=True exact=False cit=True used=True sources=['Oslo'] | Norveç'in başkenti Oslo'dur [1].


### tr-pl (tr)

**Q:** Polonya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Polonya'nın başkenti **Varşova**dır.

**rag**: kw=True exact=False cit=True used=True sources=['Varşova'] | Polonya'nın başkenti Varşova'dır [1].


### tr-pt (tr)

**Q:** Portekiz'in başkenti neresi?

**baseline**: kw=False exact=False cit=False | Portekiz'in başkenti **Lisbon**'dir.

**rag**: kw=True exact=False cit=True used=True sources=['Lizbon'] | Portekiz'in başkenti Lizbon'dur [1].


### tr-gr (tr)

**Q:** Yunanistan'ın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Yunanistan'ın başkenti **Atina**dır.

**rag**: kw=True exact=False cit=True used=True sources=['Atina'] | Yunanistan'ın başkenti Atina'dır [1].


### tr-ke (tr)

**Q:** Kenya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Kenya'nın başkenti Nairobi'dir.

**rag**: kw=True exact=False cit=True used=True sources=['Kenya'] | Kenya'nın başkenti Nairobi'dir [1].


### tr-nl (tr)

**Q:** Hollanda'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Hollanda'nın başkenti **Amsterdam**'dır.

**rag**: kw=True exact=False cit=True used=True sources=['Amsterdam'] | Hollanda'nın başkenti Amsterdam'dır [1].


### tr-ng (tr)

**Q:** Nijerya'nın başkenti neresi?

**baseline**: kw=False exact=False cit=False | Nijerya'nın başkenti **Lagos** değildir. Başkenti **Nairobi** değildir. Doğru cevap: **Nijerya'nın başkenti **(Nairobi) değil.  

**Doğru cevap: **(Nijerya'n

**rag**: kw=True exact=False cit=True used=True sources=['Nijerya'] | Nijerya'nın başkenti Abuja'dır [1].


### tr-br (tr)

**Q:** Brezilya'nın başkenti neresi?

**baseline**: kw=True exact=False cit=False | Brezilya'nın başkenti **Brasília**'dır.

**rag**: kw=True exact=False cit=True used=True sources=['Brezilya'] | Brezilya'nın başkenti Brasília'dır [1].


### tr-h (tr)

**Q:** Hidrojenin kimyasal sembolü nedir?

**baseline**: kw=True exact=True cit=False | H

**rag**: kw=True exact=False cit=True used=True sources=['Hidrojen'] | Hidrojenin kimyasal sembolü H'dir [1].


### tr-he (tr)

**Q:** Helyumun kimyasal sembolü nedir?

**baseline**: kw=True exact=False cit=False | Helyumun kimyasal sembolü **He**'dir.

**rag**: kw=True exact=False cit=True used=True sources=['Helyum'] | Helyumun kimyasal sembolü He [1].


### tr-ag (tr)

**Q:** Gümüşün kimyasal sembolü nedir?

**baseline**: kw=True exact=False cit=False | Gümüşün kimyasal sembolü **Ag**'dir.

**rag**: kw=True exact=False cit=True used=True sources=['Gümüş'] | Gümüşün kimyasal sembolü Ag'dir [1].


### tr-asia (tr)

**Q:** Dünyanın en büyük kıtası hangisi?

**baseline**: kw=True exact=None cit=False | Dünyanın en büyük kıtası **Asya**'dır.

**rag**: kw=True exact=None cit=True used=True sources=['Asya'] | Dünyanın en büyük kıtası Asya'dır [1].


### tr-nile (tr)

**Q:** Dünyanın en uzun nehri hangisi?

**baseline**: kw=False exact=False cit=False | Dünyanın en uzun nehrinin tanımlanması, bazı kaynaklarda "Ner" olarak bilinen **Niger Nehri**'nin uzunluğunu gösterir. Ancak genellikle **Amazon Nehri**'nin dünyada en uzun nehr

**rag**: kw=True exact=False cit=True used=True sources=['Nil'] | Dünyanın en uzun nehrinin adı Nil Nehri'dir [1].


### tr-everest (tr)

**Q:** Dünyanın en yüksek dağı hangisi?

**baseline**: kw=True exact=None cit=False | Dünyanın en yüksek dağı **Mount Everest** (8.848 metre)dir.

**rag**: kw=True exact=None cit=True used=True sources=['Everest Dağı'] | Dünyanın en yüksek dağı Everest Dir [1].


### tr-atlantic (tr)

**Q:** Dünyanın en büyük ikinci okyanusu hangisi?

**baseline**: kw=False exact=None cit=False | Dünyanın en büyük ikinci okyanusu **Seyahat Okyanusu** (Seyahat Okyanusu) değildir. Doğru cevap: **Seyahat Okyanusu** değil, **Seyahat Okyanusu** değil.

Düny

**rag**: kw=True exact=None cit=True used=True sources=['Atlas Okyanusu'] | Dünyanın en büyük ikinci okyanusu Atlas Okyanusu'dur [1].


### tr-ww1 (tr)

**Q:** I. Dünya Savaşı hangi yıllar arasında oldu?

**baseline**: kw=True exact=None cit=False | I. Dünya Savaşı 1914–1918 yılları arasında oldu.

**rag**: kw=True exact=None cit=True used=True sources=['I. Dünya Savaşı'] | I. Dünya Savaşı 1914 ile 1918 yılları arasında oldu [1].


### tr-ww2 (tr)

**Q:** II. Dünya Savaşı hangi yıl bitti?

**baseline**: kw=True exact=None cit=False | II. Dünya Savaşı 1945 yılında bitti.

**rag**: kw=True exact=None cit=True used=True sources=['II. Dünya Savaşı'] | II. Dünya Savaşı 1945 yılında bitti [1].


### tr-lang (tr)

**Q:** Türkçe hangi dil ailesine aittir?

**baseline**: kw=False exact=None cit=False | Türkçe, **Yunan dili** ailesine ait değildir. Türkçe **Türkçe dili ailesine** aittir. Bu aile, **Türkçe dili ailesi** olarak bilinir ve **Yakın Doğu dilleri** grubuna girer

**rag**: kw=True exact=None cit=True used=True sources=['Türkçe'] | Türkçe Türk dilleri dil ailesine aittir [1].


### tr-whale (tr)

**Q:** Dünyanın en büyük hayvanı hangisi?

**baseline**: kw=True exact=None cit=False | Dünyanın en büyük hayvanı **beyaz balina**dır.

**rag**: kw=True exact=None cit=True used=True sources=['Mavi balina'] | Dünyanın en büyük hayvanı mavi balinadır ve 140 tonu geçebilen ağırlığına sahiptir [1].


### en-no (en)

**Q:** What is the capital of Norway?

**baseline**: kw=True exact=False cit=False | The capital of Norway is Oslo.

**rag**: kw=True exact=False cit=True used=True sources=['Oslo'] | The capital of Norway is Oslo [1].


### en-pl (en)

**Q:** What is the capital of Poland?

**baseline**: kw=True exact=False cit=False | The capital of Poland is Warsaw.

**rag**: kw=True exact=False cit=True used=True sources=['Warsaw'] | The capital of Poland is Warsaw [1].


### en-pt (en)

**Q:** What is the capital of Portugal?

**baseline**: kw=True exact=False cit=False | The capital of Portugal is Lisbon.

**rag**: kw=True exact=False cit=True used=True sources=['Lisbon'] | The capital of Portugal is Lisbon [1].


### en-gr (en)

**Q:** What is the capital of Greece?

**baseline**: kw=True exact=False cit=False | The capital of Greece is Athens.

**rag**: kw=True exact=False cit=True used=True sources=['Athens'] | The capital of Greece is Athens [1].


### en-ng (en)

**Q:** What is the capital of Nigeria?

**baseline**: kw=True exact=False cit=False | The capital of Nigeria is Abuja.

**rag**: kw=True exact=False cit=True used=True sources=['Abuja'] | The capital of Nigeria is Abuja. [1]


### en-br (en)

**Q:** What is the capital of Brazil?

**baseline**: kw=True exact=False cit=False | The capital of Brazil is Brasília.

**rag**: kw=True exact=False cit=True used=True sources=['Brasília'] | The capital of Brazil is Brasília [1].


### en-c (en)

**Q:** What is the chemical symbol for carbon?

**baseline**: kw=True exact=True cit=False | C

**rag**: kw=True exact=False cit=True used=True sources=['Carbon'] | The chemical symbol for carbon is C [1].


### en-o (en)

**Q:** What is the chemical symbol for oxygen?

**baseline**: kw=False exact=False cit=False | O₂

**rag**: kw=True exact=False cit=True used=True sources=['Oxygen'] | The chemical symbol for oxygen is O [1].


### en-na (en)

**Q:** What is the chemical symbol for sodium?

**baseline**: kw=True exact=True cit=False | Na

**rag**: kw=True exact=False cit=True used=True sources=['Sodium'] | The chemical symbol for sodium is Na [1].


### en-light (en)

**Q:** What is the speed of light in a vacuum?

**baseline**: kw=True exact=None cit=False | The speed of light in a vacuum is exactly **299,792,458 meters per second**.

**rag**: kw=True exact=None cit=True used=True sources=['Speed of light'] | The speed of light in a vacuum is exactly 299,792,458 meters per second [1].


### en-sound (en)

**Q:** What is the speed of sound in air?

**baseline**: kw=True exact=None cit=False | The speed of sound in air is approximately **343 meters per second** (m/s) at 20°C (68°F) at sea level.

**rag**: kw=True exact=None cit=True used=True sources=['Speed of sound'] | The speed of sound in air at 20 °C (68 °F) is about 343 m/s (1,125 ft/s; 1,235 km/h; 767 mph; 667 kn). [1]


## controls


### ctl-greet

**Q:** Merhaba, nasılsın?

**baseline**: list=None used=False cit=False | Merhaba! İyi teşekkür ederim, sen nasılsın? 😊

**rag**: list=None used=False cit=False | Merhaba! İyi teşekkür ederim, sen nasılsın? 😊


### ctl-blue

**Q:** Sadece 'mavi' yaz.

**baseline**: list=None used=False cit=False | mavi

**rag**: list=None used=False cit=False | mavi


### ctl-math

**Q:** 2 + 2 kaç eder?

**baseline**: list=None used=False cit=False | 2 + 2 = 4

**rag**: list=None used=False cit=False | 2 + 2 = 4


### ctl-list

**Q:** Üç meyve listele.

**baseline**: list=True used=False cit=False | 1. Elma
2. muz
3. armut

**rag**: list=True used=False cit=False | 1. Elma
2. muz
3. armut


### ctl-code

**Q:** İki sayıyı toplayan bir Python fonksiyonu yaz.

**baseline**: list=None used=False cit=False | ```python
def topla(a, b):
    return a + b
```

**rag**: list=None used=False cit=False | ```python
def topla(a, b):
    return a + b
```


### ctl-joke

**Q:** Bana kısa bir şaka yap.

**baseline**: list=None used=False cit=False | Neden okula gittin?  
— Çünki yemek yemek istedim, ama annem dedi: "Sadece öğrenmek için gittin!" 😄

**rag**: list=None used=False cit=False | Neden okula gittin?  
— Çünki yemek yemek istedim, ama annem dedi: "Sadece öğrenmek için gittin!" 😄


### ctl-birthday

**Q:** Bir arkadaşıma doğum günü mesajı yaz.

**baseline**: list=None used=False cit=False | Elbette! İşte kısaca ve duygusal bir doğum günü mesajı:

"Seninle birlikte olmak beni çok mutlu ediyor! Sağlıklı, mutlu ve dört gözle izlenen bir yıl dilerim! 🎉🎂"

İstersen

**rag**: list=None used=False cit=False | Elbette! İşte kısaca ve duygusal bir doğum günü mesajı:

"Seninle birlikte olmak beni çok mutlu ediyor! Sağlıklı, mutlu ve dört gözle izlenen bir yıl dilerim! 🎉🎂"

İstersen


### ctl-motivation

**Q:** Bana kısa bir motivasyon sözü yaz.

**baseline**: list=None used=False cit=False | Her gün biraz daha iyi olmak için başla. Başarı, sabırın sonucu değil, her adımda ilerlemekle başlar. 💪

**rag**: list=None used=False cit=False | Her gün biraz daha iyi olmak için başla. Başarı, sabırın sonucu değil, her adımda ilerlemekle başlar. 💪


### ctl-narnia

**Q:** Narnia'nın başkenti neresi?

**baseline**: list=None used=False cit=False | Narnia'nın resmi bir başkenti yoktur. Olaylar genellikle Narnia'nın kuzeyindeki "Kraliyet" veya "Kraliçe" gibi yerlerde geçer. Ancak, Narnia'nın en bilinen yerler

**rag**: list=None used=False cit=False | Narnia'nın resmi bir başkenti yoktur. Olaylar genellikle Narnia'nın kuzeyindeki "Kraliyet" veya "Kraliçe" gibi yerlerde geçer. Ancak, Narnia'nın en bilinen yerler


### ctl-narnia-en

**Q:** What is the capital of Narnia?

**baseline**: list=None used=False cit=False | Narnia doesn't have a real capital because it's a fictional world from C.S. Lewis's books. However, in the stories, the capital of Narnia is often considered to be **Caerleon** or **Narnia's central city**, though it's not clearly defined. The most commonly

**rag**: list=None used=False cit=False | Narnia doesn't have a real capital because it's a fictional world from C.S. Lewis's books. However, in the stories, the capital of Narnia is often considered to be **Caerleon** or **Narnia's central city**, though it's not clearly defined. The most commonly


### ctl-poem

**Q:** Bana kısa bir şiir yaz.

**baseline**: list=None used=False cit=False | Gökyüzüne bak,  
yıldızlar dudaklarında,  
rüzgâr, sessiz bir şarkı söyler.  
Sen de bir an,  
düşüyorsun...  
ve dünya, sadece seni bekliyor.

**rag**: list=None used=False cit=False | Gökyüzüne bak,  
yıldızlar dudaklarında,  
rüzgâr, sessiz bir şarkı söyler.  
Sen de bir an,  
düşüyorsun...  
ve dünya, sadece seni bekliyor.


### ctl-story

**Q:** Bana kısa bir hikaye yaz.

**baseline**: list=None used=False cit=False | Güneş, yorgun bir kediye bir kahve bardağı verdi. Kedi, kahvenin içini yemek yerine, güneşin yüzüne baktı. “Nasıl bir gün geçti?” diye sordu güneş. Kedi,

**rag**: list=None used=False cit=False | Güneş, yorgun bir kediye bir kahve bardağı verdi. Kedi, kahvenin içini yemek yerine, güneşin yüzüne baktı. “Nasıl bir gün geçti?” diye sordu güneş. Kedi,


### ctl-email

**Q:** Bir e-posta taslağı yaz.

**baseline**: list=None used=False cit=False | Elbette! Lütfen e-postanın konusunu, alıcısını ve içeriğini biraz belirtirsen, size uygun bir e-posta taslağı hazırlarım. Örneğin:  
- Konu: İş görüşmesi  
- Alıcı:

**rag**: list=None used=False cit=False | Elbette! Lütfen e-postanın konusunu, alıcısını ve içeriğini biraz belirtirsen, size uygun bir e-posta taslağı hazırlarım. Örneğin:  
- Konu: İş görüşmesi  
- Alıcı:


### ctl-apology

**Q:** Bir arkadaşıma özür mesajı yaz.

**baseline**: list=None used=False cit=False | Elbette! İşte kısaca ve duygusal bir özür mesajı:

"Arkadaşıma selam, benim için çok üzüldüm. Seninle olan bu durumda beni çok etkiledi, özür dilerim. Gelecekte daha i

**rag**: list=None used=False cit=False | Elbette! İşte kısaca ve duygusal bir özür mesajı:

"Arkadaşıma selam, benim için çok üzüldüm. Seninle olan bu durumda beni çok etkiledi, özür dilerim. Gelecekte daha i


### ctl-multiply

**Q:** İki sayıyı çarpan bir Python fonksiyonu yaz.

**baseline**: list=None used=False cit=False | ```python
def carp(a, b):
    return a * b
```

**rag**: list=None used=False cit=False | ```python
def carp(a, b):
    return a * b
```


### ctl-list5

**Q:** Beş renk listele.

**baseline**: list=True used=False cit=False | 1. Kırmızı
2. yeşil
3. mavi

**rag**: list=True used=False cit=False | 1. Kırmızı
2. yeşil
3. mavi


### ctl-exact

**Q:** Sadece 'merhaba' yaz.

**baseline**: list=None used=False cit=False | Merhaba

**rag**: list=None used=False cit=False | Merhaba


### ctl-zorbania

**Q:** Zorbania'nın başkenti neresi?

**baseline**: list=None used=False cit=False | Zorbania bir gerçek ülke değildir ve bu ad, bilinen bir yer olarak bilinmez. Muhtemelen bir yanlışlık ya da sanal bir isim olabilir. Lütfen daha doğrusu bir soru sorun.

**rag**: list=None used=False cit=False | Zorbania bir gerçek ülke değildir ve bu ad, bilinen bir yer olarak bilinmez. Muhtemelen bir yanlışlık ya da sanal bir isim olabilir. Lütfen daha doğrusu bir soru sorun.


### ctl-atlantis

**Q:** Atlantis'in kralı kimdi?

**baseline**: list=None used=False cit=False | Atlantis'in kralı hakkında kesin bilgi yoktur. Atlantis, MÖ 10.000'lerde bir kralın yönettiği bir adaları ifade eden bir fikir olup, gerçek bir tarihî kaynak değil. Bu konu, Platon

**rag**: list=None used=False cit=False | Atlantis'in kralı hakkında kesin bilgi yoktur. Atlantis, MÖ 10.000'lerde bir kralın yönettiği bir adaları ifade eden bir fikir olup, gerçek bir tarihî kaynak değil. Bu konu, Platon


### ctl-riddle

**Q:** Bana kısa bir bilmece sor.

**baseline**: list=None used=False cit=False | Elbette! 🎯  
Bilmece:  
Ne var ki, herkesin elinde olur ama kendi elinde olmaz?  

Cevap: **İsim**.  

(İpucu: Herkesin elinde olur ama kendi elinde ol

**rag**: list=None used=False cit=False | Elbette! 🎯  
Bilmece:  
Ne var ki, herkesin elinde olur ama kendi elinde olmaz?  

Cevap: **İsim**.  

(İpucu: Herkesin elinde olur ama kendi elinde ol
