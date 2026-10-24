"""35-prompt holdout set for the quality router (not used to design the cues)."""

# (category, prompt, keywords, exact_answer or None, needs_list)
PROMPTS = [
    # Turkish facts
    ("fact_tr", "Japonya'nin baskenti neresi?", ["tokyo"], None, False),
    ("fact_tr", "Misir'in baskenti neresi?", ["kahire", "cairo"], None, False),
    ("fact_tr", "Dunyanin en buyuk colu hangisi?", ["sahra", "sahara"], None, False),
    ("fact_tr", "Bir yilda kac hafta vardir?", ["52"], None, False),
    ("fact_tr", "Insan kalbinde kac odacik vardir?", ["4"], None, False),
    ("fact_tr", "Avustralya'nin baskenti neresi?", ["kanberra", "canberra"], None, False),
    # English facts
    ("fact_en", "What is the capital of Canada?", ["ottawa"], None, False),
    ("fact_en", "What is the chemical symbol for silver?", ["ag"], None, False),
    ("fact_en", "How many continents are there?", ["7"], None, False),
    ("fact_en", "What is the largest mammal?", ["blue whale", "whale"], None, False),
    ("fact_en", "At what temperature does water boil in Fahrenheit?", ["212"], None, False),
    ("fact_en", "How many sides does a hexagon have?", ["6"], None, False),
    # Turkish instructions
    ("ins_tr", "Sadece 'yesil' yaz.", ["yesil"], "yesil", False),
    ("ins_tr", "4 + 7 kac? Sadece sayiyi yaz.", ["11"], "11", False),
    ("ins_tr", "Tesekkur eden kisa bir cumle yaz.", ["tesekkur"], None, False),
    ("ins_tr", "Su cumleyi Ingilizceye cevir: Kopek bahcede.",
     ["dog", "garden"], None, False),
    ("ins_tr", "'Hayir' kelimesini iceren kisa bir cumle yaz.", ["hayir"], None, False),
    ("ins_tr", "10'dan 6'ya kadar sayilari yaz.",
     ["10", "9", "8", "7", "6"], None, False),
    # English instructions
    ("ins_en", "Reply with exactly the word 'green'.", ["green"], "green", False),
    ("ins_en", "What is 4 + 7? Answer with just the number.", ["11"], "11", False),
    ("ins_en", "Write a short thank-you sentence.", ["thank"], None, False),
    ("ins_en", "Translate 'The dog is in the garden' into Turkish.",
     ["kopek", "bahce"], None, False),
    ("ins_en", "Write one short sentence containing the word 'no'.", ["no"], None, False),
    ("ins_en", "Count down from 10 to 6.", ["10", "9", "8", "7", "6"], None, False),
    # Math
    ("math", "8 * 9 = ? Sadece sayiyi yaz.", ["72"], "72", False),
    ("math", "50 - 18 = ? Sadece sayiyi yaz.", ["32"], "32", False),
    ("math", "7 + 8 * 2 = ? Islem onceligine dikkat et.", ["23"], "23", False),
    ("math", "20'nin yarisi kac? Sadece sayiyi yaz.", ["10"], "10", False),
    ("math", "3^3 = ? Sadece sayiyi yaz.", ["27"], "27", False),
    ("math", "100'un yarisi kac? Sadece sayiyi yaz.", ["50"], "50", False),
    # Lists
    ("list", "Uc hayvan listele.", [], None, True),
    ("list", "List three fruits.", [], None, True),
    ("list", "Uc meslek listele.", [], None, True),
    ("list", "List three European countries.", [], None, True),
    ("list", "Uc sebze listele.", [], None, True),
]
