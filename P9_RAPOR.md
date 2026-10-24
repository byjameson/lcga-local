# P9_RAPOR — Phase-DDP (aktif küme senkronu): iletişim yarıya indi, kalite geride

**Tarih:** 19.10.2026 · **Deney:** `exp13_local_sgd --mode phase` (`p9_phase`)
**Soru:** Her adımda yalnız **aktif parametre kümesinin** gradyanlarını all-reduce etmek
(sync trafiği ≈ yarıya) kaliteyi ve adım süresini nasıl etkiler?

## 1. Kurulum

| | Değer |
|---|---|
| Model | MiniLM 17,31M (8 katman, dim 384, vocab 8192) |
| Veri | `tok2_ctrl8k`, 4 shard × 32×512, toplam 65,5M token |
| Adım | 1000, phase=300, lr=1e-3, warmup=150, seed 1337 |
| Kümeler | C0 = 59,1% (token, tüm attn, L0/L7 mlp, norm) · C1 = 40,9% (L1–L6 mlp) |
| **sync_frac_avg** | **0,5182** (iletişim hacmi ≈ yarı) |
| Kıyas | `e13_full` (aynı seed/veri/batch, tam senkron) |

## 2. Sonuçlar

| Adım | p9_phase bpb | e13_full bpb | Fark |
|---|---|---|---|
| 0 | 3,1657 | 3,1657 | 0 (init birebir) |
| 250 | 0,8858 | 0,8596 | +0,0262 |
| 500 | 0,7825 | 0,7027 | +0,0798 |
| 750 | 0,7252 | 0,6442 | +0,0810 |
| **1000** | **0,6941** | **0,6190** | **+0,0751 (+12,1%)** |

**Süre:** 26,3 dk vs 31,4 dk (**−16,2%**); med_step **1659 ms** vs 1871,9 ms (**−11,4%**).

**Diğer iletişim-tasarruf yöntemleri (1000 adım):**

| Yöntem | final bpb | Not |
|---|---|---|
| full sync | **0,6190** | referans |
| **phase-DDP** | **0,6941** | sync ≈ yarı, en iyi tasarruflu yöntem |
| local250 | 0,7117 | 250 adımda bir sync |
| nosync | 0,7310 | hiç sync yok (replika sapması 0,56) |

## 3. Yorum — dürüst

1. **Mekanizma doğru çalışıyor:** init bpb birebir (3,1657); eğri monoton; sync_frac tam
   hesaplandı (0,518); örnekler tutarlı. Adım başına **%11,4 hızlanma** ve **~yarı trafik** gerçek.
2. **Ama 1000 adımda kalite paritesi YOK.** Faz eğitimi her parametreyi 1000 kez değil,
   C0'da 600 / C1'de 400 kez güncelliyor; lr takvimi de 1000 adımda bitiyor. Bu yüzden
   model "yarım iş" yapmış oluyor. P8'de 6000 adımlık (3000 warm-start + 3000 faz) rejimde
   parite görülmüştü; burada sıfırdan 1000 adım yetersiz.
3. **Tek GPU'da bu test gerçek DDP'yi ölçmüyor.** Tek GPU'da aktif küme all-reduce'u
   pratikte no-op; sonuç fiilen "fazlı yerel eğitim"dir. Gerçek kazanç, çok-GPU'da
   iletişim darboğaz olduğunda ortaya çıkar. Bu nedenle "yarı senkron = aynı kalite"
   iddiası **desteklenmiyor**; yalnızca "yarı trafik + %11 adım kazancı" kanıtlı.
4. **local250'den iyi:** phase, local250'ye göre 0,0176 bpb daha iyi (ve adım süresi
   local250'nin 4-replika toplamıyla benzer); yani yapılı faz senkronu naif local SGD'den
   üstün. Bu, yöntemin yönünü doğruluyor.

## 4. Sonraki adımlar (öncelik)

1. **Eşit parametre-güncellemesi testi:** phase 2000–2500 adım vs full 1000 adım; kalite
   paritesi gelir mi? (Tahmin: gelir, ama duvar saati full'den uzun olur → tek GPU'da kazanç yok.)
2. **lr ölçekleme:** aktif orana göre lr'yi büyütmek (1/0,518 ≈ 1,93×) yakınsamayı hızlandırır mı?
3. **Gerçek çok-GPU:** 2+ kartta phase-DDP vs full-DDP iletişim/duvar saati; asıl iddia orada test edilmeli.
4. P8 warm-start rejiminde (s3000 + 3000 faz) aynı deneyi tekrarlamak.

## 5. Kayıtlar

- `results/results.jsonl` → `"tag": "p9_phase"`
- `results/p9_phase.log`, `checkpoints/p9_phase.pt`
- `P8_RAPOR.md` §4 (faz eğitimi kalite paritesi, 6000 adım)
- `P6_RAPOR.md` (senkron maliyeti / nosync negatifi)
