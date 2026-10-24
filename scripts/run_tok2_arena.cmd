@echo off
rem P5 "Tokenizer 2" - Asama 2 arenasi (KALAN KOLLAR)
rem ctrl8k TAMAM (bpb 0.5612). wb16k: oturum-yenileme kesintisi -> ckpt step 1000'den RESUME.
rem Bu script oturumdan BAGIMSIZ baslatilir (WMI); loglar: results\a2_*.log + a2_runner.out
rem Bitis isareti: results\a2_DONE.txt
cd /d "%~dp0.."
echo [arena2] kalan kollar basladi: %date% %time% >> results\a2_runner.out
echo [arena2] kol 2/4 devam: a2_wb16k (resume @1000) >> results\a2_runner.out
python experiments\exp11_own_lm.py --tag a2_wb16k --steps 6000 --batch 32 --warmup 300 --vocab 16384 --data_prefix tok2_wb16k --tok_path data\tok2\bpe_wb_16k.json --n_eval 128 --resume checkpoints\a2_wb16k_ckpt.pt > results\a2_wb16k.log 2>&1
if errorlevel 1 echo [arena2][NOT] a2_wb16k exit=%errorlevel% (Triton kapanis 1 olabilir; log'a bak) >> results\a2_runner.out
echo [arena2] kol 3/4: a2_sw8k >> results\a2_runner.out
python experiments\exp11_own_lm.py --tag a2_sw8k --steps 6000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_sw8k --tok_path data\tok2\bpe_sw_8k.json --n_eval 128 > results\a2_sw8k.log 2>&1
if errorlevel 1 echo [arena2][NOT] a2_sw8k exit=%errorlevel% >> results\a2_runner.out
echo [arena2] kol 4/4: a2_sw16k >> results\a2_runner.out
python experiments\exp11_own_lm.py --tag a2_sw16k --steps 6000 --batch 32 --warmup 300 --vocab 16384 --data_prefix tok2_sw16k --tok_path data\tok2\bpe_sw_16k.json --n_eval 128 > results\a2_sw16k.log 2>&1
if errorlevel 1 echo [arena2][NOT] a2_sw16k exit=%errorlevel% >> results\a2_runner.out
echo [arena2] bitti: %date% %time% >> results\a2_runner.out
echo bitti %date% %time% > results\a2_DONE.txt
