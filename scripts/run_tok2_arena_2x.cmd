@echo off
rem P5 "Tokenizer 2" - Asama 2b: 2x butce arenasi (SIFIRDAN, 12000 adim x 32 x 512 = 196.6M token)
rem Soru: sw16k yakinsayinca ctrl8k'yi geciyor mu? (1x'te 0.5612 vs 0.5769; sw16k hala hizli iniyordu)
rem Sira: sw16k -> ctrl8k -> wb16k -> sw8k  (en kritik kiyas once; kesinti olsa bile tamamlanmis olur)
rem Ayni veri, ayni ayar (batch 32, warmup 300, n_eval 128); loglar results\a3_*.log
rem Bitis isareti: results\a3_DONE.txt
cd /d "%~dp0.."
echo [arena3] basladi: %date% %time% >> results\a3_runner.out
echo [arena3] kol 1/4: a3_sw16k (2x) >> results\a3_runner.out
python experiments\exp11_own_lm.py --tag a3_sw16k --steps 12000 --batch 32 --warmup 300 --vocab 16384 --data_prefix tok2_sw16k --tok_path data\tok2\bpe_sw_16k.json --n_eval 128 > results\a3_sw16k.log 2>&1
if errorlevel 1 echo [arena3][NOT] a3_sw16k exit=%errorlevel% >> results\a3_runner.out
echo [arena3] kol 2/4: a3_ctrl8k (2x) >> results\a3_runner.out
python experiments\exp11_own_lm.py --tag a3_ctrl8k --steps 12000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 > results\a3_ctrl8k.log 2>&1
if errorlevel 1 echo [arena3][NOT] a3_ctrl8k exit=%errorlevel% >> results\a3_runner.out
echo [arena3] kol 3/4: a3_wb16k (2x) >> results\a3_runner.out
python experiments\exp11_own_lm.py --tag a3_wb16k --steps 12000 --batch 32 --warmup 300 --vocab 16384 --data_prefix tok2_wb16k --tok_path data\tok2\bpe_wb_16k.json --n_eval 128 > results\a3_wb16k.log 2>&1
if errorlevel 1 echo [arena3][NOT] a3_wb16k exit=%errorlevel% >> results\a3_runner.out
echo [arena3] kol 4/4: a3_sw8k (2x) >> results\a3_runner.out
python experiments\exp11_own_lm.py --tag a3_sw8k --steps 12000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_sw8k --tok_path data\tok2\bpe_sw_8k.json --n_eval 128 > results\a3_sw8k.log 2>&1
if errorlevel 1 echo [arena3][NOT] a3_sw8k exit=%errorlevel% >> results\a3_runner.out
echo [arena3] bitti: %date% %time% >> results\a3_runner.out
echo bitti %date% %time% > results\a3_DONE.txt
