@echo off
rem P4.2: dongu sayisini egitimde rastgelelestirme (L ~ U{1..4}) -- 18.10.2026
rem Referanslar: loop2x4 0.5761 (L=2 sabit), flat4 0.5889, flat8 0.5612
cd /d "%~dp0.."

echo [p4_2] basladi: %date% %time% >> results\p4_2_runner.out
echo [p4_2] 0/2 GPU smoke (20 adim, loops_rand 4) >> results\p4_2_runner.out
python experiments\exp11_own_lm.py --tag p4_2smoke --steps 20 --batch 32 --warmup 5 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 16 --layers 4 --loops 2 --loops_rand 4 --loop_sweep --eval_every 10 --gen_every 100000 --ckpt_every 100000 > results\p4_2smoke.log 2>&1
if errorlevel 1 goto :fail
echo [p4_2] smoke OK >> results\p4_2_runner.out

echo [p4_2] 1/2 ana kol: p4_looprand14 (6000 adim, L~U{1..4}) >> results\p4_2_runner.out
python experiments\exp11_own_lm.py --tag p4_looprand14 --steps 6000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 --layers 4 --loops 2 --loops_rand 4 --loop_sweep > results\p4_looprand14.log 2>&1
if errorlevel 1 echo [p4_2][NOT] p4_looprand14 exit=%errorlevel% >> results\p4_2_runner.out

echo [p4_2] bitti: %date% %time% >> results\p4_2_runner.out
echo bitti %date% %time% > results\p4_2_DONE.txt
exit /b 0

:fail
echo [p4_2][FAIL] GPU smoke hata verdi -- zincir durdu: %date% %time% >> results\p4_2_runner.out
echo FAIL %date% %time% > results\p4_2_FAIL.txt
exit /b 1
