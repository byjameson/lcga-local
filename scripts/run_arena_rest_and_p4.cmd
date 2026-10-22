@echo off
rem P5 2x arenasi KALAN 2 kol + P4.1 (LoopLM mini testi) zinciri -- 18.10.2026
rem 1) a3_wb16k: ckpt step 2000'den RESUME (log'a append)
rem 2) a3_sw8k: sifirdan  -> a3_DONE.txt
rem 3) P4: GPU smoke (gecmezse DURUR) -> p4_loop2x4 -> p4_flat4 -> p4_DONE.txt
rem NOT: biten kollari tekrarlamaz. Tek WMI cagrisi ile oturumdan bagimsiz baslatilir.
cd /d "%~dp0.."

echo [arena3-rest] basladi: %date% %time% >> results\a3_runner.out
echo [arena3-rest] kol 3/4 devam: a3_wb16k (resume @2000) >> results\a3_runner.out
python experiments\exp11_own_lm.py --tag a3_wb16k --steps 12000 --batch 32 --warmup 300 --vocab 16384 --data_prefix tok2_wb16k --tok_path data\tok2\bpe_wb_16k.json --n_eval 128 --resume checkpoints\a3_wb16k_ckpt.pt >> results\a3_wb16k.log 2>&1
if errorlevel 1 echo [arena3-rest][NOT] a3_wb16k exit=%errorlevel% >> results\a3_runner.out
echo [arena3-rest] kol 4/4: a3_sw8k >> results\a3_runner.out
python experiments\exp11_own_lm.py --tag a3_sw8k --steps 12000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_sw8k --tok_path data\tok2\bpe_sw_8k.json --n_eval 128 > results\a3_sw8k.log 2>&1
if errorlevel 1 echo [arena3-rest][NOT] a3_sw8k exit=%errorlevel% >> results\a3_runner.out
echo [arena3-rest] bitti: %date% %time% >> results\a3_runner.out
echo bitti %date% %time% > results\a3_DONE.txt

echo [p4] chain basladi: %date% %time% >> results\p4_runner.out
echo [p4] 1/3 GPU smoke (30 adim) >> results\p4_runner.out
python experiments\exp11_own_lm.py --tag p4smoke_gpu --steps 30 --batch 32 --warmup 5 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 16 --layers 4 --loops 2 --loop_sweep --eval_every 15 --gen_every 100 --ckpt_every 100000 > results\p4smoke_gpu.log 2>&1
if errorlevel 1 goto :p4fail
echo [p4] smoke OK >> results\p4_runner.out
echo [p4] 2/3: p4_loop2x4 (6000 adim) >> results\p4_runner.out
python experiments\exp11_own_lm.py --tag p4_loop2x4 --steps 6000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 --layers 4 --loops 2 --loop_sweep > results\p4_loop2x4.log 2>&1
if errorlevel 1 echo [p4][NOT] p4_loop2x4 exit=%errorlevel% >> results\p4_runner.out
echo [p4] 3/3: p4_flat4 (6000 adim) >> results\p4_runner.out
python experiments\exp11_own_lm.py --tag p4_flat4 --steps 6000 --batch 32 --warmup 300 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 --layers 4 > results\p4_flat4.log 2>&1
if errorlevel 1 echo [p4][NOT] p4_flat4 exit=%errorlevel% >> results\p4_runner.out
echo [p4] bitti: %date% %time% >> results\p4_runner.out
echo bitti %date% %time% > results\p4_DONE.txt
exit /b 0

:p4fail
echo [p4][FAIL] GPU smoke hata verdi -- zincir durdu: %date% %time% >> results\p4_runner.out
echo FAIL %date% %time% > results\p4_FAIL.txt
exit /b 1
