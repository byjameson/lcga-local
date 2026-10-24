@echo off
rem exp13 (P6): senkron maliyeti -- full vs local250 vs nosync (18.10.2026)
rem P4 zinciri bitmeden baslamaz (p4_DONE/p4_FAIL bekler; ~2 sa cap); GPU smoke kapili.
cd /d "%~dp0.."

set /a waited=0
:waitp4
if exist results\p4_DONE.txt goto go
if exist results\p4_FAIL.txt goto go
set /a waited+=1
if %waited% gtr 120 goto go
ping -n 61 127.0.0.1 >nul
goto waitp4
:go

echo [exp13] basladi: %date% %time% >> results\exp13_runner.out
echo [exp13] 0/3 GPU smoke (local, 60 adim) >> results\exp13_runner.out
python experiments\exp13_local_sgd.py --tag e13smoke_gpu --mode local --sync_every 30 --steps 60 --block 30 --batch 32 --shards 4 --warmup 10 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 32 --gen_tokens 40 > results\e13smoke_gpu.log 2>&1
if errorlevel 1 goto :fail
echo [exp13] smoke OK >> results\exp13_runner.out
echo [exp13] 1/3 full >> results\exp13_runner.out
python experiments\exp13_local_sgd.py --tag e13_full --mode full --steps 1000 --block 250 --batch 32 --shards 4 --warmup 150 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 > results\e13_full.log 2>&1
if errorlevel 1 echo [exp13][NOT] full exit=%errorlevel% >> results\exp13_runner.out
echo [exp13] 2/3 local250 >> results\exp13_runner.out
python experiments\exp13_local_sgd.py --tag e13_local250 --mode local --sync_every 250 --steps 1000 --block 250 --batch 32 --shards 4 --warmup 150 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 > results\e13_local250.log 2>&1
if errorlevel 1 echo [exp13][NOT] local250 exit=%errorlevel% >> results\exp13_runner.out
echo [exp13] 3/3 nosync >> results\exp13_runner.out
python experiments\exp13_local_sgd.py --tag e13_nosync --mode nosync --steps 1000 --block 250 --batch 32 --shards 4 --warmup 150 --vocab 8192 --data_prefix tok2_ctrl8k --tok_path data\ts_bpe8k.json --n_eval 128 > results\e13_nosync.log 2>&1
if errorlevel 1 echo [exp13][NOT] nosync exit=%errorlevel% >> results\exp13_runner.out
echo [exp13] bitti: %date% %time% >> results\exp13_runner.out
echo bitti %date% %time% > results\e13_DONE.txt
exit /b 0

:fail
echo [exp13][FAIL] GPU smoke hata verdi -- zincir durdu: %date% %time% >> results\exp13_runner.out
echo FAIL %date% %time% > results\e13_FAIL.txt
exit /b 1
