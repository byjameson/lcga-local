@echo off
rem MoE-lite (exp15): p8_DONE bekler -> moe4 kolu (E=4 top-1, 6000 adim, batch 32)
cd /d "%~dp0.."
echo [moe] basladi, p8_DONE bekleniyor: %date% %time% >> results\moe_runner.out
:wait
if exist results\p8_DONE.txt goto go
ping -n 31 127.0.0.1 >nul
goto wait
:go
echo [moe] moe4 kolu basliyor: %date% %time% >> results\moe_runner.out
python experiments\exp11_own_lm.py --tag moe4 --steps 6000 --batch 32 --warmup 300 --vocab 8192 --n_eval 128 --data_prefix ts --moe_experts 4 > results\moe4.log 2>&1
echo [moe] moe4 bitti (rc=%errorlevel%): %date% %time% >> results\moe_runner.out
echo bitti %date% %time% > results\moe_DONE.txt
exit /b 0
