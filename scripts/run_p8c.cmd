@echo off
rem P8v2 dogrulama (seed 1338): related + random kollarini ayni protokolle tekrarlar
rem moe_DONE bekler -> rel_s1338 -> rnd_s1338 -> p8v2_verify_DONE.txt
cd /d "%~dp0.."
echo [p8c] basladi, moe_DONE bekleniyor: %date% %time% >> results\p8_runner.out
:wait
if exist results\moe_DONE.txt goto go
ping -n 31 127.0.0.1 >nul
goto wait
:go
echo [p8c] rel_s1338 kolu basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule related --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rel_s1338 --steps 3000 --phase 300 --seed 1338 > results\p8_rel_s1338.log 2>&1
echo [p8c] rel_s1338 bitti (rc=%errorlevel%): %date% %time% >> results\p8_runner.out
echo [p8c] rnd_s1338 kolu basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule random --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rnd_s1338 --steps 3000 --phase 300 --seed 1338 > results\p8_rnd_s1338.log 2>&1
echo [p8c] rnd_s1338 bitti (rc=%errorlevel%): %date% %time% >> results\p8_runner.out
echo bitti %date% %time% > results\p8v2_verify_DONE.txt
echo [p8c] P8v2 dogrulama TAMAM: %date% %time% >> results\p8_runner.out
exit /b 0
