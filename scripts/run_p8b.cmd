@echo off
rem P8v2: exp14 --phase 300 (fazli parcali egitim; P8v1 adim-alternasyon hatasi duzeltildi)
rem Kollar: related -> random; bitince results\p8_DONE.txt
cd /d "%~dp0.."
echo [p8b] basladi (v2, phase=300): %date% %time% >> results\p8_runner.out
echo [p8b] related kolu basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule related --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rel --steps 3000 --phase 300 > results\p8_rel.log 2>&1
echo [p8b] related bitti (rc=%errorlevel%): %date% %time% >> results\p8_runner.out
echo [p8b] random kolu basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule random --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rnd --steps 3000 --phase 300 > results\p8_rnd.log 2>&1
echo [p8b] random bitti (rc=%errorlevel%): %date% %time% >> results\p8_runner.out
echo bitti %date% %time% > results\p8_DONE.txt
echo [p8b] P8v2 TAMAM: %date% %time% >> results\p8_runner.out
exit /b 0
