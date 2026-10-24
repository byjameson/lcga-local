@echo off
rem P8 (exp14): P3 bittikten sonra otomatik: corr -> related kolu -> random kolu
cd /d "%~dp0.."
echo [p8] basladi, P3 bekleniyor: %date% %time% >> results\p8_runner.out
:wait
if exist results\p3_resume_DONE.txt goto go
ping -n 31 127.0.0.1 >nul
goto wait
:go
echo [p8] P3 bitti, corr basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode corr --resume checkpoints\p3_ckpt_s3000.pt --out results\p8_clusters.npz > results\p8_corr.log 2>&1
if errorlevel 1 ( echo [p8][NOT] corr FAIL >> results\p8_runner.out & echo CORR_FAIL > results\p8_FAIL.txt & exit /b 1 )
echo [p8] related kolu basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule related --cluster_file results\p8_clusters.npz --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rel --steps 3000 --batch 48 --n_eval 128 --eval_every 500 --ckpt_every 1000 --data_prefix ts --steps_total 18000 --warmup 500 --lr 1e-3 > results\p8_rel.log 2>&1
echo [p8] related bitti (rc=%errorlevel%): %date% %time% >> results\p8_runner.out
echo [p8] random kolu basliyor: %date% %time% >> results\p8_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule random --cluster_file results\p8_clusters.npz --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rnd --steps 3000 --batch 48 --n_eval 128 --eval_every 500 --ckpt_every 1000 --data_prefix ts --steps_total 18000 --warmup 500 --lr 1e-3 > results\p8_rnd.log 2>&1
echo [p8] random bitti (rc=%errorlevel%): %date% %time% >> results\p8_runner.out
echo bitti %date% %time% > results\p8_DONE.txt
exit /b 0
