@echo off
rem P9b: kume-fazli "phase-DDP" -- from scratch, esit butce, ~yarim gradyan senkronu
rem p9a_DONE bekler -> p9_phase (1000 adim x 4 shard x 32x512) -> p9b_DONE.txt
cd /d "%~dp0.."
echo [p9b] basladi, p9a_DONE bekleniyor: %date% %time% >> results\p9_runner.out
:wait
if exist results\p9a_DONE.txt goto go
ping -n 31 127.0.0.1 >nul
goto wait
:go
echo [p9b] phase kolu basliyor: %date% %time% >> results\p9_runner.out
python experiments\exp13_local_sgd.py --mode phase --tag p9_phase --steps 1000 --shards 4 --batch 32 --data_prefix tok2_ctrl8k --phase 300 --cluster_file results\p8_clusters.npz --tok_path data\ts_bpe8k.json > results\p9_phase.log 2>&1
echo [p9b] phase bitti (rc=%errorlevel%): %date% %time% >> results\p9_runner.out
echo bitti %date% %time% > results\p9b_DONE.txt
exit /b 0
