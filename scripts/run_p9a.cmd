@echo off
rem P9a: ikinci rastgele-partition kolu (kesim profili eslesli: frac0~0.546, cut~34)
rem rel_s1337 ile AYNI tohum/butce; tek fark partition -> partition-robustluk testi
cd /d "%~dp0.."
echo [p9a] basladi: %date% %time% >> results\p9_runner.out
python experiments\exp14_p8_clusters.py --mode train --schedule random --cluster_file results\p8_clusters_rnd2.npz --resume checkpoints\p3_ckpt_s3000.pt --tag p8_rnd2_s1337 --steps 3000 --phase 300 --seed 1337 > results\p8_rnd2_s1337.log 2>&1
echo [p9a] rnd2 bitti (rc=%errorlevel%): %date% %time% >> results\p9_runner.out
echo bitti %date% %time% > results\p9a_DONE.txt
exit /b 0
