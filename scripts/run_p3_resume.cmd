@echo off
rem P3 resume: step 2000/18000'den devam (18.10.2026 gece kosusu)
cd /d "%~dp0.."
rem not: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True test edildi -> Windows'ta desteklenmiyor, kullanilmadi
echo [p3r] basladi: %date% %time% >> results\p3_resume_runner.out
python experiments\exp11_own_lm.py --tag p3 --steps 18000 --batch 48 --warmup 500 --vocab 8192 --n_eval 128 --resume checkpoints\p3_ckpt.pt > results\p3_resume.log 2>&1
if errorlevel 1 echo [p3r][NOT] p3 resume exit=%errorlevel% >> results\p3_resume_runner.out
echo [p3r] bitti: %date% %time% >> results\p3_resume_runner.out
echo bitti %date% %time% > results\p3_resume_DONE.txt
exit /b 0
