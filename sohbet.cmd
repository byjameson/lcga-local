@echo off
chcp 65001 >nul
cd /d "%~dp0"
python llm.py %*
if errorlevel 1 pause
