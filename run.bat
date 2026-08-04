@echo off
setlocal
cd /d "%~dp0"
set "AFGI_DATA_SOURCE=postgres"

".venv_pg\Scripts\python.exe" -m streamlit run app.py

if errorlevel 1 pause
endlocal
