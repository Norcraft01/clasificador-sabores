@echo off
chcp 65001 >nul
title Clasificador de sabores
cd /d "%~dp0"

echo ==========================================
echo    Clasificador de sabores en comida (v2)
echo ==========================================
echo.

where uv >nul 2>nul
if errorlevel 1 (
  echo [1/2] Instalando uv (gestor de Python, solo la primera vez)...
  powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

nvidia-smi >nul 2>nul
if errorlevel 1 (set "EXTRA=cpu") else (set "EXTRA=gpu")
echo [2/2] Preparando el entorno (torch %EXTRA%). La primera vez tarda varios minutos...
echo       Cuando aparezca "Running on local URL", se abre el navegador solo.
echo.
uv run --extra %EXTRA% python app.py
echo.
echo La app se cerro. Si hubo un error, mandale captura de esta ventana al equipo.
pause
