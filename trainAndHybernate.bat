@echo off
setlocal EnableExtensions

REM Optional: run relative to this script’s folder
pushd "%~dp0"

echo [%%date%% %%time%%] Starting training…

REM Use explicit interpreter path to avoid file-association oddities
REM and ensure we WAIT until the real process ends.
START "" /WAIT "C:\Users\randolf.schaerfig\.conda\envs\mediapipe\python.exe" ".\trainSketch2Image.py"
set "ERR=%ERRORLEVEL%"

echo.
echo [%%date%% %%time%%] Training finished with exit code: %ERR%
echo Regardless of errors, system will now hibernate.

REM Ensure hibernation is enabled (ignore errors if not admin)
powercfg -hibernate on >nul 2>&1

REM Try hibernate via the most reliable path
shutdown /h
