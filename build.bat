@echo off
setlocal enabledelayedexpansion
echo ========================================
echo  BASI BOT - BUILD SCRIPT
echo  Multi-Agent Discord LLM Chatbot System
echo ========================================
echo.

echo [1/7] Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed and in PATH
    pause
    exit /b 1
)
echo Virtual environment created successfully.
echo.

echo [2/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo.

echo [3/7] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [4/7] Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install packages
    pause
    exit /b 1
)
echo.

echo [5/7] Setting up FFmpeg binaries...
if not exist "bin" mkdir bin

REM Check if FFmpeg already exists
if exist "bin\ffmpeg.exe" (
    echo FFmpeg already installed in bin\ffmpeg.exe
    goto :ffprobe_check
)

echo Downloading FFmpeg for Windows...
REM Download FFmpeg essentials build (smaller, has what we need)
set FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip
set FFMPEG_ZIP=bin\ffmpeg-download.zip

REM Use PowerShell to download (works on all modern Windows)
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%FFMPEG_ZIP%' -UseBasicParsing }"
if errorlevel 1 (
    echo WARNING: Could not download FFmpeg automatically.
    echo Please manually download FFmpeg from: https://github.com/BtbN/FFmpeg-Builds/releases
    echo Extract ffmpeg.exe and ffprobe.exe to the bin\ folder.
    goto :skip_ffmpeg
)

echo Extracting FFmpeg...
REM Extract using PowerShell
powershell -Command "& { Expand-Archive -Path '%FFMPEG_ZIP%' -DestinationPath 'bin\ffmpeg-temp' -Force }"
if errorlevel 1 (
    echo ERROR: Failed to extract FFmpeg
    goto :skip_ffmpeg
)

REM Find and copy the binaries (they're in a subfolder)
for /d %%D in (bin\ffmpeg-temp\ffmpeg-*) do (
    if exist "%%D\bin\ffmpeg.exe" (
        copy "%%D\bin\ffmpeg.exe" "bin\ffmpeg.exe" >nul
        copy "%%D\bin\ffprobe.exe" "bin\ffprobe.exe" >nul
        echo Copied FFmpeg binaries to bin\
    )
)

REM Cleanup temp files
if exist "bin\ffmpeg-temp" rmdir /s /q "bin\ffmpeg-temp"
if exist "%FFMPEG_ZIP%" del "%FFMPEG_ZIP%"

:ffprobe_check
if exist "bin\ffprobe.exe" (
    echo FFprobe installed in bin\ffprobe.exe
) else (
    echo WARNING: ffprobe.exe not found - some video features may not work
)

:skip_ffmpeg
echo.

echo [6/7] Verifying FFmpeg installation...
if exist "bin\ffmpeg.exe" (
    echo FFmpeg found: bin\ffmpeg.exe
    bin\ffmpeg.exe -version 2>nul | findstr /C:"ffmpeg version" >nul
    if errorlevel 1 (
        echo WARNING: FFmpeg binary may be corrupted
    ) else (
        echo FFmpeg is working correctly
    )
) else (
    echo WARNING: FFmpeg not installed - video generation features will be disabled
    echo To enable video features, manually download FFmpeg and place in bin\
)
echo.

echo [7/7] Creating required directories...
if not exist "config" mkdir config
if not exist "data" mkdir data
if not exist "data\video_temp" mkdir data\video_temp
echo.

echo ========================================
echo  BUILD SUCCESSFUL
echo ========================================
echo.
echo FFmpeg Status:
if exist "bin\ffmpeg.exe" (
    echo   [OK] ffmpeg.exe installed
) else (
    echo   [!!] ffmpeg.exe NOT FOUND - video features disabled
)
if exist "bin\ffprobe.exe" (
    echo   [OK] ffprobe.exe installed
) else (
    echo   [!!] ffprobe.exe NOT FOUND
)
echo.
echo To start the bot, run: run.bat
echo.
pause
