@echo off
setlocal
cd /d "%~dp0"
python -c "import cv2, numpy, PIL, customtkinter, onnxruntime" 2>nul
if errorlevel 1 (
    echo Dang cai dat thu vien can thiet...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Khong the cai dat thu vien. Kiem tra Python va ket noi mang.
        pause
        exit /b 1
    )
)
python main.py
if errorlevel 1 pause
