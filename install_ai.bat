@echo off
chcp 65001 >nul
title No photoshop - 포커싱 탭 AI 설치
echo ============================================================
echo  No photoshop 포커싱 탭 AI 구성요소 설치
echo  (NVIDIA GPU 권장 · 인터넷 필요 · 약 15~30분, ~6GB)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [1/3] 파이썬이 없어 설치합니다 ^(winget^) ...
    winget install -e --id Python.Python.3.10 --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo.
        echo   winget 설치 실패. https://www.python.org/downloads/ 에서
        echo   Python 3.10+ 를 설치할 때 "Add to PATH"를 체크한 뒤 이 파일을 다시 실행하세요.
        pause & exit /b 1
    )
    set "PATH=%LOCALAPPDATA%\Programs\Python\Python310;%LOCALAPPDATA%\Programs\Python\Python310\Scripts;%PATH%"
) else (
    echo [1/3] 파이썬 확인됨
)

echo [2/3] PyTorch ^(CUDA 12.8^) 설치 중 ... ^(가장 오래 걸립니다^)
python -m pip install --upgrade pip -q
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo   CUDA 버전 설치 실패 - CPU 버전으로 대체합니다 ^(느림^)
    python -m pip install torch torchvision
)

echo [3/3] AI 라이브러리 설치 중 ...
python -m pip install transformers huggingface_hub opencv-python numpy pillow einops easydict tqdm

echo.
python -c "import torch; print('설치 완료  torch', torch.__version__, '  CUDA:', torch.cuda.is_available())"
echo.
echo 완료! 이제 No photoshop.exe 를 실행하고 포커싱 탭에 영상을 넣으세요.
echo (첫 영상 로드 시 AI 모델 ~2GB가 자동 다운로드됩니다 - 한 번만)
pause
