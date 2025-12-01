# 새 환경 생성
conda create -n whisper python=3.10 -y

# 새 환경 활성화
conda activate whisper

# PyTorch GPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 패키지 설치
pip install openai-whisper
pip install fastapi 
pip install uvicorn 
pip install websockets 
pip install deep_translator

-- cmd에서 진행
winget install Gyan.FFmpeg
where ffmpeg

경로 복사

# whisper 환경으로 전환
conda activate whisper

# 환경 변수에 추가
$env:PATH += ";C:\Users\user\anaconda3\Library\bin" <- 예시 경로

++ pip install silero-vad
