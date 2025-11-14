# Phase 1: HTTP API 기반 STT 서버 테스트 가이드

## 📋 구현 현황

✅ **서버 코드**: [server.py](server.py) - Flask 기반 HTTP API (포트: 8000) + ngrok 이용 (다른 LAN에서도 접속 가능)
✅ **클라이언트 코드**: [client.py](client.py) - 음성 녹음 + 서버 전송
✅ **웹사이트 구동**: [client.html](client.html) - 음성 녹음 + 서버 전송
✅ **앱 실행**: [STT자막.exe](STT자막.exe) - 음성 녹음 + 서버 전송

## 🚀 실행 방법

### 1단계: 필수 패키지 설치

```bash
# 기본 패키지 (이미 설치되어 있음)
pip install torch openai-whisper librosa deep-translator

# 추가 패키지 (Flask, pyaudio)
pip install flask requests pyaudio

pip install websocketsm

# ffmpeg
choco install ffmpeg (window 환경)

conda install -c conda-forge ffmpeg (anaconda 가상환경)
```

**macOS에서 pyaudio 설치 시 에러 발생하면:**
```bash
brew install portaudio
pip install pyaudio
```

**Windows에서 pyaudio 설치 시 에러 발생하면:**
```bash
pip install pipwin
pipwin install pyaudio
```

---

### 2-1단계: 서버 실행 (팀 컴퓨터)

```bash
# 로컬 서버 시작
python server.py
```

**정상 실행 시 출력:**
```
🤖 Whisper 모델 로딩 중...
✅ 모델 로드 완료 (device=cpu)
============================================================
🚀 STT 서버 시작
============================================================
📡 접속 주소: http://0.0.0.0:8001
🔗 Health Check: http://0.0.0.0:8001/health
🎤 STT 엔드포인트: http://0.0.0.0:8001/stt
============================================================
 * Serving Flask app 'server'
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8001
 * Running on http://192.168.0.XXX:8001
```

### 2-2단계: ngrok 실행 (팀 컴퓨터)

```bash
# ngrok 서버 시작
ngrok http 8001
```

### 3단계: 클라이언트 실행 (다른 컴퓨터/휴대폰)

```bash
# client.py default 주소를 해당 주소로 설정
DEFAULT_WS = "wss://edra-raspiest-eagerly.ngrok-free.dev/ws"
```

```bash
** 실행 커맨드 ** 

1. 녹음 가능한 오디오 장치 찾기

window
ffmpeg -list_devices true -f dshow -i dummy

mac
ffmpeg -f avfoundation -list_devices true -i ""

2. python client.py --device "Headset Microphone(Oculus Virtual Audio Device)"
```

---
