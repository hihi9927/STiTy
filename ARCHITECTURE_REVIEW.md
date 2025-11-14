# STiTy SimulStreaming 아키텍처 리뷰 문서

> **Note**: 본 문서는 A팀(AlignAtt/CIF 핵심 알고리즘)과 B팀(LLM 번역/VAD) 리뷰 영역을 **제외**한 나머지 시스템 구조를 설명합니다.

---

## 목차
1. [전체 시스템 개요](#1-전체-시스템-개요)
2. [Layer 1: 프론트엔드 (Electron)](#2-layer-1-프론트엔드-electron)
3. [Layer 2: 백엔드 - 통신 계층](#3-layer-2-백엔드---통신-계층)
4. [데이터 흐름 및 프로토콜](#4-데이터-흐름-및-프로토콜)
5. [설정 및 상태 관리](#5-설정-및-상태-관리)
6. [성능 최적화 전략](#6-성능-최적화-전략)
7. [에러 처리 및 복원력](#7-에러-처리-및-복원력)

---

## 1. 전체 시스템 개요

### 1.1 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 (User)                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ 음성 입력
                       ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: 프론트엔드 (Electron App)                          │
│  - Presentation Layer (UI)                                   │
│  - Audio Capture Layer (48kHz → 16kHz 변환)                │
│  - Communication Layer (WebSocket Client)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │ PCM Int16 Binary (16kHz)
                       ↓ (WebSocket)
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: 백엔드 (Python WebSocket Server)                   │
│  - WebSocket Handler                                         │
│  - Audio Processing Pipeline                                 │
│  - [A팀] AlignAtt STT Engine ← 제외                         │
│  - [B팀] Translation Module ← 제외                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ JSON Response
                       ↓
                    화면 표시
```

### 1.2 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| **프론트엔드** | Electron + Node.js | 데스크톱 앱 프레임워크 |
| | Web Audio API | 오디오 캡처 및 처리 |
| | WebSocket (WSS) | 실시간 양방향 통신 |
| **백엔드** | Python 3.x | 서버 런타임 |
| | websockets 라이브러리 | WebSocket 서버 |
| | asyncio | 비동기 처리 |
| | numpy | 오디오 데이터 처리 |

---

## 2. Layer 1: 프론트엔드 (Electron)

### 2.1 Presentation Layer (UI)

**파일**: `SimulStreaming/app/index.html`, `app.js`, `styles.css`

#### 2.1.1 UI 컴포넌트 구조

```html
<!-- 메인 자막 패널 -->
<div class="panel" id="mainPanel">
  <div id="current">
    <!-- 원문 표시 영역 -->
    <div class="text-line original" id="currTextOriginal"></div>

    <!-- 번역 표시 영역 -->
    <div class="text-line translated" id="currTextTranslated"></div>

    <!-- 타이핑 커서 -->
    <span class="caret"></span>
  </div>

  <!-- 상태 표시 그라데이션 바 -->
  <div class="rule"></div>
</div>
```

#### 2.1.2 시각적 특징

| 기능 | 구현 | 코드 위치 |
|------|------|-----------|
| **투명 윈도우** | `transparent: true`, `frame: false` | main.js:21-25 |
| **Always-on-Top** | `alwaysOnTop: true` | main.js:26 |
| **드래그 가능** | Long-press (300ms) 감지 → 창 이동 | app.js:661-682 |
| **자동 투명도** | desktopCapturer로 배경 밝기 감지 | app.js:509-605 |
| **상태 표시** | 그라데이션 바 색상 (파란색/빨간색) | app.js:30-47 |

#### 2.1.3 표시 모드 (Display Mode)

```javascript
// app.js:144-156
if (state.displayMode === 'translateOnly') {
  // 번역만 표시
  DOM.currTextOriginal.textContent = trans || orig;
  DOM.currTextTranslated.textContent = '';
} else if (state.displayMode === 'transcriptOnly') {
  // 전사만 표시
  DOM.currTextOriginal.textContent = orig;
  DOM.currTextTranslated.textContent = '';
} else {
  // 번역+전사 (기본)
  DOM.currTextOriginal.textContent = orig;
  DOM.currTextTranslated.textContent = trans;
}
```

**설정 위치**: `settings.html` → localStorage 저장

---

### 2.2 Audio Capture Layer

**파일**: `SimulStreaming/app/app.js` (라인 282-402)

#### 2.2.1 오디오 캡처 파이프라인

```
Microphone (navigator.mediaDevices)
    ↓ 48kHz, Mono, Float32
AudioContext.createMediaStreamSource()
    ↓
ScriptProcessorNode (bufferSize: 4096)
    ↓ onaudioprocess 이벤트
[다운샘플링] 48kHz → 16kHz
    ↓ ratio = 3:1
[양자화] Float32 (-1.0~1.0) → Int16 (-32768~32767)
    ↓
Buffer 누적 (8000 샘플 = 500ms)
    ↓
WebSocket.send(Int16Array.buffer)
```

#### 2.2.2 주요 코드 분석

**마이크 접근 및 초기화**
```javascript
// app.js:288-296
state.mediaStream = await navigator.mediaDevices.getUserMedia({
  audio: {
    channelCount: 1,              // 모노
    sampleRate: 48000,            // 48kHz (브라우저 표준)
    echoCancellation: true,       // 에코 제거
    noiseSuppression: true,       // 노이즈 억제
    autoGainControl: true         // 자동 게인 조절
  }
});
```

**AudioContext 생성**
```javascript
// app.js:327-329
state.audioContext = new (window.AudioContext || window.webkitAudioContext)({
  sampleRate: 48000
});
```

**ScriptProcessorNode를 사용한 실시간 처리**
```javascript
// app.js:334-335
const bufferSize = 4096;  // 약 85ms @ 48kHz
const processor = state.audioContext.createScriptProcessor(bufferSize, 1, 1);
```

> **Note**: ScriptProcessorNode는 deprecated이지만 Electron 환경에서의 안정성을 위해 사용.
> AudioWorklet은 보안 컨텍스트 및 CORS 이슈로 Electron에서 까다로울 수 있음.

**다운샘플링 (48kHz → 16kHz)**
```javascript
// app.js:348-358
const targetSampleRate = 16000;
const sourceSampleRate = state.audioContext.sampleRate;  // 48000
const ratio = sourceSampleRate / targetSampleRate;       // 3.0
const newLength = Math.floor(inputData.length / ratio);
const downsampled = new Float32Array(newLength);

for (let i = 0; i < newLength; i++) {
  const srcIndex = Math.floor(i * ratio);
  downsampled[i] = inputData[srcIndex];  // 단순 샘플링 (nearest neighbor)
}
```

**Float32 → Int16 변환**
```javascript
// app.js:361-365
const pcmData = new Int16Array(downsampled.length);
for (let i = 0; i < downsampled.length; i++) {
  const s = Math.max(-1, Math.min(1, downsampled[i]));  // 클리핑
  pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
}
```

**500ms 청크 버퍼링**
```javascript
// app.js:368-388
const targetSamplesPerChunk = 8000;  // 16kHz * 0.5초 = 8000 샘플
state.audioBuffer.push(pcmData);

const totalSamples = state.audioBuffer.reduce((sum, arr) => sum + arr.length, 0);
if (totalSamples >= targetSamplesPerChunk) {
  // 모든 버퍼를 하나로 합치기
  const combinedBuffer = new Int16Array(totalSamples);
  let offset = 0;
  for (const buf of state.audioBuffer) {
    combinedBuffer.set(buf, offset);
    offset += buf.length;
  }

  // WebSocket 전송
  state.ws.send(combinedBuffer.buffer);

  // 버퍼 초기화
  state.audioBuffer = [];
}
```

#### 2.2.3 성능 특성

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| **샘플링 레이트** | 48kHz → 16kHz | 브라우저 표준 → Whisper 입력 |
| **비트 깊이** | Float32 → Int16 | 32bit → 16bit (메모리 절반) |
| **버퍼 크기** | 4096 샘플 | 약 85ms 처리 단위 |
| **전송 주기** | 500ms | 8000 샘플 = 16KB 전송 |
| **레이턴시** | ~100ms | 캡처 + 버퍼링 지연 |

---

### 2.3 Communication Layer

**파일**: `SimulStreaming/app/app.js` (라인 198-280)

#### 2.3.1 WebSocket 생명주기

```
[Disconnected] ─┬─ connectWebSocket() ──→ [Connecting]
                │                             │
                │                         ws.onopen
                │                             ↓
                │                         [Connected] ─┬─ send Binary
                │                             ↑        │  (PCM data)
                │                             │        │
                │                        ws.onmessage ←┘
                │                             │
                │                        (JSON 처리)
                │                             │
                ├─ ws.onerror ────────────────┤
                │                             │
                └─ ws.onclose ←───────────────┘
                        │
                    (재연결 대기)
```

#### 2.3.2 WebSocket 연결 설정

```javascript
// app.js:214-215
state.ws = new WebSocket(state.SERVER_URL);
state.ws.binaryType = 'arraybuffer';  // Binary 데이터를 ArrayBuffer로 수신
```

#### 2.3.3 연결 수립 시퀀스

```javascript
// app.js:217-229
state.ws.onopen = () => {
  state.isConnecting = false;
  updateServerStatus(true);  // UI 업데이트

  // 서버에 시작 명령 전송
  const startMsg = {
    type: 'start',
    lang: 'auto',       // 언어 자동 감지
    polish: true,       // 문장 다듬기 활성화
    translate: true     // 번역 활성화
  };
  state.ws.send(JSON.stringify(startMsg));
};
```

#### 2.3.4 메시지 수신 처리

```javascript
// app.js:231-263
state.ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  const t = data.type;

  if (t === 'partial_cumulative') {
    // 중간 결과 (현재 사용 안 함)
    const original = data.polished || data.original || '';
    if (original) {
      showResult(original, '');
    }
  } else if (t === 'final') {
    // 최종 결과
    let original = '';
    let translated = '';

    // 언어에 따라 원문과 번역 구분
    if (data.language === 'ko' || data.language === 'Korean') {
      // 한국어 원문 → 영어 번역
      original = data.ko || data.polished || data.original || '';
      translated = data.en || '';
    } else {
      // 영어 원문 → 한국어 번역
      original = data.en || data.polished || data.original || '';
      translated = data.ko || '';
    }

    if (original) {
      showResult(original, translated);
    }
  }
};
```

#### 2.3.5 에러 및 종료 처리

```javascript
// app.js:265-274
state.ws.onerror = (error) => {
  state.isConnecting = false;
  updateServerStatus(false);
};

state.ws.onclose = (event) => {
  state.isConnecting = false;
  updateServerStatus(false);
  state.ws = null;
  // 자동 재연결은 startRecording()에서 처리
};
```

#### 2.3.6 재연결 메커니즘

```javascript
// app.js:309-322 (startRecording 내부)
await connectWebSocket();

// WebSocket이 열릴 때까지 대기 (최대 3초)
let retries = 0;
while ((!state.ws || state.ws.readyState !== WebSocket.OPEN) && retries < 10) {
  await new Promise(resolve => setTimeout(resolve, 300));
  retries++;
}

if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
  alert('서버 연결에 실패했습니다. 서버 URL을 확인해주세요.');
  return;
}
```

---

## 3. Layer 2: 백엔드 - 통신 계층

**파일**: `SimulStreaming/whisper_streaming/whisper_websocket_server.py`

> **Note**: AlignAtt STT 엔진(A팀)과 Translation 모듈(B팀) 제외

### 3.1 WebSocket Server Handler

#### 3.1.1 서버 초기화

```python
# whisper_websocket_server.py:234-301
def main_websocket_server(factory, add_args):
    """
    Main WebSocket server entry point

    factory: ASR 및 online processor 객체를 생성하는 함수
    add_args: 백엔드별 추가 인자 설정 함수
    """
    import websockets

    parser = argparse.ArgumentParser()

    # 서버 옵션
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
        help="Whisper warm-up용 오디오 파일 경로")

    args = parser.parse_args()

    # ASR 및 online processor 생성
    asr, online = asr_factory(args, factory)
    min_chunk = args.vac_chunk_size if args.vac else args.min_chunk_size
```

#### 3.1.2 WebSocket 서버 설정

```python
# whisper_websocket_server.py:286-299
async with websockets.serve(
    server_handler,
    args.host,
    args.port,
    ping_interval=60,               # 60초마다 ping 전송 (연결 유지)
    ping_timeout=120,               # 120초 타임아웃
    max_size=10 * 1024 * 1024,      # 10MB 최대 메시지 크기
    close_timeout=10                # 연결 종료 타임아웃 10초
):
    logger.info(f'WebSocket server listening on ws://{args.host}:{args.port}')
    await asyncio.Future()  # 무한 실행
```

**설정 의미**:
- `ping_interval=60`: 클라이언트가 오랫동안 조용해도 연결 유지 (NAT 타임아웃 방지)
- `max_size=10MB`: 큰 오디오 청크도 수용 (10초 오디오 ≈ 320KB)
- `close_timeout=10`: 종료 시 최대 10초 대기

---

### 3.2 WebSocket Handler 클래스

**파일**: `SimulStreaming/whisper_streaming/whisper_websocket_server.py` (라인 18-226)

#### 3.2.1 클래스 구조

```python
class WebSocketHandler:
    """Handles WebSocket connection and ASR processing for one client"""

    def __init__(self, websocket, online_asr_proc, min_chunk):
        self.websocket = websocket          # WebSocket 연결 객체
        self.online_asr_proc = online_asr_proc  # [A팀] ASR 프로세서
        self.min_chunk = min_chunk          # 최소 청크 크기 (초)
        self.is_first = True                # 첫 번째 청크 여부
        self.audio_buffer = []              # 오디오 버퍼
        self.running = False                # 실행 상태
```

#### 3.2.2 메시지 전송

```python
# whisper_websocket_server.py:29-34
async def send_message(self, message_dict):
    """Send JSON message to client"""
    try:
        await self.websocket.send(json.dumps(message_dict))
    except Exception as e:
        logger.error(f"Error sending message: {e}")
```

**전송 메시지 타입**:
1. `hello`: 연결 환영 메시지
2. `ready`: 서버 준비 완료
3. `final`: 최종 전사 결과 (원문 + 번역)
4. `error`: 에러 메시지

#### 3.2.3 연결 수립 시퀀스

```python
# whisper_websocket_server.py:178-226
async def handle(self):
    """Main handler for WebSocket connection"""
    try:
        logger.info(f"New WebSocket connection from {self.websocket.remote_address}")

        # 1. Hello 메시지 전송
        await self.send_message({
            'type': 'hello',
            'message': 'Connected to Whisper Streaming Server'
        })

        # 2. ASR 초기화
        self.online_asr_proc.init()
        self.running = True

        # 3. 메시지 수신 루프
        async for message in self.websocket:
            if isinstance(message, bytes):
                # Binary 데이터 = 오디오
                if self.running:
                    await self.process_audio_chunk(message)
            else:
                # Text 데이터 = JSON 명령
                data = json.loads(message)
                msg_type = data.get('type', '')

                if msg_type == 'start':
                    logger.info("Received start command")
                    await self.send_message({
                        'type': 'ready',
                        'message': 'Ready to receive audio'
                    })

                elif msg_type == 'stop':
                    logger.info("Received stop command")
                    self.running = False
                    break

    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        logger.info("WebSocket connection closed")
```

---

### 3.3 Audio Processing Pipeline

**파일**: `SimulStreaming/whisper_streaming/whisper_websocket_server.py` (라인 123-176)

#### 3.3.1 PCM 변환

```python
# whisper_websocket_server.py:123-138
def convert_pcm_to_float(self, pcm_data):
    """Convert RAW PCM Int16 data to float32 numpy array"""
    try:
        # PCM 데이터는 클라이언트에서 이미 16kHz Int16 형식
        # bytes → Int16 array 변환
        audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

        # Int16 → Float32 변환 (-1.0 to 1.0)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        logger.debug(f"Converted PCM: {len(audio_float)} samples")
        return audio_float

    except Exception as e:
        logger.error(f"Error converting PCM: {e}")
        return None
```

**변환 과정**:
1. `np.frombuffer`: bytes → Int16 array
2. `astype(np.float32)`: Int16 → Float32
3. `/ 32768.0`: [-32768, 32767] → [-1.0, 1.0] 정규화

#### 3.3.2 오디오 청크 처리

```python
# whisper_websocket_server.py:140-176
async def process_audio_chunk(self, audio_data):
    """Process incoming audio data"""
    try:
        # 1. PCM → Float32 변환
        audio = self.convert_pcm_to_float(audio_data)

        if audio is None or len(audio) == 0:
            logger.warning("Failed to convert audio or empty audio")
            return

        logger.debug(f"Received audio chunk: {len(audio)} samples")

        # 2. 버퍼에 추가
        self.audio_buffer.append(audio)

        # 3. 충분한 데이터가 모이면 처리
        total_samples = sum(len(x) for x in self.audio_buffer)
        min_samples = self.min_chunk * SAMPLING_RATE  # min_chunk(초) * 16000

        if total_samples >= min_samples or not self.is_first:
            # 4. 모든 버퍼를 하나로 병합
            if self.audio_buffer:
                audio_chunk = np.concatenate(self.audio_buffer)
                self.audio_buffer = []
                self.is_first = False

                # 5. [A팀 영역] ASR에 오디오 삽입
                self.online_asr_proc.insert_audio_chunk(audio_chunk)

                # 6. [A팀 영역] 처리 및 결과 반환
                result = self.online_asr_proc.process_iter()

                # 7. 결과 전송 (번역 포함)
                await self.send_result(result)

    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
```

**버퍼링 전략**:
- `is_first=True`: 첫 번째는 최소 청크 크기까지 대기 (워밍업)
- `is_first=False`: 이후는 매 청크마다 처리 (낮은 레이턴시)

#### 3.3.3 버퍼 크기 계산

```python
# 예시: min_chunk = 1.0초
min_samples = 1.0 * 16000 = 16000 샘플
                          = 32000 바이트 (Int16)
                          = 1초 오디오
```

---

### 3.4 결과 전송

**파일**: `SimulStreaming/whisper_streaming/whisper_websocket_server.py` (라인 81-122)

> **Note**: `detect_and_translate()` 함수는 B팀 영역이므로 제외

#### 3.4.1 결과 메시지 구성

```python
# whisper_websocket_server.py:81-122
async def send_result(self, iteration_output):
    """Send transcription result to client"""
    if iteration_output:
        start_ms = int(iteration_output['start'] * 1000)
        end_ms = int(iteration_output['end'] * 1000)
        text = iteration_output['text'].strip()

        if not text:
            logger.debug("Empty text in segment")
            return

        message = f"{start_ms} {end_ms} {text}"
        print(message, flush=True, file=sys.stderr)

        # [B팀 영역] 언어 감지 및 번역
        detected_lang = iteration_output.get('language', 'en')
        lang, ko_text, en_text = self.detect_and_translate(text, detected_lang)

        # 결과 메시지 구성
        result_msg = {
            'type': 'final',
            'start': start_ms,           # 시작 시간 (밀리초)
            'end': end_ms,               # 종료 시간 (밀리초)
            'original': text,            # 원본 텍스트
            'polished': text,            # 다듬어진 텍스트
            'language': lang             # 감지된 언어 (ko/en)
        }

        # 번역 결과 추가
        if lang == 'ko':
            # 한국어 원문 → 영어 번역
            result_msg['ko'] = text      # 원문
            result_msg['en'] = en_text if en_text else ""  # 번역
        else:
            # 영어 원문 → 한국어 번역
            result_msg['en'] = text      # 원문
            result_msg['ko'] = ko_text if ko_text else ""  # 번역

        # 클라이언트로 전송
        await self.send_message(result_msg)
    else:
        logger.debug("No text in this segment")
```

#### 3.4.2 응답 메시지 형식

```json
{
  "type": "final",
  "start": 1234,              // 밀리초
  "end": 5678,                // 밀리초
  "original": "Hello world",  // 원본 텍스트
  "polished": "Hello world",  // 다듬어진 텍스트
  "language": "en",           // ko 또는 en
  "ko": "안녕하세요",         // 한국어 텍스트 (원문 또는 번역)
  "en": "Hello world"         // 영어 텍스트 (원문 또는 번역)
}
```

**필드 설명**:
- `start`, `end`: 오디오 세그먼트의 타임스탬프 (자막 동기화용)
- `original`: Whisper가 직접 출력한 원본 텍스트
- `polished`: 후처리된 텍스트 (현재는 동일)
- `language`: 감지된 언어 코드
- `ko`, `en`: 각 언어의 텍스트 (원문 또는 번역)

---

## 4. 데이터 흐름 및 프로토콜

### 4.1 전체 데이터 흐름

```
[사용자 발화]
    ↓
[마이크 캡처] 48kHz Float32
    ↓ (85ms 버퍼)
[다운샘플링] 48kHz → 16kHz
    ↓
[양자화] Float32 → Int16
    ↓ (500ms 버퍼)
[WebSocket 전송] Binary (16KB)
    ↓ (네트워크 ~10-50ms)
[서버 수신] bytes
    ↓
[변환] Int16 → Float32
    ↓
[버퍼링] min_chunk까지 누적
    ↓
[A팀] AlignAtt STT
    ↓
[B팀] 번역
    ↓
[WebSocket 응답] JSON (~1-2KB)
    ↓ (네트워크 ~10-50ms)
[클라이언트 수신]
    ↓
[UI 업데이트] DOM 조작
    ↓
[화면 표시]
```

### 4.2 프로토콜 상세

#### 4.2.1 클라이언트 → 서버

**메시지 타입 1: 제어 명령 (JSON Text)**

```json
// 1. 시작 명령
{
  "type": "start",
  "lang": "auto",       // 언어 힌트 (auto, ko, en)
  "polish": true,       // 문장 다듬기
  "translate": true     // 번역 활성화
}

// 2. 종료 명령
{
  "type": "stop"
}
```

**메시지 타입 2: 오디오 데이터 (Binary)**

```
ArrayBuffer (Int16)
├─ 크기: 16KB (8000 샘플 × 2바이트)
├─ 샘플링 레이트: 16kHz
├─ 채널: 1 (모노)
├─ 인코딩: PCM Int16 Little-Endian
└─ 재생 시간: 500ms
```

#### 4.2.2 서버 → 클라이언트

**메시지 타입: JSON Text만**

```json
// 1. 환영 메시지
{
  "type": "hello",
  "message": "Connected to Whisper Streaming Server"
}

// 2. 준비 완료
{
  "type": "ready",
  "message": "Ready to receive audio"
}

// 3. 최종 결과
{
  "type": "final",
  "start": 1234,
  "end": 5678,
  "original": "원본 텍스트",
  "polished": "다듬어진 텍스트",
  "language": "ko",
  "ko": "한국어 텍스트",
  "en": "English text"
}

// 4. 에러
{
  "type": "error",
  "message": "오류 메시지"
}
```

### 4.3 타이밍 분석

| 단계 | 시간 | 설명 |
|------|------|------|
| 오디오 캡처 | 85ms | ScriptProcessorNode 버퍼 |
| 클라이언트 버퍼링 | 500ms | 8000 샘플 누적 |
| WebSocket 전송 | 10-50ms | 네트워크 지연 (로컬/원격) |
| 서버 변환 | <5ms | Int16 → Float32 (numpy) |
| 서버 버퍼링 | 0-1000ms | min_chunk 대기 (첫 번째만) |
| **[A팀] STT 추론** | **100-500ms** | **AlignAtt 디코딩** |
| **[B팀] 번역** | **50-200ms** | **Google API 호출** |
| WebSocket 응답 | 10-50ms | 네트워크 지연 |
| UI 업데이트 | <10ms | DOM 조작 |
| **총 레이턴시** | **0.8-2.5초** | **발화 종료 후 표시까지** |

---

## 5. 설정 및 상태 관리

### 5.1 설정 창 (Settings Window)

**파일**: `SimulStreaming/app/settings.html`, `settings.js`

#### 5.1.1 IPC 통신 구조

```
┌─────────────────────┐
│   Main Window       │
│   (app.js)          │
└──────────┬──────────┘
           │ IPC
           ↓
┌─────────────────────┐
│   Main Process      │
│   (main.js)         │
└──────────┬──────────┘
           │ IPC
           ↓
┌─────────────────────┐
│   Settings Window   │
│   (settings.js)     │
└─────────────────────┘
```

#### 5.1.2 설정 항목

**파일**: `SimulStreaming/app/main.js` (라인 183-262)

```javascript
// 설정 창 생성
ipcMain.on('open-settings-window', () => {
  if (settingsWindow && !settingsWindow.isDestroyed()) {
    settingsWindow.show();
    return;
  }

  settingsWindow = new BrowserWindow({
    width: 600,
    height: 800,
    resizable: false,
    minimizable: false,
    maximizable: false,
    alwaysOnTop: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  settingsWindow.loadFile('settings.html');
});
```

**설정 변경 이벤트**:

```javascript
// main.js:264-313
ipcMain.on('opacity-changed', (event, value) => {
  if (mainWindow) {
    mainWindow.webContents.send('opacity-changed', value);
  }
});

ipcMain.on('text-size-changed', (event, value) => {
  if (mainWindow) {
    mainWindow.webContents.send('text-size-changed', value);
  }
});

ipcMain.on('server-url-changed', (event, value) => {
  if (mainWindow) {
    mainWindow.webContents.send('server-url-changed', value);
  }
});

ipcMain.on('display-mode-changed', (event, value) => {
  if (mainWindow) {
    mainWindow.webContents.send('display-mode-changed', value);
  }
});

ipcMain.on('auto-opacity-changed', (event, value) => {
  if (mainWindow) {
    mainWindow.webContents.send('auto-opacity-changed', value);
  }
});
```

#### 5.1.3 LocalStorage 저장

```javascript
// app.js:87-118
function updateOpacity(value, autoAdjust = false) {
  const transparency = value / 100;
  let bgOpacity = 0.95 * (1 - transparency);

  // 자동 조정 로직...

  DOM.mainPanel.style.background = `rgba(20,20,20,${bgOpacity})`;
  localStorage.setItem('panelOpacity', value);  // 영구 저장
}

function updateTextSize(value) {
  const scale = value / 100;
  const originalSize = 20;
  const translatedOriginalSize = 16;

  DOM.currTextOriginal.style.fontSize = `${originalSize * scale}px`;
  DOM.currTextTranslated.style.fontSize = `${translatedOriginalSize * scale}px`;
  localStorage.setItem('textSize', value);  // 영구 저장
}
```

**저장되는 설정**:
- `panelOpacity`: 투명도 (0-100)
- `textSize`: 텍스트 크기 (0-200%)
- `serverUrl`: 서버 URL
- `displayMode`: 표시 모드 (both/translateOnly/transcriptOnly)
- `autoAdjustOpacity`: 자동 투명도 조정 (true/false)

---

### 5.2 상태 동기화

#### 5.2.1 상태 전파 메커니즘

```javascript
// app.js:49-85
function updateServerStatus(connected) {
  state.isServerConnected = connected;
  updateGradientBar();  // UI 업데이트

  // 설정 창으로 상태 전달
  if (window.require) {
    const { ipcRenderer } = window.require('electron');
    ipcRenderer.send('status-update', 'server', connected);
  }
}

function updateRecordingStatus(recording) {
  state.isRecording = recording;
  updateGradientBar();  // UI 업데이트

  // 설정 창으로 상태 전달
  if (window.require) {
    const { ipcRenderer } = window.require('electron');
    ipcRenderer.send('status-update', 'recording', recording);
  }
}
```

#### 5.2.2 로그 히스토리 관리

```javascript
// app.js:165-185
function addToLog(text) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });

  state.translationHistory.unshift({ time: timeStr, text });

  // 최대 50개 유지
  if (state.translationHistory.length > 50) {
    state.translationHistory.pop();
  }

  // 설정 창으로 전달
  if (window.require) {
    const { ipcRenderer } = window.require('electron');
    ipcRenderer.send('status-update', 'translation', state.translationHistory);
  }
}
```

---

## 6. 성능 최적화 전략

### 6.1 클라이언트 최적화

#### 6.1.1 오디오 처리 최적화

**다운샘플링 알고리즘**:
```javascript
// app.js:348-358
// Nearest Neighbor 샘플링 (간단하지만 효율적)
for (let i = 0; i < newLength; i++) {
  const srcIndex = Math.floor(i * ratio);
  downsampled[i] = inputData[srcIndex];
}
```

**장점**:
- CPU 사용량 최소화 (곱셈 + floor 연산만)
- 실시간 처리 가능
- 메모리 효율적

**단점**:
- Aliasing 발생 가능 (고주파 성분)
- Whisper는 강건하므로 문제 없음

#### 6.1.2 버퍼 관리

```javascript
// app.js:368-388
// 버퍼 병합 전략
const combinedBuffer = new Int16Array(totalSamples);
let offset = 0;
for (const buf of state.audioBuffer) {
  combinedBuffer.set(buf, offset);  // TypedArray.set (빠름)
  offset += buf.length;
}
```

**장점**:
- `TypedArray.set()`은 네이티브 구현 (매우 빠름)
- 메모리 재할당 최소화
- 복사 오버헤드 최소

#### 6.1.3 UI 업데이트 최적화

```javascript
// app.js:135-163
function showResult(original, translated) {
  // DOM 조작 최소화 (textContent만 변경)
  DOM.currTextOriginal.textContent = orig;
  DOM.currTextTranslated.textContent = trans;

  // CSS 클래스 토글 (리플로우 최소화)
  DOM.current.classList.add('typing');
}
```

**최적화 기법**:
- `textContent` 사용 (innerHTML보다 빠름)
- CSS 애니메이션 활용 (GPU 가속)
- 불필요한 리플로우 방지

---

### 6.2 서버 최적화

#### 6.2.1 비동기 처리

```python
# whisper_websocket_server.py:178-226
async def handle(self):
    """비동기 메시지 처리"""
    async for message in self.websocket:
        if isinstance(message, bytes):
            await self.process_audio_chunk(message)  # 비동기 처리
```

**장점**:
- 여러 클라이언트 동시 처리
- I/O 대기 시간 활용
- CPU 효율 극대화

#### 6.2.2 NumPy 벡터화

```python
# whisper_websocket_server.py:128-131
audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
audio_float = audio_int16.astype(np.float32) / 32768.0
```

**장점**:
- C 레벨 최적화 (SIMD)
- Python 루프보다 100배 빠름
- 메모리 효율적

#### 6.2.3 버퍼 병합 최적화

```python
# whisper_websocket_server.py:162-163
if self.audio_buffer:
    audio_chunk = np.concatenate(self.audio_buffer)  # NumPy concatenate
    self.audio_buffer = []
```

**장점**:
- `np.concatenate()`는 메모리 연속성 보장
- 캐시 효율 극대화
- Whisper 입력에 최적

---

### 6.3 네트워크 최적화

#### 6.3.1 WebSocket 설정

```python
# whisper_websocket_server.py:289-296
async with websockets.serve(
    server_handler,
    args.host,
    args.port,
    ping_interval=60,       # Keep-alive
    max_size=10 * 1024 * 1024,  # 10MB
):
```

**최적화 포인트**:
- `ping_interval=60`: NAT 타임아웃 방지
- `max_size=10MB`: 큰 청크도 수용
- `close_timeout=10`: 빠른 정리

#### 6.3.2 데이터 압축

**현재**: 압축 없음 (Raw PCM Int16)

**이유**:
- PCM은 압축률 낮음 (엔트로피 높음)
- 압축/해제 오버헤드 > 대역폭 절약
- 로컬 네트워크에서는 불필요

**개선 가능**:
- Opus 압축 (32kbps) → 4배 절약
- WebSocket 압축 (permessage-deflate)

---

## 7. 에러 처리 및 복원력

### 7.1 클라이언트 에러 처리

#### 7.1.1 WebSocket 에러

```javascript
// app.js:265-274
state.ws.onerror = (error) => {
  state.isConnecting = false;
  updateServerStatus(false);  // UI 업데이트
};

state.ws.onclose = (event) => {
  state.isConnecting = false;
  updateServerStatus(false);
  state.ws = null;
  // 재연결은 startRecording()에서 처리
};
```

**복원 전략**:
- 자동 재연결 (재녹음 시도 시)
- UI 피드백 (빨간색 상태바)
- 사용자 알림

#### 7.1.2 마이크 접근 실패

```javascript
// app.js:282-301
async function initAudioStream() {
  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { /* ... */ }
    });
  } catch (error) {
    alert('마이크 접근이 거부되었습니다.\n브라우저 설정에서 마이크 권한을 허용해주세요.');
    throw error;  // 상위로 전파
  }
}
```

**사용자 가이드**:
- 명확한 에러 메시지
- 해결 방법 안내
- 권한 재요청 가능

#### 7.1.3 연결 타임아웃

```javascript
// app.js:309-322
let retries = 0;
while ((!state.ws || state.ws.readyState !== WebSocket.OPEN) && retries < 10) {
  await new Promise(resolve => setTimeout(resolve, 300));
  retries++;
}

if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
  alert('서버 연결에 실패했습니다. 서버 URL을 확인해주세요.');
  return;
}
```

**타임아웃 설정**:
- 최대 3초 대기 (10회 × 300ms)
- 명확한 실패 알림
- 재시도 가능

---

### 7.2 서버 에러 처리

#### 7.2.1 예외 처리

```python
# whisper_websocket_server.py:140-176
async def process_audio_chunk(self, audio_data):
    try:
        # 처리 로직...
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        import traceback
        traceback.print_exc()  # 디버깅용 스택 트레이스
```

**에러 로깅**:
- 모든 예외 캐치 및 로그
- 스택 트레이스 출력
- 연결 유지 (치명적 에러 아님)

#### 7.2.2 연결 종료 처리

```python
# whisper_websocket_server.py:220-226
except Exception as e:
    logger.error(f"Error in WebSocket handler: {e}")
    import traceback
    traceback.print_exc()
finally:
    logger.info("WebSocket connection closed")
    # 리소스 정리 (ASR 프로세서는 공유되므로 유지)
```

**정리 작업**:
- 로그 기록
- 연결 종료 알림
- 리소스 해제 (필요 시)

#### 7.2.3 빈 오디오 처리

```python
# whisper_websocket_server.py:146-148
if audio is None or len(audio) == 0:
    logger.warning("Failed to convert audio or empty audio")
    return  # 조용히 건너뛰기
```

**전략**:
- 경고 로그만 출력
- 에러 전파 안 함 (클라이언트 혼란 방지)
- 다음 청크 대기

---

### 7.3 복원력 설계

#### 7.3.1 무상태(Stateless) 설계

**WebSocket Handler**:
- 각 연결마다 독립적인 핸들러 인스턴스
- 연결 종료 시 자동 정리
- 다른 클라이언트에 영향 없음

**장점**:
- 장애 격리
- 수평 확장 가능
- 간단한 재연결

#### 7.3.2 버퍼 오버플로우 방지

```python
# whisper_websocket_server.py:289-296
max_size=10 * 1024 * 1024  # 10MB 제한
```

**보호 메커니즘**:
- 최대 메시지 크기 제한
- 메모리 고갈 방지
- DoS 공격 완화

#### 7.3.3 Graceful Degradation

**클라이언트**:
- 서버 연결 실패 → UI는 계속 동작
- 번역 실패 → 원문만 표시
- 마이크 실패 → 명확한 에러 메시지

**서버**:
- 번역 실패 → 원문만 반환
- STT 실패 → 빈 결과 (건너뛰기)
- 부분 실패 → 다음 청크 계속 처리

---

## 결론

본 문서에서는 A팀(AlignAtt/CIF 핵심 알고리즘)과 B팀(LLM 번역/VAD)의 리뷰 영역을 제외한 **시스템 인프라 및 통신 계층**을 상세히 분석했습니다.

### 핵심 요약

1. **프론트엔드 (Electron)**:
   - Web Audio API 기반 실시간 오디오 캡처
   - 클라이언트 사이드 다운샘플링 (48kHz → 16kHz)
   - WebSocket을 통한 PCM Int16 직접 전송

2. **백엔드 (Python WebSocket Server)**:
   - 비동기 WebSocket 핸들러
   - NumPy 기반 효율적인 오디오 변환
   - JSON 기반 결과 전송 (원문 + 번역)

3. **통신 프로토콜**:
   - Binary: PCM Int16 (16kHz, 500ms 청크)
   - JSON: 제어 명령 및 결과
   - 양방향 실시간 스트리밍

4. **성능 최적화**:
   - 클라이언트: TypedArray 최적화, 버퍼 병합
   - 서버: NumPy 벡터화, 비동기 처리
   - 네트워크: Keep-alive, 적절한 버퍼 크기

5. **복원력**:
   - 자동 재연결
   - Graceful degradation
   - 명확한 에러 처리 및 사용자 피드백

### 개선 가능 영역

1. **오디오 품질**: Anti-aliasing 필터 추가 (현재는 Nearest Neighbor)
2. **네트워크 효율**: Opus 압축 또는 WebSocket 압축
3. **모니터링**: 레이턴시 측정 및 시각화
4. **보안**: WSS (TLS) 강제, 인증 메커니즘
5. **확장성**: 로드 밸런싱, 다중 서버 지원

---

**작성일**: 2025-01-XX
**버전**: 1.0
**대상**: SimulStreaming 아키텍처 리뷰

