# STiTy SimulStreaming 아키텍처 설명

> **Note**: A팀(AlignAtt/CIF), B팀(LLM 번역/VAD) 발표 내용은 제외하고 나머지 시스템 구조만 설명합니다.

---

## 전체 흐름 요약

```
사용자가 말함
→ 마이크로 녹음 (48kHz)
→ 클라이언트에서 16kHz로 변환
→ WebSocket으로 서버에 전송
→ [A팀] STT 처리
→ [B팀] 번역 처리
→ 결과를 다시 클라이언트로 전송
→ 화면에 자막 표시
```

---

## Layer 1: 프론트엔드 (Electron)

### 1. Presentation Layer (UI)
- **역할**: 화면에 자막 표시
- **주요 기능**:
  - 투명한 창으로 항상 위에 표시 (Always-on-Top)
  - 원문과 번역을 동시에 또는 선택적으로 표시
  - 그라데이션 바로 상태 표시 (파란색=정상, 빨간색=오류)
  - 설정 창에서 투명도, 텍스트 크기, 서버 URL 조정

### 2. Audio Capture Layer
- **역할**: 마이크로 음성 녹음 및 전처리
- **처리 과정**:
  1. `navigator.mediaDevices`로 마이크 접근 (48kHz, 모노)
  2. `AudioContext` + `ScriptProcessorNode`로 실시간 오디오 캡처
  3. **48kHz → 16kHz 다운샘플링** (Whisper는 16kHz 필요)
  4. **Float32 → Int16 변환** (메모리 절약)
  5. 500ms 청크(8000 샘플)씩 버퍼링 후 전송

- **왜 클라이언트에서 변환?**
  - 서버 부담 감소
  - 네트워크 대역폭 절약 (48kHz보다 3배 작음)
  - FFmpeg 같은 무거운 툴 불필요

### 3. Communication Layer
- **역할**: 서버와 실시간 통신
- **WebSocket 사용**:
  - **클라이언트 → 서버**: Binary (PCM Int16 오디오 데이터)
  - **서버 → 클라이언트**: JSON (결과 메시지)
- **자동 재연결**: 서버 끊기면 자동으로 재접속 시도

---

## WebSocket 통신

### 클라이언트가 서버에 보내는 것

**1. 제어 메시지 (JSON)**
```json
// 시작
{"type": "start", "lang": "auto", "polish": true, "translate": true}

// 종료
{"type": "stop"}
```

**2. 오디오 데이터 (Binary)**
- 형식: PCM Int16 (16kHz, 모노)
- 크기: 16KB (8000 샘플 × 2바이트)
- 주기: 500ms마다

### 서버가 클라이언트에 보내는 것

**JSON 메시지만**
```json
// 연결 환영
{"type": "hello", "message": "Connected to Whisper Streaming Server"}

// 준비 완료
{"type": "ready", "message": "Ready to receive audio"}

// 최종 결과
{
  "type": "final",
  "start": 1234,           // 시작 시간 (ms)
  "end": 5678,             // 종료 시간 (ms)
  "original": "원본 텍스트",
  "language": "ko",
  "ko": "한국어 텍스트",    // 원문 또는 번역
  "en": "English text"     // 원문 또는 번역
}
```

---

## Layer 2: 백엔드 (Python WebSocket Server)

### 1. WebSocket Server
- **역할**: 클라이언트 연결 관리
- **설정**:
  - `ping_interval=60초`: 연결 유지 (NAT 타임아웃 방지)
  - `max_size=10MB`: 큰 오디오도 수용
  - 클라이언트마다 독립적인 세션

### 2. Audio Processing Pipeline
- **역할**: 오디오 데이터 변환 및 버퍼링
- **처리 과정**:
  1. Binary 메시지 수신 (PCM Int16)
  2. `numpy`로 변환: `Int16 → Float32` (Whisper 입력 형식)
  3. 버퍼에 누적 (최소 청크 크기까지)
  4. **[A팀 영역]** `online_asr_proc.insert_audio_chunk(audio)` 호출

- **FFmpeg 없음!**:
  - 기존 `server/` 폴더는 WebM → PCM 변환에 FFmpeg 사용
  - SimulStreaming은 **이미 PCM으로 받으므로 불필요**

### 3. STT & Translation Engine
- **[A팀 영역]** AlignAtt 기반 SimulWhisper
  - Cross-Attention 가중치 분석해서 실시간 디코딩
  - `process_iter()` 반복 호출해서 결과 생성

- **[B팀 영역]** Google Translator
  - 한국어 ↔ 영어 양방향 번역
  - 텍스트 분석으로 언어 자동 감지

### 4. WebSocket 응답
- **역할**: 결과를 JSON으로 클라이언트에 전송
- **포함 내용**:
  - 원문 텍스트
  - 번역 텍스트
  - 타임스탬프 (자막 싱크용)
  - 감지된 언어

---

## 핵심: AlignAtt 정책 (간단 요약)

**[A팀 발표 영역이지만 개념만]**

- **문제**: 실시간 스트리밍에서는 언제 텍스트를 출력해야 할지 결정이 어려움
- **해결**: Whisper의 Cross-Attention 가중치를 보고 "여기까지는 확실하다" 판단
- **장점**: 낮은 지연 + 높은 정확도 균형

---

## 주요 특징 (기존 server/와 차이점)

### 기존 server/ 폴더
```
WebM/Opus (클라이언트)
  ↓
FFmpeg 변환 (서버)
  ↓
PCM 16kHz
  ↓
VAD 버퍼링
  ↓
Whisper small
```

### SimulStreaming/ 폴더
```
PCM Int16 (클라이언트에서 이미 변환)
  ↓
numpy 변환만
  ↓
AlignAtt 실시간 디코딩
  ↓
Whisper 기반 (더 정교한 정책)
```

**차이점 요약**:
1. **클라이언트 처리**: 48→16kHz 변환을 클라이언트에서 (서버 부담 감소)
2. **FFmpeg 제거**: PCM 직접 전송으로 서버 단순화
3. **실시간 정책**: VAD 대신 AlignAtt로 더 정교한 타이밍 제어

---

## 데이터 흐름 타이밍

| 단계 | 시간 | 설명 |
|------|------|------|
| 오디오 캡처 | 85ms | ScriptProcessorNode 버퍼 |
| 클라이언트 버퍼링 | 500ms | 8000 샘플 모을 때까지 |
| 네트워크 전송 | 10-50ms | WebSocket (로컬/원격에 따라) |
| 서버 변환 | <5ms | numpy 변환 (매우 빠름) |
| **[A팀] STT** | **100-500ms** | **AlignAtt 디코딩** |
| **[B팀] 번역** | **50-200ms** | **Google API** |
| 응답 전송 | 10-50ms | JSON 반환 |
| UI 업데이트 | <10ms | 화면 표시 |
| **총 지연** | **0.8-2.5초** | **말 끝나고 자막까지** |

---

## 설정 및 상태 관리

### 사용자 설정 (settings.html)
- **서버 URL**: WebSocket 서버 주소
- **표시 모드**: 번역만 / 전사만 / 둘 다
- **투명도**: 0-100%
- **자동 투명도**: 배경 밝기에 따라 자동 조정
- **텍스트 크기**: 0-200%

### 상태 동기화
- **메인 창 ↔ 설정 창**: IPC 통신으로 실시간 동기화
- **저장 위치**: `localStorage` (브라우저 로컬 저장소)
- **로그 기록**: 최근 50개 번역 기록 저장

---

## 에러 처리

### 클라이언트
- **마이크 접근 거부**: 명확한 에러 메시지 + 해결 방법 안내
- **서버 연결 실패**: 자동 재연결 시도 (최대 3초)
- **WebSocket 끊김**: 상태바 빨간색으로 변경 + 재연결

### 서버
- **빈 오디오**: 조용히 건너뛰기 (경고 로그만)
- **변환 실패**: 에러 로그 출력 + 다음 청크 계속 처리
- **번역 실패**: 원문만 반환 (Graceful degradation)

---

## 성능 최적화

### 클라이언트
- **다운샘플링**: Nearest Neighbor (간단하지만 Whisper엔 충분)
- **버퍼 병합**: `TypedArray.set()` 사용 (네이티브 구현, 매우 빠름)
- **UI 업데이트**: `textContent` 사용 (innerHTML보다 빠름)

### 서버
- **비동기 처리**: `asyncio`로 여러 클라이언트 동시 처리
- **NumPy 벡터화**: C 레벨 최적화 (Python 루프보다 100배 빠름)
- **Keep-alive**: 60초 ping으로 연결 유지

---

## 요약

1. **프론트엔드**: 마이크 → 16kHz 변환 → WebSocket 전송
2. **WebSocket**: Binary(오디오) 업로드 / JSON(결과) 다운로드
3. **백엔드**: PCM 수신 → [A팀 STT] → [B팀 번역] → 응답
4. **핵심**: 클라이언트 전처리로 서버 단순화 + AlignAtt로 낮은 지연

**전체 레이턴시**: 발화 종료 후 약 1-2초 내 자막 표시
