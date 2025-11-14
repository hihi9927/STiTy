# server_ws_local.py (Silero VAD + 32KB Segment + Timestamp-based Split)
import os, io, json, re, asyncio, tempfile, wave, contextlib
from datetime import datetime
from typing import Tuple, List, Optional
import struct

import torch, whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Silero VAD
try:
    from silero_vad import load_silero_vad
    _HAS_VAD = True
except Exception:
    _HAS_VAD = False
    
from deep_translator import GoogleTranslator

# ================= App & CORS =================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ================= Whisper (1회 로드) =================
print("🤖 Loading Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_NAME = "small"
model = whisper.load_model(_MODEL_NAME, device=device)
print(f"✅ Whisper ready on {device} (model={_MODEL_NAME})")

# ================= Silero VAD 로드 =================
if _HAS_VAD:
    print("🎙️ Loading Silero VAD...")
    vad_model = load_silero_vad()
    print("✅ Silero VAD ready")
else:
    vad_model = None
    print("⚠️ Silero VAD not available")

# ================= 설정 =================
PING_INTERVAL = 20.0

# Audio settings
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # s16le

# 버퍼 및 세그먼트 설정
MAX_BUFFER_BYTES = 480 * 1024  # 480KB
SEGMENT_SIZE_BYTES = 32 * 1024  # 32KB

# VAD 설정 (기존과 동일)
SILENCE_LIMIT_MS = 500
VAD_CHUNK_MS = 30  # Silero VAD는 30ms 청크 사용

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

# ====== 간단 후처리 ======
_punct_re = re.compile(r"\s+([,.!?])")
def polish(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    s = _punct_re.sub(r"\1", s)
    if re.match(r"[a-z]", s):
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?" and len(s.split()) > 3:
        s += "."
    return s

# ====== 번역 ======
def translate_to_en(text: str) -> str:
    if not text.strip():
        return ""
    try:
        return GoogleTranslator(source="ko", target="en").translate(text)
    except Exception as e:
        print("⚠️ translate fail:", e)
        return ""

# ====== Whisper 래퍼 (타임스탬프 포함) ======
def transcribe_with_timestamps(path: str) -> Tuple[str, List[dict]]:
    """
    한국어 고정, word_timestamps 활성화
    Returns: (full_text, segments_with_words)
    """
    print(f"🎤 Transcribe: {os.path.basename(path)}")
    result = model.transcribe(
        path,
        language="ko",
        task="transcribe",
        word_timestamps=True,
        fp16=False
    )
    text = (result.get("text") or "").strip()
    segments = result.get("segments", [])
    print(f"📝 (ko) {text[:80]}{'...' if len(text)>80 else ''}")
    return text, segments

# ================= 헬스 =================
@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "model": _MODEL_NAME,
        "device": device,
        "time": now_iso(),
        "vad": "silero" if _HAS_VAD else "none"
    })

# ================= 유틸: WAV 작성 =================
def write_wav(path: str, pcm: bytes, sr: int = SAMPLE_RATE):
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sr)
        wf.writeframes(pcm)

# ================= State 관리 =================
class State:
    def __init__(self):
        # 전체 버퍼 (최대 480KB)
        self.buffer = bytearray()
        
        # 이미 클라이언트에게 전송한 텍스트 길이 (바이트 기준 위치)
        self.sent_audio_bytes = 0
        
        # VAD 상태
        self.last_speech_time_ms: Optional[int] = None
        self.trailing_silence_start_ms: Optional[int] = None
        
        # 옵션
        self.do_polish = True
        self.do_translate = True

def pcm_to_tensor(pcm: bytes) -> torch.Tensor:
    """PCM bytes를 Silero VAD용 tensor로 변환 (16kHz, mono)"""
    samples = torch.frombuffer(pcm, dtype=torch.int16).float() / 32768.0
    return samples

async def check_vad_silence(state: State, pcm_chunk: bytes, current_time_ms: int) -> bool:
    """
    Silero VAD를 사용하여 음성/무음 판정
    Returns: True if speech detected, False if silence
    """
    if not _HAS_VAD or not vad_model:
        return True  # VAD 없으면 항상 음성으로 간주
    
    try:
        audio_tensor = pcm_to_tensor(pcm_chunk)
        if len(audio_tensor) == 0:
            return False
            
        # Silero VAD는 0~1 사이의 확률 반환
        speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()
        is_speech = speech_prob > 0.5
        
        if is_speech:
            state.last_speech_time_ms = current_time_ms
            state.trailing_silence_start_ms = None
            return True
        else:
            if state.last_speech_time_ms is not None:
                if state.trailing_silence_start_ms is None:
                    state.trailing_silence_start_ms = current_time_ms
            return False
    except Exception as e:
        print(f"⚠️ VAD error: {e}")
        return True

def find_sentence_boundaries(segments: List[dict]) -> List[float]:
    """
    Whisper segments에서 마침표(., !, ?) 위치의 타임스탬프 추출
    Returns: List of timestamps in seconds
    """
    boundaries = []
    for seg in segments:
        words = seg.get("words", [])
        for word in words:
            word_text = word.get("word", "").strip()
            if word_text and word_text[-1] in ".!?":
                boundaries.append(word["end"])
    return boundaries

async def transcribe_and_split(state: State, ws: WebSocket):
    """
    버퍼의 내용을 Whisper로 transcribe하고,
    마침표 타임스탬프 기준으로 분할하여 클라이언트에게 전송
    """
    if len(state.buffer) == 0:
        return
    
    # WAV 파일 생성
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(wav_path, bytes(state.buffer), SAMPLE_RATE)
    
    # Whisper 실행
    full_text, segments = await asyncio.to_thread(
        transcribe_with_timestamps, wav_path
    )
    
    try:
        os.remove(wav_path)
    except:
        pass
    
    if not full_text:
        return
    
    # 마침표 경계 찾기
    boundaries = find_sentence_boundaries(segments)
    
    if not boundaries:
        # 마침표가 없으면 전체를 하나의 문장으로 처리
        print("📤 No sentence boundary found, sending full text")
        out = full_text
        if state.do_polish:
            out = polish(out)
        en = translate_to_en(out) if state.do_translate else ""
        
        await ws.send_json({
            "type": "final",
            "time": now_iso(),
            "language": "ko",
            "original": full_text,
            "polished": out,
            "en": en
        })
        
        # 전체 버퍼 flush
        state.sent_audio_bytes = len(state.buffer)
        state.buffer.clear()
        return
    
    # 마침표 경계 기준으로 분할 전송
    last_boundary_sec = 0.0
    for boundary_sec in boundaries:
        # 해당 경계까지의 오디오 바이트 위치 계산
        boundary_bytes = int(boundary_sec * SAMPLE_RATE * SAMPLE_WIDTH)
        
        if boundary_bytes <= state.sent_audio_bytes:
            continue  # 이미 전송한 부분
        
        # 해당 경계까지의 텍스트 추출
        segment_text = ""
        for seg in segments:
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            
            # 이 세그먼트가 현재 경계 이전에 끝나면 포함
            if seg_end <= boundary_sec and seg_start >= last_boundary_sec:
                segment_text += seg.get("text", "")
        
        if segment_text.strip():
            out = segment_text.strip()
            if state.do_polish:
                out = polish(out)
            en = translate_to_en(out) if state.do_translate else ""
            
            print(f"📤 Sending segment: {out[:50]}...")
            await ws.send_json({
                "type": "final",
                "time": now_iso(),
                "language": "ko",
                "original": segment_text.strip(),
                "polished": out,
                "en": en
            })
            
            state.sent_audio_bytes = boundary_bytes
        
        last_boundary_sec = boundary_sec
    
    # 전송한 부분 버퍼에서 제거
    state.buffer = state.buffer[state.sent_audio_bytes:]
    state.sent_audio_bytes = 0

async def process_pcm_stream(state: State, ws: WebSocket, pcm_queue: "asyncio.Queue[bytes]"):
    """
    PCM 스트림 처리:
    - 32KB 세그먼트 단위로 버퍼에 추가
    - VAD 기준 또는 마침표 기준으로 분할
    - 버퍼가 480KB 초과 시 전체 flush
    """
    vad_chunk_bytes = int(SAMPLE_RATE * (VAD_CHUNK_MS / 1000) * SAMPLE_WIDTH)
    
    while True:
        chunk = await pcm_queue.get()
        if chunk is None:
            break
        
        # 32KB 단위로 처리
        current_segment = bytearray()
        buf = memoryview(chunk)
        idx = 0
        
        while idx < len(buf):
            remaining = min(SEGMENT_SIZE_BYTES - len(current_segment), len(buf) - idx)
            current_segment.extend(buf[idx:idx+remaining])
            idx += remaining
            
            # 32KB 세그먼트 완성
            if len(current_segment) >= SEGMENT_SIZE_BYTES:
                state.buffer.extend(current_segment)
                current_time_ms = int(asyncio.get_event_loop().time() * 1000)
                
                # VAD 체크 (30ms 청크 단위)
                for vad_idx in range(0, len(current_segment), vad_chunk_bytes):
                    vad_chunk = bytes(current_segment[vad_idx:vad_idx+vad_chunk_bytes])
                    if len(vad_chunk) < vad_chunk_bytes:
                        break
                    
                    is_speech = await check_vad_silence(state, vad_chunk, current_time_ms)
                    
                    # 500ms 무음 감지 시 분할 처리
                    if not is_speech and state.trailing_silence_start_ms:
                        silence_duration = current_time_ms - state.trailing_silence_start_ms
                        if silence_duration >= SILENCE_LIMIT_MS:
                            print(f"🔇 VAD silence detected ({silence_duration}ms), processing buffer")
                            await transcribe_and_split(state, ws)
                            state.trailing_silence_start_ms = None
                
                # 버퍼 크기 체크 (480KB 초과 시 전체 flush)
                if len(state.buffer) >= MAX_BUFFER_BYTES:
                    print(f"⚠️ Buffer full ({len(state.buffer)} bytes), flushing all")
                    await transcribe_and_split(state, ws)
                
                # Whisper 입력 (32KB마다)
                await transcribe_and_split(state, ws)
                
                current_segment.clear()
        
        # 남은 데이터 처리
        if current_segment:
            state.buffer.extend(current_segment)

# ================= WebSocket =================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 WS connected")
    await ws.send_json({
        "type": "hello",
        "message": "local subtitle stream ready (Silero VAD + Timestamp split)",
        "time": now_iso()
    })
    
    state = State()
    running = True
    
    pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
    
    async def pinger():
        while running:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send_json({"type": "ping", "t": now_iso()})
            except Exception:
                break
    
    ping_task = asyncio.create_task(pinger())
    
    # ffmpeg with highpass filter
    ff = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-af", "highpass=f=150",
        "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE),
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )
    
    async def reader_task():
        try:
            while True:
                data = await ff.stdout.read(4096)
                if not data:
                    await asyncio.sleep(0.005)
                    continue
                try:
                    pcm_queue.put_nowait(data)
                except asyncio.QueueFull:
                    _ = await pcm_queue.get()
                    await pcm_queue.put(data)
        except Exception as e:
            print("⚠️ pcm reader error:", e)
        finally:
            await pcm_queue.put(None)
    
    rtask = asyncio.create_task(reader_task())
    vtask = asyncio.create_task(process_pcm_stream(state, ws, pcm_queue))
    
    try:
        while True:
            frame = await ws.receive()
            
            if "text" in frame and frame["text"] is not None:
                try:
                    msg = json.loads(frame["text"])
                except Exception:
                    await ws.send_json({"type": "error", "message": "invalid json"})
                    continue
                
                t = msg.get("type")
                if t == "start":
                    state.do_polish = bool(msg.get("polish", True))
                    state.do_translate = bool(msg.get("translate", True))
                    await ws.send_json({
                        "type": "ready",
                        "lang": "ko",
                        "polish": state.do_polish,
                        "translate": state.do_translate
                    })
                
                elif t == "stop":
                    print("🛑 Manual 'stop' received, flushing all buffers")
                    await ws.send_json({"type": "status", "message": "finalizing", "time": now_iso()})
                    await transcribe_and_split(state, ws)
                
                else:
                    await ws.send_json({"type": "error", "message": "unknown command"})
            
            elif "bytes" in frame and frame["bytes"] is not None:
                data: bytes = frame["bytes"]
                if data:
                    try:
                        ff.stdin.write(data)
                        await ff.stdin.drain()
                    except Exception as e:
                        print("❌ ffmpeg stdin error:", e)
                        break
            
            elif frame.get("type") == "websocket.disconnect":
                break
            else:
                await ws.send_json({"type": "error", "message": "unsupported frame"})
    
    except WebSocketDisconnect:
        print("🔌 WS disconnected")
    except Exception as e:
        print("❌ WS error:", e)
    finally:
        running = False
        try:
            ping_task.cancel()
        except:
            pass
        try:
            if ff and ff.stdin:
                with contextlib.suppress(Exception):
                    ff.stdin.close()
            if ff:
                with contextlib.suppress(Exception):
                    await ff.wait()
        except:
            pass
        try:
            rtask.cancel()
        except:
            pass
        try:
            await pcm_queue.put(None)
        except:
            pass
        try:
            await vtask
        except:
            pass
        print("🧹 session cleaned")

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "8001"))
    print("=" * 60)
    print(f"🚀 Subtitle WS (Silero VAD + 32KB segments + Timestamp split)")
    print("=" * 60)
    print(f"WS: ws://0.0.0.0:{PORT}/ws")
    print(f"Health: http://0.0.0.0:{PORT}/health")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
