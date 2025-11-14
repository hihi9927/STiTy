# server_ws_local.py (VAD Buffering + Final-Only + Small Model + HPF)
import os, io, json, re, asyncio, tempfile, wave, contextlib
from datetime import datetime
from typing import Tuple, List
import struct

import torch, whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
try:
    import webrtcvad
    _HAS_VAD = True
except Exception:
    webrtcvad = None
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

# ================= 설정 =================
PING_INTERVAL = 20.0

# VAD
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # s16le
VAD_FRAME_MS = 20
SILENCE_LIMIT_MS = 500 # 900 -> 700 (더 민감하게 문장 종료 감지)
VAD_AGGR = 3

# VAD 최소 발화 시간 1.5s (1.5초 이상은 즉시 처리, 미만은 버퍼링)
MIN_UTTERANCE_SEC = 0.5 

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

# ====== 간단 후처리(문장 다듬기; 로컬 규칙) ======
_punct_re = re.compile(r"\s+([,.!?])")
def polish(text: str) -> str:
    s = text.strip()
    if not s:
        return s
    s = _punct_re.sub(r"\1", s)
    # 첫글자 대문자 (영어일 때만 약하게)
    if re.match(r"[a-z]", s):
        s = s[0].upper() + s[1:]
    # 마침표 보정
    if s and s[-1] not in ".!?" and len(s.split()) > 3:
        s += "."
    return s

# ====== 번역 (deep_translator 사용; 실패시 빈문자열) ======
def translate_auto(text: str, src_lang: str) -> Tuple[str, str]:
    if not text.strip():
        return "", ""
    try:
        if src_lang == "ko":
            en = GoogleTranslator(source="ko", target="en").translate(text)
            return "", en
        elif src_lang == "en":
            ko = GoogleTranslator(source="en", target="ko").translate(text)
            return ko, ""
        else:
            en = GoogleTranslator(source=src_lang, target="en").translate(text)
            ko = GoogleTranslator(source="en", target="ko").translate(en)
            return ko, en
    except Exception as e:
        print("⚠️ translate fail:", e)
        return "", ""

# ====== Whisper 래퍼 ======
def transcribe_path(path: str, lang_hint: str = "auto") -> Tuple[str, str]:
    kwargs = dict(fp16=False)
    if lang_hint and lang_hint != "auto":
        kwargs["language"] = lang_hint
    print(f"🎤 Transcribe: {os.path.basename(path)}")
    result = model.transcribe(path, **kwargs)
    text = (result.get("text") or "").strip()
    lang = result.get("language") or lang_hint or "unknown"
    print(f"📝 ({lang}) {text[:80]}{'...' if len(text)>80 else ''}")
    return text, lang

# ================= 헬스 =================
@app.get("/health")
async def health():
    return JSONResponse({"status":"ok","model":_MODEL_NAME,"device":device,"time":now_iso(),"vad":_HAS_VAD})

# ================= 유틸: WAV 작성 =================
def write_wav(path: str, pcm: bytes, sr: int = SAMPLE_RATE):
    with contextlib.closing(wave.open(path, "wb")) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)  # 2 bytes
        wf.setframerate(sr)
        wf.writeframes(pcm)

# ================= 세그먼트 구조 =================
class Segment:
    __slots__ = ("start_ms","end_ms","pcm_bytes","text_partial")
    def __init__(self, start_ms: int, end_ms: int, pcm_bytes: bytes, text_partial: str = ""):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.pcm_bytes = pcm_bytes
        self.text_partial = text_partial
        
class State:
    def __init__(self):
        self.segments: List[Segment] = []
        self.cur_pcm = bytearray()
        self.utt_start_ms: int | None = None
        self.last_speech_ms: int | None = None

        # 옵션
        self.user_lang_pref = "auto"
        self.do_polish = True
        self.do_translate = True

        # ★ 발화 경계 관리
        self.utt_id: int = 0
        self.in_utt: bool = False
        self.post_final_block_ms: int = 0

def total_pcm_len(state: State) -> int:
    return sum(len(s.pcm_bytes) for s in state.segments) + len(state.cur_pcm)

async def build_wav_from_segments(state: State, include_tail: bool = True, use_tail_only: bool = False) -> str:
    parts = []
    for seg in state.segments:
        if seg.pcm_bytes: parts.append(seg.pcm_bytes)
    if include_tail and state.cur_pcm:
        parts.append(bytes(state.cur_pcm))
    raw = b"".join(parts)

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    write_wav(wav_path, raw, SAMPLE_RATE)
    return wav_path

async def transcribe_async(wav_path: str, lang_hint: str):
    return await asyncio.to_thread(transcribe_path, wav_path, lang_hint)

async def finalize_and_flush(state: State, ws: WebSocket):
    # 진행중 오디오를 세그먼트로 편입
    if state.cur_pcm:
        seg = Segment(state.utt_start_ms or int(asyncio.get_event_loop().time()*1000),
                      state.last_speech_ms or int(asyncio.get_event_loop().time()*1000),
                      bytes(state.cur_pcm))
        state.segments.append(seg)

    if not state.segments:
        state.cur_pcm.clear()
        state.utt_start_ms = None
        state.last_speech_ms = None
        return

    wav_path = await build_wav_from_segments(state, include_tail=False, use_tail_only=False)
    final_text, lang = await transcribe_async(wav_path, state.user_lang_pref)
    try: os.remove(wav_path)
    except: pass

    out = final_text
    if state.do_polish: out = polish(out)
    ko, en = ("","")
    if state.do_translate:
        ko, en = translate_auto(out, lang)

    await ws.send_json({"type":"final","time":now_iso(),"language":lang,
                        "original":final_text, "polished":out, "ko":ko, "en":en})

    # flush
    state.segments.clear()
    state.cur_pcm.clear()
    state.utt_start_ms = None
    state.last_speech_ms = None

# ================= VAD 처리 =================
def vad_is_speech(vad, frame: bytes) -> bool:
    if not _HAS_VAD:
        # VAD가 없으면 항상 speech로 취급 -> 종료는 stop 신호로만
        return True
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except Exception:
        return True

async def process_pcm_stream(state: State, ws: WebSocket, pcm_queue: "asyncio.Queue[bytes]"):
    """PCM 프레임을 받아 VAD 세그먼트와 누적 partial을 관리"""
    vad = webrtcvad.Vad(VAD_AGGR) if _HAS_VAD else None
    frame_bytes = int(SAMPLE_RATE * (VAD_FRAME_MS/1000) * SAMPLE_WIDTH)
    trailing_silence_start: int | None = None

    while True:
        chunk = await pcm_queue.get()
        if chunk is None:
            break
        # chunk를 frame_bytes 단위로 자르기
        buf = memoryview(chunk)
        idx = 0
        while idx + frame_bytes <= len(buf):
            frame = bytes(buf[idx:idx+frame_bytes])
            idx += frame_bytes

            is_speech = vad_is_speech(vad, frame)
            tnow = int(asyncio.get_event_loop().time() * 1000)

            if is_speech:
                if state.utt_start_ms is None:
                    state.utt_start_ms = tnow
                state.cur_pcm.extend(frame)
                state.last_speech_ms = tnow
                trailing_silence_start = None
            else:
                # 무음 프레임
                if state.utt_start_ms is not None and state.last_speech_ms is not None:
                    if trailing_silence_start is None:
                        trailing_silence_start = tnow
                        
                    if (tnow - trailing_silence_start) >= SILENCE_LIMIT_MS:
                        
                        current_audio_sec = len(state.cur_pcm) / (SAMPLE_RATE * SAMPLE_WIDTH)
                        
                        if current_audio_sec < MIN_UTTERANCE_SEC:
                            # [로직 1] 발화가 1.5초 미만 -> 버퍼링
                            print(f"VAD: Short utterance detected ({current_audio_sec:.2f}s), buffering.")
                            if state.cur_pcm:
                                seg = Segment(state.utt_start_ms, state.last_speech_ms, bytes(state.cur_pcm))
                                state.segments.append(seg)
                            # cur_pcm만 비우고 상태 리셋 (segments는 유지)
                            state.cur_pcm.clear()
                            state.utt_start_ms = None
                            state.last_speech_ms = None
                            trailing_silence_start = None
                        
                        else:
                            # [로직 2] 발화가 1.5초 이상 -> 즉시 처리
                            print(f"VAD: Long utterance detected ({current_audio_sec:.2f}s), flushing.")
                            await finalize_and_flush(state, ws)
                            # 상태 초기화는 finalize에서 수행됨
                            trailing_silence_start = None

            # partial(중간) 결과 비활성화 (주석 유지)
            # await maybe_emit_partial_cumulative(state, ws)

# ================= WebSocket =================
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("🔌 WS connected")
    await ws.send_json({"type":"hello","message":"local subtitle stream ready","time":now_iso()})

    state = State()
    running = True

    # PCM 파이프: ffmpeg stdin<-webm bytes, stdout->pcm_queue
    pcm_queue: asyncio.Queue = asyncio.Queue(maxsize=32)

    async def pinger():
        while running:
            await asyncio.sleep(PING_INTERVAL)
            try:
                await ws.send_json({"type":"ping","t":now_iso()})
            except Exception:
                break
    ping_task = asyncio.create_task(pinger())

    # [수정] ffmpeg 명령어에 highpass 필터 추가
    ff = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        
        # [수정] 150Hz 이하의 저주파 노이즈(PC 팬, 에어컨 험) 제거 필터
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
                # push to queue
                try:
                    pcm_queue.put_nowait(data)
                except asyncio.QueueFull:
                    # drop oldest to keep latency low
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
            # ---- 제어 텍스트 ----
            if "text" in frame and frame["text"] is not None:
                try:
                    msg = json.loads(frame["text"])
                except Exception:
                    await ws.send_json({"type":"error","message":"invalid json"})
                    continue

                t = msg.get("type")
                if t == "start":
                    state.user_lang_pref = msg.get("lang", "auto")
                    state.do_polish = bool(msg.get("polish", True))
                    state.do_translate = bool(msg.get("translate", True))
                    await ws.send_json({"type":"ready","lang":state.user_lang_pref,
                                        "polish":state.do_polish, "translate":state.do_translate})

                elif t == "update_config":
                    # 설정 업데이트 처리
                    if "lang" in msg:
                        state.user_lang_pref = msg.get("lang", "auto")
                    if "polish" in msg:
                        state.do_polish = bool(msg.get("polish", True))
                    if "translate" in msg:
                        state.do_translate = bool(msg.get("translate", True))
                    print(f"⚙️ Config updated: lang={state.user_lang_pref}, polish={state.do_polish}, translate={state.do_translate}")
                    await ws.send_json({"type":"config_updated","lang":state.user_lang_pref,
                                        "polish":state.do_polish, "translate":state.do_translate})

                elif t == "stop":
                    print("VAD: Manual 'stop' received, flushing all buffers.")
                    await ws.send_json({"type":"status","message":"finalizing","time":now_iso()})
                    await finalize_and_flush(state, ws)

                else:
                    await ws.send_json({"type":"error","message":"unknown command"})

            # ---- 바이너리(WebM/Opus 바이트) ----
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
                await ws.send_json({"type":"error","message":"unsupported frame"})

    except WebSocketDisconnect:
        print("🔌 WS disconnected")
    except Exception as e:
        print("❌ WS error:", e)
    finally:
        running = False
        try:
            ping_task.cancel()
        except: pass
        try:
            if ff and ff.stdin:
                with contextlib.suppress(Exception):
                    ff.stdin.close()
            if ff:
                with contextlib.suppress(Exception):
                    await ff.wait()
        except: pass
        try:
            rtask.cancel()
        except: pass
        try:
            await pcm_queue.put(None)
        except: pass
        try:
            await vtask
        except: pass
        print("🧹 session cleaned")

if __name__ == "__main__":
    PORT = int(os.getenv("PORT","8001"))
    print("="*60)
    print(f"🚀 Local Subtitle WS server (model={_MODEL_NAME}, VAD, Final-Only, Buffering 1.5s, HPF 150Hz)")
    print("="*60)
    print(f"WS: ws://0.0.0.0:{PORT}/ws")
    print(f"Health: http://0.0.0.0:{PORT}/health")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
