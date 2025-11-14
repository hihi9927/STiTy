#!/usr/bin/env python3
# live_ws_client.py (improved Windows device detection)
import asyncio
import json
import os
import sys
import platform
import argparse
import contextlib
import subprocess
import signal
import re

import websockets

DEFAULT_WS = "wss://edra-raspiest-eagerly.ngrok-free.dev/ws"

# --------------------------
# ffmpeg 커맨드 생성/장치 탐색
# --------------------------
def _has_encoder(name: str) -> bool:
    """ffmpeg에 해당 오디오 인코더가 있는지 간단 체크"""
    try:
        out = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=5
        )
        txt = (out.stdout or "") + (out.stderr or "")
        pattern = rf'^\s*[A-Z\.]+\s+{re.escape(name)}\s+'
        return bool(re.search(pattern, txt, re.MULTILINE))
    except Exception as e:
        print(f"⚠️ 인코더 체크 실패 ({name}): {e}")
        return False

def auto_detect_device() -> tuple[str, str]:
    """
    OS별로 자동으로 기본 오디오 입력 장치를 찾습니다.
    Returns: (device_name, backend) - Windows/macOS는 backend=None
    """
    system = platform.system().lower()
    
    if system.startswith("win"):
        # Windows: dshow 장치 목록에서 (audio) 태그로 오디오 장치 찾기
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout + result.stderr
            
            # (audio) 태그가 있는 줄에서 장치명 추출
            lines = output.split('\n')
            audio_devices = []
            
            for i, line in enumerate(lines):
                # (audio) 또는 (Audio)가 포함된 줄 찾기
                if '(audio)' in line.lower() and '"' in line:
                    # "장치명" 추출
                    match = re.search(r'"([^"]+)"', line)
                    if match:
                        device_name = match.group(1)
                        # Alternative name 줄은 제외 (@ 문자로 시작하는 장치 ID)
                        if not device_name.startswith('@'):
                            audio_devices.append(device_name)
            
            if audio_devices:
                # 마이크 관련 키워드가 있는 장치 우선 선택
                mic_keywords = ['마이크', 'mic', 'microphone', 'headset', 'input', '입력']
                
                for device in audio_devices:
                    for kw in mic_keywords:
                        if kw in device.lower():
                            print(f"🎤 자동 감지된 장치: {device}")
                            return (device, None)
                
                # 키워드가 없으면 첫 번째 오디오 장치 사용
                device = audio_devices[0]
                print(f"🎤 자동 감지된 장치: {device}")
                return (device, None)
            
            # 장치를 찾지 못한 경우
            print("⚠️ 오디오 장치를 찾을 수 없습니다.")
            print("\n📋 ffmpeg 전체 출력:")
            print(output)
            print("\n💡 해결 방법:")
            print("1. 'python client.py --list-devices' 로 장치 확인")
            print("2. 'python client.py --device \"장치명\"' 으로 수동 지정")
            return (None, None)
            
        except FileNotFoundError:
            print("❌ ffmpeg를 찾을 수 없습니다. ffmpeg를 설치하고 PATH에 추가하세요.")
            return (None, None)
        except Exception as e:
            print(f"⚠️ Windows 장치 감지 실패: {e}")
            import traceback
            traceback.print_exc()
            return (None, None)
    
    elif system == "darwin":
        # macOS: avfoundation 장치 목록에서 첫 번째 오디오 장치 찾기
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                capture_output=True, text=True, timeout=5
            )
            output = result.stdout + result.stderr
            
            # AVFoundation audio devices 섹션에서 첫 번째 장치의 인덱스 추출
            for line in output.split('\n'):
                # [AVFoundation indev @ ...] [0] Built-in Microphone
                match = re.search(r'\[(\d+)\].*?(?:Microphone|Audio|Input)', line, re.IGNORECASE)
                if match:
                    device_idx = match.group(1)
                    device = f":{device_idx}"
                    print(f"🎤 자동 감지된 장치: 인덱스 {device_idx}")
                    return (device, None)
            
            # 기본값으로 :0 사용
            print("🎤 기본 장치 사용: :0")
            return (":0", None)
            
        except Exception as e:
            print(f"⚠️ macOS 장치 감지 실패, 기본값(:0) 사용: {e}")
            return (":0", None)
    
    else:
        # Linux: PulseAudio 우선, 없으면 ALSA
        # 1. PulseAudio 시도
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    # 모니터가 아닌 실제 입력 소스 찾기
                    if '.monitor' not in line and line.strip():
                        device = line.split()[1]  # 두 번째 컬럼이 장치명
                        print(f"🎤 자동 감지된 장치 (PulseAudio): {device}")
                        return (device, "pulse")
                
                # 모니터만 있는 경우 default 사용
                print("🎤 PulseAudio 기본 장치 사용: default")
                return ("default", "pulse")
        except FileNotFoundError:
            pass  # pactl 없음, ALSA 시도
        except Exception as e:
            print(f"⚠️ PulseAudio 감지 실패: {e}")
        
        # 2. ALSA 시도
        try:
            result = subprocess.run(
                ["arecord", "-l"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                # ALSA 장치가 있으면 default 사용
                print("🎤 ALSA 기본 장치 사용: default")
                return ("default", "alsa")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"⚠️ ALSA 감지 실패: {e}")
        
        # 최종 fallback
        print("🎤 기본 장치 사용: default (pulse)")
        return ("default", "pulse")

def ffmpeg_cmd_for_os(args) -> list[str]:
    """
    OS별로 ffmpeg가 마이크를 인코딩하여 stdout(pipe:1)로 내보내도록 명령 생성
    - opus/libopus 우선, 없으면 PCM으로 폴백
    """
    # 장치 자동 감지 (args.device가 없을 경우)
    device = args.device
    backend = args.backend
    
    if device is None:
        print("🔍 오디오 장치 자동 감지 중...")
        device, detected_backend = auto_detect_device()
        if device is None:
            print("\n❌ 자동 감지 실패. 장치를 수동으로 지정해주세요.")
            print("   예: python client.py --device \"마이크 (Realtek High Definition Audio)\"")
            raise ValueError("오디오 장치를 찾을 수 없음")
        if detected_backend:
            backend = detected_backend
    
    # 인코더 선택 및 확인
    has_libopus = _has_encoder("libopus")
    has_opus = _has_encoder("opus")
    
    if has_libopus:
        encoder = "libopus"
        format_opts = [
            "-ac", "1",
            "-ar", "48000",
            "-c:a", encoder,
            "-b:a", args.bitrate,
            "-f", "webm",
            "pipe:1"
        ]
    elif has_opus:
        encoder = "opus"
        format_opts = [
            "-ac", "1",
            "-ar", "48000",
            "-c:a", encoder,
            "-b:a", args.bitrate,
            "-strict", "-2",
            "-f", "webm",
            "pipe:1"
        ]
    else:
        format_opts = [
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "pipe:1"
        ]
    
    common_opts = format_opts

    system = platform.system().lower()
    if system.startswith("win"):
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "dshow",
            "-i", f"audio={device}",
        ] + common_opts

    elif system == "darwin":
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-i", device,
        ] + common_opts

    else:
        if backend == "pulse":
            return [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "pulse", "-i", device,
            ] + common_opts
        elif backend == "alsa":
            return [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "alsa", "-i", device,
            ] + common_opts
        else:
            raise SystemExit("Linux에서는 --backend pulse|alsa 중 하나를 선택하세요.")

def list_devices(args):
    """ffmpeg로 OS별 입력 장치 나열"""
    system = platform.system().lower()
    print("🔎 사용 가능한 오디오 장치 목록:\n")
    print("=" * 60)
    
    if system.startswith("win"):
        cmd = ["ffmpeg", "-hide_banner", "-f", "dshow", "-list_devices", "true", "-i", "dummy"]
        print("Windows DirectShow 장치:")
        print("-" * 60)
    elif system == "darwin":
        cmd = ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""]
        print("macOS AVFoundation 장치:")
        print("-" * 60)
    else:
        print("Linux 오디오 장치:\n")
        print("📍 PulseAudio 소스:")
        print("-" * 60)
        with contextlib.suppress(Exception):
            subprocess.run(["pactl", "list", "short", "sources"], check=False)
        print("\n📍 ALSA 장치:")
        print("-" * 60)
        with contextlib.suppress(Exception):
            subprocess.run(["arecord", "-l"], check=False)
        print("\n" + "=" * 60)
        print("\n✅ 사용 예:")
        print("   python client.py --device \"default\" --backend pulse")
        print("   python client.py --device \"hw:0,0\" --backend alsa")
        return
    
    with contextlib.suppress(Exception):
        subprocess.run(cmd, check=False)
    
    print("\n" + "=" * 60)
    print("\n✅ 사용 예:")
    if system.startswith("win"):
        print("   python client.py --device \"마이크 (Realtek High Definition Audio)\"")
    elif system == "darwin":
        print("   python client.py --device \":0\"")

# --------------------------
# WebSocket 수신 핸들러
# --------------------------
async def receiver(ws):
    try:
        async for msg in ws:
            with contextlib.suppress(Exception):
                data = json.loads(msg)
                t = data.get("type")
                if t == "hello":
                    print(f"👋 {data.get('message','')} {data.get('time','')}")
                elif t == "ready":
                    print(f"✅ 서버 준비(lang={data.get('lang')}, polish={data.get('polish')}, translate={data.get('translate')})")
                elif t == "partial_cumulative":
                    txt = data.get("polished") or data.get("original","")
                    print(f"🟡 [누적{data.get('seq')}] {txt}")
                elif t == "final":
                    txt = data.get("polished") or data.get("original","")
                    print(f"🟢 [최종] {txt}")
                    if data.get("ko"):
                        print(f"   (ko) {data['ko']}")
                    if data.get("en"):
                        print(f"   (en) {data['en']}")
                elif t == "error":
                    print(f"❗ 서버 오류: {data.get('message')}")
                elif t == "status":
                    print(f"📊 상태: {data.get('message')}")
    except websockets.ConnectionClosed:
        print("🔌 서버와의 WS 연결이 종료되었습니다.")

# --------------------------
# WebSocket 송신(마이크 → ffmpeg → WS)
# --------------------------
async def sender(ws, args):
    try:
        cmd = ffmpeg_cmd_for_os(args)
    except ValueError as e:
        print(f"\n❌ {e}")
        return
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.DEVNULL
        )
    except Exception as e:
        print(f"❌ ffmpeg 실행 실패: {e}")
        return

    # stderr 모니터링
    async def monitor_stderr():
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                err_msg = line.decode('utf-8', errors='ignore').strip()
                if err_msg and 'error' in err_msg.lower():
                    print(f"⚠️ ffmpeg: {err_msg}")
        except Exception:
            pass

    stderr_task = asyncio.create_task(monitor_stderr())

    # Ctrl+C 처리
    stopping = False
    def _sigint(*_):
        nonlocal stopping
        stopping = True
        print("\n⏹️  중지 요청... 잠시만요(최종 flush)")
    
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(s, _sigint)

    # 시작 설정 전송
    try:
        await ws.send(json.dumps({
            "type": "start",
            "lang": args.lang,
            "polish": not args.no_polish,
            "translate": not args.no_translate,
        }))
    except Exception as e:
        print(f"❌ start 메시지 전송 실패: {e}")
        return

    sent = 0
    chunk_count = 0
    
    try:
        first_chunk = True
        
        while True:
            chunk = await proc.stdout.read(args.chunk)
            if not chunk:
                if proc.returncode is not None:
                    print(f"⚠️ ffmpeg 종료됨 (코드: {proc.returncode})")
                break
            
            if first_chunk:
                print(f"✅ 오디오 스트림 시작! (첫 청크: {len(chunk)} bytes)")
                first_chunk = False
            
            try:
                await ws.send(chunk)
                sent += len(chunk)
                chunk_count += 1
                
                if chunk_count % 50 == 0:
                    print(f"📤 전송 중... {sent/1024:.1f} KB ({chunk_count} chunks)")
                
            except Exception as e:
                print(f"❌ WS 전송 실패: {e}")
                break
            
            if chunk_count % 10 == 0:
                await asyncio.sleep(0.001)
            
            if stopping:
                break
                
    except Exception as e:
        print(f"❌ 스트리밍 오류: {e}")
    finally:
        try:
            await ws.send(json.dumps({"type": "stop"}))
        except Exception as e:
            print(f"⚠️ stop 신호 전송 실패: {e}")
        
        with contextlib.suppress(Exception):
            proc.kill()
            await proc.wait()
        
        stderr_task.cancel()
        print(f"📦 전송 종료 (총 {sent/1024:.1f} KB, {chunk_count} chunks)")

# --------------------------
# 메인 실행
# --------------------------
async def run(args):
    if args.list_devices:
        list_devices(args)
        return

    print(f"🔌 WS 연결 시도: {args.server}")
    try:
        async with websockets.connect(
            args.server, 
            max_size=None, 
            ping_interval=20,
            ping_timeout=30
        ) as ws:
            recv_task = asyncio.create_task(receiver(ws))
            send_task = asyncio.create_task(sender(ws, args))
            
            done, pending = await asyncio.wait(
                {recv_task, send_task},
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            
            await asyncio.sleep(0.5)
            
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket 오류: {e}")
    except Exception as e:
        print(f"❌ 연결 오류: {e}")

def parse_args():
    p = argparse.ArgumentParser(
        description="리얼 라이브 STT WebSocket 클라이언트 (자동 장치 감지)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 자동 장치 감지 (권장)
  python client.py
  
  # 장치 목록 확인
  python client.py --list-devices
  
  # 특정 장치 지정
  python client.py --device "마이크 이름"
  
  # 서버 주소 변경
  python client.py --server ws://localhost:8001/ws
        """
    )
    p.add_argument("--server", default=DEFAULT_WS, help="WS 서버 주소")
    p.add_argument("--lang", default="auto", help="언어 힌트 (auto|ko|en 등)")
    p.add_argument("--no-polish", action="store_true", help="문장 다듬기 끄기")
    p.add_argument("--no-translate", action="store_true", help="번역 끄기")
    p.add_argument("--bitrate", default="32k", help="Opus 비트레이트")
    p.add_argument("--frame", type=int, default=20, help="Opus frame_duration(ms)")
    p.add_argument("--chunk", type=int, default=8192, help="WS 청크 크기(bytes)")
    p.add_argument("--device", default=None, help="입력 장치 (자동 감지 가능)")
    p.add_argument("--backend", default="pulse", help="Linux 백엔드(pulse|alsa)")
    p.add_argument("--list-devices", action="store_true", help="장치 목록 확인")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n👋 종료합니다.")
