#!/usr/bin/env python3
from whisper_streaming.whisper_online_main import *

import sys
import argparse
import os
import logging
import numpy as np
import asyncio
import json
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16000


class WebSocketHandler:
    """Handles WebSocket connection and ASR processing for one client"""

    def __init__(self, websocket, online_asr_proc, min_chunk):
        self.websocket = websocket
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.is_first = True
        self.audio_buffer = []
        self.running = False

    async def send_message(self, message_dict):
        """Send JSON message to client"""
        try:
            await self.websocket.send(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def detect_and_translate(self, text, detected_lang, lang_probs):
        """Use Whisper's language detection, fallback to probability for other languages"""
        try:
            # Whisper에서 감지한 언어
            lang = detected_lang.lower() if detected_lang else 'en'

            # 한국어 정규화
            if lang in ['korean']:
                lang = 'ko'
            elif lang in ['english']:
                lang = 'en'

            # 영어나 한국어가 아닌 다른 언어로 감지된 경우
            if lang not in ['ko', 'en']:
                logger.info(f"Whisper detected '{lang}' which is neither EN nor KO")

                # 영어와 한국어 확률 비교
                if lang_probs:
                    en_prob = lang_probs.get('en', 0.0)
                    ko_prob = lang_probs.get('ko', 0.0)

                    logger.info(f"Language probabilities - EN: {en_prob:.4f}, KO: {ko_prob:.4f}")

                    # 더 높은 확률로 강제 선택
                    if ko_prob > en_prob:
                        lang = 'ko'
                        logger.info(f"Forced to Korean based on probability ({ko_prob:.4f} > {en_prob:.4f})")
                    else:
                        lang = 'en'
                        logger.info(f"Forced to English based on probability ({en_prob:.4f} >= {ko_prob:.4f})")
                else:
                    # 확률 정보가 없으면 기본값은 영어
                    logger.warning("No language probabilities available, defaulting to English")
                    lang = 'en'
            else:
                logger.info(f"Whisper detected: {lang}")

            ko_text = None
            en_text = None

            # 한국어 → 영어
            if lang == 'ko':
                en_text = GoogleTranslator(source='ko', target='en').translate(text)
                ko_text = None  # 원문이 이미 한국어
                logger.debug(f"Translated KO→EN: {text} → {en_text}")
            # 영어 → 한국어
            else:
                ko_text = GoogleTranslator(source='en', target='ko').translate(text)
                en_text = None  # 원문이 이미 영어
                logger.debug(f"Translated EN→KO: {text} → {ko_text}")

            return lang, ko_text, en_text

        except Exception as e:
            logger.error(f"Translation error: {e}")
            # 에러 발생 시 기본값으로 영어 선택
            return 'en', None, None

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

            # Whisper 언어 감지 사용, 영어/한국어 외에는 확률로 판단
            detected_lang = iteration_output.get('language', 'en')
            lang_probs = iteration_output.get('language_probs', None)

            logger.info(f"Text to translate: {text}")
            logger.info(f"Whisper detected language: {detected_lang}")

            lang, ko_text, en_text = self.detect_and_translate(text, detected_lang, lang_probs)

            logger.info(f"Translation result - lang: {lang}, ko_text: {ko_text}, en_text: {en_text}")

            # polished는 번역 결과 (없으면 원문)
            polished = text
            if lang == 'ko' and en_text:
                # 한국어 → 영어 번역
                polished = en_text
            elif lang == 'en' and ko_text:
                # 영어 → 한국어 번역
                polished = ko_text

            # 번역 결과를 포함한 메시지 생성
            result_msg = {
                'type': 'final',
                'start': start_ms,
                'end': end_ms,
                'original': text,
                'polished': polished,
                'language': lang
            }

            # 번역 결과 추가 (선택적)
            if ko_text:
                result_msg['ko'] = ko_text
            if en_text:
                result_msg['en'] = en_text

            await self.send_message(result_msg)
        else:
            logger.debug("No text in this segment")

    def convert_to_numpy(self, audio_data):
        """Convert Float32 binary data to numpy array"""
        try:
            # Client sends Float32 array directly (no conversion needed)
            audio_float = np.frombuffer(audio_data, dtype=np.float32)

            logger.debug(f"Received Float32 audio: {len(audio_float)} samples")
            return audio_float

        except Exception as e:
            logger.error(f"Error converting audio: {e}")
            return None

    async def process_audio_chunk(self, audio_data):
        """Process incoming audio data"""
        try:
            # Convert binary Float32 to numpy array
            audio = self.convert_to_numpy(audio_data)

            if audio is None or len(audio) == 0:
                logger.warning("Failed to convert audio or empty audio")
                return

            logger.debug(f"Received audio chunk: {len(audio)} samples")

            # Add to buffer
            self.audio_buffer.append(audio)

            # Process when buffer has enough data
            total_samples = sum(len(x) for x in self.audio_buffer)
            min_samples = self.min_chunk * SAMPLING_RATE

            if total_samples >= min_samples or not self.is_first:
                # Concatenate all buffered audio
                if self.audio_buffer:
                    audio_chunk = np.concatenate(self.audio_buffer)
                    self.audio_buffer = []
                    self.is_first = False

                    # Insert audio into ASR
                    self.online_asr_proc.insert_audio_chunk(audio_chunk)

                    # Process and get result
                    result = self.online_asr_proc.process_iter()
                    await self.send_result(result)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            import traceback
            traceback.print_exc()

    async def handle(self):
        """Main handler for WebSocket connection"""
        try:
            logger.info(f"New WebSocket connection from {self.websocket.remote_address}")

            # Send hello message
            await self.send_message({
                'type': 'hello',
                'message': 'Connected to Whisper Streaming Server'
            })

            # Initialize ASR
            self.online_asr_proc.init()
            self.running = True

            # Process incoming messages
            async for message in self.websocket:
                if isinstance(message, bytes):
                    # Binary data = audio
                    if self.running:
                        await self.process_audio_chunk(message)
                else:
                    # Text data = JSON command
                    try:
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

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message}")

        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 즉시 디코딩 중단 및 정리
            logger.info("WebSocket connection closing, flushing ASR processor...")
            self.running = False

            # ASR 프로세서 정리 (디코딩 즉시 중단)
            try:
                # 버퍼 비우기
                self.audio_buffer = []

                # ASR 디코딩 즉시 종료
                if hasattr(self.online_asr_proc, 'finish'):
                    self.online_asr_proc.finish()
                    logger.info("ASR processor finished and flushed")
                else:
                    logger.info("ASR processor flushed (no finish method)")
            except Exception as e:
                logger.error(f"Error flushing ASR processor: {e}")

            logger.info("WebSocket connection closed")


async def websocket_server(websocket, online_asr_proc, min_chunk):
    """WebSocket server handler"""
    handler = WebSocketHandler(websocket, online_asr_proc, min_chunk)
    await handler.handle()


def main_websocket_server(factory, add_args):
    """
    Main WebSocket server entry point

    factory: function that creates the ASR and online processor object from args and logger.
    add_args: add specific args for the backend
    """
    import websockets

    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()

    # server options
    parser.add_argument("--host", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--warmup-file", type=str, dest="warmup_file",
            help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. "
            "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

    # options from whisper_online
    processor_args(parser)

    add_args(parser)

    args = parser.parse_args()

    set_logging(args, logger)

    # setting whisper object by args
    asr, online = asr_factory(args, factory)
    if args.vac:
        min_chunk = args.vac_chunk_size
    else:
        min_chunk = args.min_chunk_size

    # warm up the ASR because the very first transcribe takes more time than the others.
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.warmup(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. "+msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # ASR Pool 생성 (다중 클라이언트 지원)
    # 미리 여러 개의 ASR 인스턴스를 만들어서 풀에 저장
    MAX_CLIENTS = 2  # 동시 접속 가능한 최대 클라이언트 수
    asr_pool = []

    logger.info(f"Creating ASR pool with {MAX_CLIENTS} instances...")
    for i in range(MAX_CLIENTS):
        pool_asr, pool_online = factory(args)

        # Warmup
        if args.warmup_file and os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            pool_asr.warmup(a)

        asr_pool.append((pool_asr, pool_online))
        logger.info(f"Created ASR instance {i+1}/{MAX_CLIENTS}")

    # 사용 가능한 ASR 추적
    available_asr = list(range(MAX_CLIENTS))  # [0, 1]
    asr_lock = asyncio.Lock()

    # Start WebSocket server
    async def server_handler(websocket):
        # 사용 가능한 ASR 가져오기
        async with asr_lock:
            if not available_asr:
                logger.warning(f"No available ASR for client {websocket.remote_address}, rejecting connection")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': f'Server busy. Maximum {MAX_CLIENTS} clients supported.'
                }))
                await websocket.close()
                return

            asr_index = available_asr.pop(0)
            logger.info(f"Assigned ASR #{asr_index} to client {websocket.remote_address} ({len(available_asr)} remaining)")

        try:
            # 풀에서 ASR 가져오기
            pool_asr, pool_online = asr_pool[asr_index]

            # 풀의 ASR을 사용하되, 새로운 OnlineASRProcessor 인스턴스 생성
            # (기존 ASR 모델을 재사용하여 GPU 메모리 절약)
            client_online = type(pool_online)(pool_asr)

            # VAC 설정 적용
            if args.vac:
                from whisper_streaming.vac_online_processor import VACOnlineASRProcessor
                client_online = VACOnlineASRProcessor(args.min_chunk_size, client_online)

            await websocket_server(websocket, client_online, min_chunk)
        finally:
            # 연결 종료 시 ASR 반환
            async with asr_lock:
                available_asr.append(asr_index)
                logger.info(f"Released ASR #{asr_index}, {len(available_asr)} now available")

    async def main():
        logger.info(f'Starting WebSocket server on ws://{args.host}:{args.port}')

        async with websockets.serve(
            server_handler,
            args.host,
            args.port,
            ping_interval=None,  # Disable ping/pong for continuous audio streaming
            ping_timeout=None,
            max_size=10 * 1024 * 1024  # 10MB max message size
        ):
            logger.info(f'WebSocket server listening on ws://{args.host}:{args.port}')
            await asyncio.Future()  # run forever

    asyncio.run(main())