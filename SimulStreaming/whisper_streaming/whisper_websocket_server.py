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

    def detect_and_translate(self, text, detected_lang, lang_probs=None):
        """Detect language and translate English ↔ Korean"""
        try:
            # 언어 감지 (Whisper에서 제공하는 언어 또는 자동 감지)
            lang = detected_lang.lower() if detected_lang else 'en'

            # 한국어 또는 영어가 아닌 경우, 확률 기반 또는 텍스트 분석으로 판단
            if lang not in ['ko', 'korean', 'en', 'english']:
                # lang_probs가 제공된 경우, EN과 KO 확률 비교
                if lang_probs:
                    en_prob = lang_probs.get('en', 0.0)
                    ko_prob = lang_probs.get('ko', 0.0)

                    logger.info(f"Language probabilities - EN: {en_prob:.4f}, KO: {ko_prob:.4f}")

                    # EN과 KO 중 더 높은 확률의 언어 선택
                    if ko_prob > en_prob:
                        lang = 'ko'
                        logger.info(f"Selected Korean based on probability ({ko_prob:.4f} > {en_prob:.4f})")
                    else:
                        lang = 'en'
                        logger.info(f"Selected English based on probability ({en_prob:.4f} >= {ko_prob:.4f})")
                else:
                    # 확률 정보가 없으면 텍스트 분석
                    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3')
                    if korean_chars > len(text) * 0.3:
                        lang = 'ko'
                    else:
                        lang = 'en'
                    logger.info(f"Unknown language '{detected_lang}', auto-detected as: {lang} (text-based)")

            # 한국어 정규화
            if lang in ['korean']:
                lang = 'ko'
            elif lang in ['english']:
                lang = 'en'

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
            return lang, None, None

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

            # 언어 감지 및 번역 (항상 수행)
            detected_lang = iteration_output.get('language', 'en')
            lang_probs = iteration_output.get('language_probs', None)

            logger.info(f"Language detection - detected_lang: {detected_lang}, lang_probs: {lang_probs}")
            logger.info(f"Text to translate: {text}")

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

    def convert_pcm_to_float(self, pcm_data):
        """Convert RAW PCM Int16 data to float32 numpy array"""
        try:
            # PCM data is already in 16kHz Int16 format from client
            # Convert bytes to Int16 array
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)

            # Convert Int16 to Float32 (-1.0 to 1.0)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            logger.debug(f"Converted PCM: {len(audio_float)} samples")
            return audio_float

        except Exception as e:
            logger.error(f"Error converting PCM: {e}")
            return None

    async def process_audio_chunk(self, audio_data):
        """Process incoming audio data"""
        try:
            # Convert RAW PCM to float32
            audio = self.convert_pcm_to_float(audio_data)

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

    # Start WebSocket server
    async def server_handler(websocket):
        await websocket_server(websocket, online, min_chunk)

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