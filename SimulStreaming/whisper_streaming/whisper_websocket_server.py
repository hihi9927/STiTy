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

        # Translation buffer: accumulate text segments until sentence break
        self.translation_buffer = []  # List of text segments
        self.detected_language = None  # Store detected language
        self.last_vad_status = None  # Track VAD status for silence detection
        self.nonvoice_count = 0  # Count consecutive nonvoice detections

    async def send_message(self, message_dict):
        """Send JSON message to client"""
        try:
            await self.websocket.send(json.dumps(message_dict))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def detect_and_translate(self, text, detected_lang, lang_probs):
        """Translate based on language already detected and mapped to EN/KO by Whisper"""
        try:
            # Language is already set to 'en' or 'ko' by simul_whisper.py
            lang = detected_lang.lower() if detected_lang else 'en'

            logger.info(f"[detect_and_translate] Language from Whisper: {lang}")

            ko_text = None
            en_text = None

            # 한국어 → 영어 번역
            if lang == 'ko':
                en_text = GoogleTranslator(source='ko', target='en').translate(text)
                logger.debug(f"Translated KO→EN: {text} → {en_text}")
            # 영어 → 한국어 번역
            else:
                ko_text = GoogleTranslator(source='en', target='ko').translate(text)
                logger.debug(f"Translated EN→KO: {text} → {ko_text}")

            return lang, ko_text, en_text

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return 'en', None, None

    async def send_result(self, iteration_output):
        """Send transcription result to client"""
        # Track VAD status
        vad_status = iteration_output.get('vad_status') if iteration_output else None
        if vad_status:
            if vad_status == 'nonvoice':
                self.nonvoice_count += 1
            else:
                self.nonvoice_count = 0
            self.last_vad_status = vad_status
            logger.debug(f"[VAD] Status: {vad_status}, nonvoice_count: {self.nonvoice_count}")

        if iteration_output and iteration_output.get('text'):
            start_ms = int(iteration_output['start'] * 1000)
            end_ms = int(iteration_output['end'] * 1000)
            text = iteration_output['text'].strip()

            if not text:
                logger.debug("Empty text in segment")
                return

            message = f"{start_ms} {end_ms} {text}"
            print(message, flush=True, file=sys.stderr)

            # Get detected language
            detected_lang = iteration_output.get('language', 'en')
            lang_probs = iteration_output.get('language_probs', None)

            # Store language if not set
            if self.detected_language is None:
                self.detected_language = detected_lang

            logger.info(f"[send_result] Text segment: {text}")
            logger.info(f"[send_result] Detected language: {detected_lang}")

            # Add to translation buffer
            self.translation_buffer.append({
                'text': text,
                'start': start_ms,
                'end': end_ms
            })

            # Check if this segment ends with sentence-ending punctuation
            # This indicates the sentence is complete and ready for translation
            sentence_complete = any(text.endswith(p) for p in ['.', '!', '?', '。', '!', '?'])

            # Also check if VAD detected silence (nonvoice for 2+ consecutive detections)
            # This indicates the speaker has paused, so we should translate
            vad_silence_detected = self.nonvoice_count >= 2

            if sentence_complete or vad_silence_detected:
                # Sentence is complete or silence detected - translate the entire buffer
                reason = "punctuation" if sentence_complete else "VAD silence"
                logger.info(f"[send_result] Translation triggered by {reason}. Translating {len(self.translation_buffer)} segments")

                # Combine all buffered text
                full_text = ' '.join(seg['text'] for seg in self.translation_buffer)
                first_start = self.translation_buffer[0]['start']
                last_end = self.translation_buffer[-1]['end']

                # Translate the complete sentence
                lang, ko_text, en_text = self.detect_and_translate(full_text, self.detected_language, lang_probs)

                logger.info(f"[send_result] Translation result - lang: {lang}, ko: {ko_text}, en: {en_text}")

                # polished는 번역 결과 (없으면 원문)
                polished = full_text
                if lang == 'ko' and en_text:
                    polished = en_text
                elif lang == 'en' and ko_text:
                    polished = ko_text

                # Send the complete sentence with translation
                result_msg = {
                    'type': 'final',
                    'start': first_start,
                    'end': last_end,
                    'original': full_text,
                    'polished': polished,
                    'language': lang
                }

                if ko_text:
                    result_msg['ko'] = ko_text
                if en_text:
                    result_msg['en'] = en_text

                await self.send_message(result_msg)

                # Clear the buffer and reset VAD counter
                self.translation_buffer = []
                self.nonvoice_count = 0
                logger.info("[send_result] Translation buffer cleared, VAD counter reset")
            else:
                # Sentence not complete yet - just send partial result without translation
                logger.info(f"[send_result] Partial segment (buffer: {len(self.translation_buffer)} segments)")
                result_msg = {
                    'type': 'partial',
                    'start': start_ms,
                    'end': end_ms,
                    'original': text,
                    'language': detected_lang
                }
                await self.send_message(result_msg)
        else:
            logger.debug("No text in this segment")

    def convert_to_numpy(self, audio_data):
        """Convert Int16 binary data to numpy array"""
        try:
            # Client sends Int16 PCM data
            audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

            # Convert to float32 for Whisper (range: -32768 ~ 32767 -> -1.0 ~ 1.0)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            logger.debug(f"Received Int16 audio: {len(audio_int16)} samples -> {len(audio_float)} float32 samples")
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

                        if msg_type == 'finish':
                            logger.info("Received finish command - flushing buffer")
                            # Flush remaining audio buffer
                            result = self.online_asr_proc.finish()
                            if result:
                                await self.send_result(result)

                            # Flush remaining translation buffer
                            if self.translation_buffer:
                                logger.info(f"Flushing remaining translation buffer ({len(self.translation_buffer)} segments)")
                                full_text = ' '.join(seg['text'] for seg in self.translation_buffer)
                                first_start = self.translation_buffer[0]['start']
                                last_end = self.translation_buffer[-1]['end']

                                # Translate the remaining text
                                lang, ko_text, en_text = self.detect_and_translate(full_text, self.detected_language, None)

                                polished = full_text
                                if lang == 'ko' and en_text:
                                    polished = en_text
                                elif lang == 'en' and ko_text:
                                    polished = ko_text

                                result_msg = {
                                    'type': 'final',
                                    'start': first_start,
                                    'end': last_end,
                                    'original': full_text,
                                    'polished': polished,
                                    'language': lang
                                }

                                if ko_text:
                                    result_msg['ko'] = ko_text
                                if en_text:
                                    result_msg['en'] = en_text

                                await self.send_message(result_msg)
                                self.translation_buffer = []

                            logger.info("Buffer flushed")
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

    # Start WebSocket server
    async def server_handler(websocket):
        # warmup된 online 객체를 직접 사용 (단일 클라이언트만 지원)
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