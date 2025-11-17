#!/usr/bin/env python3
"""
LibriSpeech Streaming Client for SimulEval WebSocket Server
Simulates real-time audio streaming by sending FLAC audio chunks at regular intervals
"""

import asyncio
import websockets
import argparse
import os
import glob
from pathlib import Path
import time
import soundfile as sf
import numpy as np
import json


class LibriSpeechStreamingClient:
    def __init__(self, server_url, dataset_path, interval_ms=500, chunk_size=8000):
        """
        Initialize streaming client

        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8765)
            dataset_path: Path to LibriSpeech dataset root
            interval_ms: Interval between chunks in milliseconds
            chunk_size: Number of audio samples per chunk (at 16kHz, 500ms = 8000 samples)
        """
        self.server_url = server_url
        self.dataset_path = Path(dataset_path)
        self.interval_ms = interval_ms
        self.chunk_size = chunk_size
        self.sample_rate = 16000
        self.current_utt_id = None
        self.current_gt = None
        self.chunk_send_time = {}  # Track when each chunk was sent

    def get_chapter_files(self, subset, speaker_id, chapter_id):
        """
        Get all FLAC files for a specific chapter, sorted by filename

        Args:
            subset: Dataset subset (e.g., 'test-clean', 'test-other')
            speaker_id: Speaker ID
            chapter_id: Chapter ID

        Returns:
            List of FLAC file paths sorted in order
        """
        chapter_path = self.dataset_path / subset / str(speaker_id) / str(chapter_id)

        if not chapter_path.exists():
            raise ValueError(f"Chapter path not found: {chapter_path}")

        # Get all FLAC files and sort them
        flac_files = sorted(glob.glob(str(chapter_path / "*.flac")))

        if not flac_files:
            raise ValueError(f"No FLAC files found in {chapter_path}")

        return flac_files

    def get_transcript(self, subset, speaker_id, chapter_id):
        """
        Read the transcript file for the chapter

        Returns:
            Dictionary mapping utterance IDs to transcripts
        """
        chapter_path = self.dataset_path / subset / str(speaker_id) / str(chapter_id)
        trans_file = chapter_path / f"{speaker_id}-{chapter_id}.trans.txt"

        transcripts = {}
        if trans_file.exists():
            with open(trans_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        transcripts[utt_id] = text

        return transcripts

    async def receive_responses(self, websocket):
        """Receive and display server responses"""
        try:
            while True:
                recv_time = time.time()
                response = await websocket.recv()
                try:
                    data = json.loads(response)
                    msg_type = data.get('type', '')

                    if msg_type == 'hello':
                        print(f"✓ Server: {data.get('message', '')}\n")
                    elif msg_type == 'partial_cumulative':
                        # Partial transcription result
                        original = data.get('original', '')
                        ko = data.get('ko', '')
                        en = data.get('en', '')
                        print(f"  [Partial] {original}", end='')
                        if ko:
                            print(f" (KO: {ko})", end='')
                        if en:
                            print(f" (EN: {en})", end='')
                        print('\r', end='')
                    elif msg_type == 'final':
                        # Final transcription result
                        original = data.get('original', '')
                        language = data.get('language', 'unknown')
                        ko = data.get('ko', '')
                        en = data.get('en', '')

                        # Calculate latency if we have send time info
                        latency_str = ""
                        if self.current_utt_id in self.chunk_send_time:
                            send_time = self.chunk_send_time[self.current_utt_id]
                            latency = (recv_time - send_time) * 1000  # ms
                            latency_str = f" [Latency: {latency:.0f}ms]"

                        # Print GT first
                        if self.current_gt:
                            print(f"\n📝 GT: {self.current_gt}")

                        # Print Whisper recognized original
                        print(f"🎙️  Whisper ({language.upper()}): {original}{latency_str}")

                        # Print translation
                        if language == 'ko' and en:
                            print(f"🇺🇸 Translation (EN): {en}")
                        elif language == 'en' and ko:
                            print(f"🇰🇷 Translation (KO): {ko}")

                        print(f"-" * 70)
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"\nError receiving response: {e}")

    async def stream_audio(self, subset, speaker_id, chapter_id, show_transcript=True, show_recognition=True):
        """
        Stream audio from a chapter to the WebSocket server

        Args:
            subset: Dataset subset (e.g., 'test-clean')
            speaker_id: Speaker ID
            chapter_id: Chapter ID
            show_transcript: Whether to display ground truth transcripts
            show_recognition: Whether to display ASR recognition results
        """
        print(f"\n{'='*70}")
        print(f"Starting stream: {subset}/{speaker_id}/{chapter_id}")
        print(f"Interval: {self.interval_ms}ms | Chunk size: {self.chunk_size} samples")
        print(f"{'='*70}\n")

        # Get files and transcripts
        flac_files = self.get_chapter_files(subset, speaker_id, chapter_id)
        transcripts = self.get_transcript(subset, speaker_id, chapter_id)

        print(f"Found {len(flac_files)} audio files\n")

        try:
            async with websockets.connect(self.server_url) as websocket:
                print(f"Connected to {self.server_url}\n")

                # Start receiving responses in background
                receive_task = None
                if show_recognition:
                    receive_task = asyncio.create_task(self.receive_responses(websocket))

                total_chunks_sent = 0

                for file_idx, flac_file in enumerate(flac_files, 1):
                    # Extract utterance ID from filename
                    utt_id = Path(flac_file).stem

                    # Set current utterance info for response handler
                    self.current_utt_id = utt_id
                    self.current_gt = transcripts.get(utt_id, None)

                    # Read audio file
                    audio, sr = sf.read(flac_file)

                    # Ensure audio is mono
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)

                    # Resample if necessary (LibriSpeech is already 16kHz, but just in case)
                    if sr != self.sample_rate:
                        print(f"Warning: Resampling from {sr}Hz to {self.sample_rate}Hz")
                        # Simple resampling (for production, use librosa.resample)
                        audio = np.interp(
                            np.linspace(0, len(audio), int(len(audio) * self.sample_rate / sr)),
                            np.arange(len(audio)),
                            audio
                        )

                    # Show file info (GT will be shown when receiving response)
                    if show_transcript:
                        print(f"\n[File {file_idx}/{len(flac_files)}] {utt_id}")
                        print(f"⏱️  Audio Length: {len(audio)/self.sample_rate:.2f}s")

                    # Stream audio in chunks
                    num_chunks = (len(audio) + self.chunk_size - 1) // self.chunk_size

                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, len(audio))
                        chunk = audio[start_idx:end_idx]

                        # Convert to bytes (Float32 format)
                        chunk_float32 = chunk.astype(np.float32)
                        chunk_bytes = chunk_float32.tobytes()

                        # Record send time for the first chunk
                        if chunk_idx == 0:
                            self.chunk_send_time[utt_id] = time.time()

                        # Send chunk
                        await websocket.send(chunk_bytes)
                        total_chunks_sent += 1

                        # Wait for interval
                        await asyncio.sleep(self.interval_ms / 1000.0)

                        # Print progress (only if not showing recognition)
                        if not show_recognition and chunk_idx % 10 == 0:
                            progress = (chunk_idx + 1) / num_chunks * 100
                            print(f"  Progress: {progress:.1f}% ({chunk_idx+1}/{num_chunks} chunks)", end='\r')

                    if not show_recognition:
                        print(f"  Progress: 100.0% ({num_chunks}/{num_chunks} chunks) - Complete!")

                    # Small gap between files
                    await asyncio.sleep(self.interval_ms / 1000.0)

                # Wait a bit for final results
                await asyncio.sleep(2.0)

                # Cancel receive task
                if receive_task:
                    receive_task.cancel()
                    try:
                        await receive_task
                    except asyncio.CancelledError:
                        pass

                print(f"\n{'='*70}")
                print(f"Stream completed!")
                print(f"Total files: {len(flac_files)}")
                print(f"Total chunks sent: {total_chunks_sent}")
                print(f"Total duration: {total_chunks_sent * self.chunk_size / self.sample_rate:.2f}s")
                print(f"{'='*70}\n")

        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
        except Exception as e:
            print(f"Error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Stream LibriSpeech audio to SimulEval WebSocket server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream a specific chapter with 500ms intervals
  python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686

  # Stream with 250ms intervals (faster)
  python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686 --interval 250

  # Stream with custom chunk size
  python streaming_client.py --subset test-clean --speaker 1089 --chapter 134686 --chunk-size 4000
        """
    )

    parser.add_argument(
        '--server',
        type=str,
        default='wss://edra-raspiest-eagerly.ngrok-free.dev/ws',
        help='WebSocket server URL (default: wss://edra-raspiest-eagerly.ngrok-free.dev/ws)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='.',
        help='Path to LibriSpeech dataset root (default: current directory)'
    )

    parser.add_argument(
        '--subset',
        type=str,
        required=True,
        choices=['test-clean', 'test-other', 'dev-clean', 'dev-other',
                 'train-clean-100', 'train-clean-360', 'train-other-500'],
        help='Dataset subset to use'
    )

    parser.add_argument(
        '--speaker',
        type=int,
        required=True,
        help='Speaker ID (e.g., 1089, 121, 237)'
    )

    parser.add_argument(
        '--chapter',
        type=int,
        required=True,
        help='Chapter ID (e.g., 134686, 134691)'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=500,
        help='Interval between chunks in milliseconds (default: 500ms)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8000,
        help='Number of samples per chunk (default: 8000 = 500ms at 16kHz)'
    )

    parser.add_argument(
        '--no-transcript',
        action='store_true',
        help='Do not display ground truth transcripts'
    )

    parser.add_argument(
        '--no-recognition',
        action='store_true',
        help='Do not display ASR recognition results from server'
    )

    args = parser.parse_args()

    # Create client
    client = LibriSpeechStreamingClient(
        server_url=args.server,
        dataset_path=args.dataset,
        interval_ms=args.interval,
        chunk_size=args.chunk_size
    )

    # Run streaming
    asyncio.run(client.stream_audio(
        subset=args.subset,
        speaker_id=args.speaker,
        chapter_id=args.chapter,
        show_transcript=not args.no_transcript,
        show_recognition=not args.no_recognition
    ))


if __name__ == '__main__':
    main()
