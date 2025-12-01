#!/usr/bin/env python3
import sys
import os

# Add parent directory to path to import whisper_streaming
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'SimulStreaming'))

# Import the factory and args functions
from librispeech_whisper import librispeech_whisper_args, librispeech_asr_factory
from server_test import main_websocket_server

if __name__ == "__main__":
    # Set default warmup file if not specified
    if '--warmup-file' not in sys.argv:
        warmup_path = os.path.join(parent_dir, 'SimulStreaming', 'whisper_streaming', 'samples_jfk.wav')
        if os.path.exists(warmup_path):
            sys.argv.extend(['--warmup-file', warmup_path])

    main_websocket_server(librispeech_asr_factory, add_args=librispeech_whisper_args)
