#!/usr/bin/env python3
import sys
import os
from simulstreaming_whisper import simulwhisper_args, simul_asr_factory
from whisper_streaming.whisper_websocket_server import main_websocket_server

if __name__ == "__main__":
    # Set default warmup file if not specified
    if '--warmup-file' not in sys.argv:
        warmup_path = os.path.join(os.path.dirname(__file__), 'whisper_streaming', 'samples_jfk.wav')
        if os.path.exists(warmup_path):
            sys.argv.extend(['--warmup-file', warmup_path])

    main_websocket_server(simul_asr_factory, add_args=simulwhisper_args)
