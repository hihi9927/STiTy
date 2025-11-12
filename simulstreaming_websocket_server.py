#!/usr/bin/env python3
from simulstreaming_whisper import simulwhisper_args, simul_asr_factory
from whisper_streaming.whisper_websocket_server import main_websocket_server

if __name__ == "__main__":
    main_websocket_server(simul_asr_factory, add_args=simulwhisper_args)
