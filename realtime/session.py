import base64
import json
import os
import threading
from typing import Callable, Optional

import websocket


class RealtimeClient:
    """Minimal WebSocket client for gpt-realtime.

    Streams PCM16 audio chunks and collects output audio (g711_ulaw) per response.
    """

    def __init__(
        self,
        on_audio: Callable[[bytes, int], None],
        on_text: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.on_audio = on_audio
        self.on_text = on_text or (lambda t: None)
        self.on_error = on_error or (lambda e: None)

        self._ws: Optional[websocket.WebSocketApp] = None
        self._open_evt = threading.Event()
        self._buffered_audio: bytearray = bytearray()
        # Default to 8 kHz since output is PCMU (G.711 µ-law)
        self._current_sr = 8000

    def connect(self):
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        headers = ["Authorization: Bearer " + api_key]

        def on_open(ws):
            self._open_evt.set()

        def on_message(ws, message):
            try:
                data = json.loads(message)
            except Exception:
                # some servers may send binary events separately; ignore
                return
            t = data.get("type")
            # Collect audio delta variants (accept both 'audio' and 'delta' keys)
            if t in ("response.output_audio.delta", "response.audio.delta"):
                b64 = data.get("audio") or data.get("delta") or ""
                if b64:
                    self._buffered_audio.extend(base64.b64decode(b64))
                # Sample rate may be provided as 'rate' or 'sample_rate'
                sr = data.get("rate") or data.get("sample_rate")
                if sr:
                    try:
                        self._current_sr = int(sr)
                    except Exception:
                        pass
            elif t in ("response.completed", "response.output_audio.done", "response.done"):
                if self._buffered_audio:
                    self.on_audio(bytes(self._buffered_audio), self._current_sr)
                    self._buffered_audio.clear()
            elif t in ("response.output_text.delta", "response.text.delta"):
                self.on_text(data.get("delta", ""))
            elif t == "error":
                self.on_error(data.get("error", {}).get("message", str(data)))

        def on_error(ws, err):
            self.on_error(str(err))

        self._ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
        )

        t = threading.Thread(target=self._ws.run_forever, kwargs=dict(ping_interval=20, ping_timeout=10), daemon=True)
        t.start()
        self._open_evt.wait(timeout=10)
        if not self._open_evt.is_set():
            raise RuntimeError("Failed to connect to gpt-realtime")

    def send(self, obj: dict):
        if self._ws:
            self._ws.send(json.dumps(obj))

    def update_session(self):
        prompt_id = os.environ.get("REALTIME_PROMPT_ID")
        event = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": "gpt-realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": "high",
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": "cedar",
                        "speed": 1,
                    },
                },
            },
        }
        if prompt_id:
            event["session"]["prompt"] = {"id": prompt_id}
        event["session"]["instructions"] = (
            "- Respond in the Russian language.\n"
            "- Use short, natural phrases; avoid repetition.\n"
            "- If audio is unintelligible, ask to repeat concisely.\n"
            "- Keep answers under two sentences; speak FAST, human-like, but calm.\n"
        )
        self.send(event)
        # For PCMU output, default rate remains 8000 unless server overrides it

    def send_audio_chunk(self, pcm16_mono_bytes: bytes):
        evt = {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16_mono_bytes).decode("ascii")}
        self.send(evt)

    def flush(self):
        # Signal we’re done with current buffer; server VAD may also auto-create response
        self.send({"type": "input_audio_buffer.commit"})
        # For safety, ask to create a response if not using VAD
        self.send({"type": "response.create"})
