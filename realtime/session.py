import base64
import json
import os
import threading
from typing import Callable, Optional

import websocket
from utils.logging import get_logger, bind


class RealtimeClient:
    """Minimal WebSocket client for gpt-realtime.

    Streams PCM16 audio chunks and collects output audio (g711_ulaw) per response.
    """

    def __init__(
        self,
        on_audio: Callable[[bytes, int], None],
        on_text: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        context: Optional[dict[str, str]] = None,
        on_audio_done: Optional[Callable[[], None]] = None,
        on_speech_started: Optional[Callable[[dict], None]] = None,
        on_assistant_stream_start: Optional[Callable[[str], None]] = None,
    ):
        self.on_audio = on_audio
        self.on_text = on_text or (lambda t: None)
        self.on_error = on_error or (lambda e: None)
        self.on_audio_done = on_audio_done or (lambda: None)
        self.on_speech_started = on_speech_started or (lambda ev: None)
        self.on_assistant_stream_start = on_assistant_stream_start or (lambda item_id: None)

        self._ws: Optional[websocket.WebSocketApp] = None
        self._open_evt = threading.Event()
        self._send_lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None
        self._buffered_audio: bytearray = bytearray()
        self._current_sr = 8000     # PCMU (G.711 µ-law)
        self._current_assistant_item_id: Optional[str] = None
        
        self.log = get_logger("realtime")
        if context:
            self.log = bind(self.log, **context)

    def connect(self):
        url = f"wss://api.openai.com/v1/realtime?model={os.environ.get("OPENAI_MODEL", "gpt-realtime")}"
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        headers = ["Authorization: Bearer " + api_key]

        def on_open(ws):
            self._open_evt.set()
            self.log.info("Realtime connected")

        def on_message(ws, message):
            try:
                data = json.loads(message)
            except Exception:
                return
            t = data.get("type")
            # Collect audio deltas
            if t == "response.output_audio.delta":
                b64 = data.get("delta") or ""
                if b64:
                    chunk = base64.b64decode(b64)
                    if chunk:
                        self.on_audio(chunk, self._current_sr)
                # Track assistant message item id if present and signal start once
                item_id = data.get("item_id")
                if item_id and item_id != self._current_assistant_item_id:
                    self._current_assistant_item_id = item_id
                    try:
                        self.on_assistant_stream_start(item_id)
                    except Exception:
                        pass
            elif t in ("response.output_audio.done", "response.done"):
                # Signal that current response audio finished
                self.on_audio_done()
            elif t == "response.text.delta":
                self.on_text(data.get("delta", ""))
            elif t == "input_audio_buffer.speech_started":
                # Server VAD detected speech — client should interrupt playback and truncate server-side audio
                try:
                    self.on_speech_started(data)
                except Exception:
                    pass
            elif t == "error":
                self.on_error(data.get("error", {}).get("message", str(data)))

        def on_error(ws, err):
            self.on_error(str(err))
            self.log.error("Realtime WS error", error=str(err))

        self._ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
        )

        t = threading.Thread(target=self._ws.run_forever, kwargs=dict(ping_interval=20, ping_timeout=10), daemon=True)
        t.start()
        self._ws_thread = t
        self._open_evt.wait(timeout=10)
        if not self._open_evt.is_set():
            self.log.error("Failed to connect to gpt-realtime")
            raise RuntimeError("Failed to connect to gpt-realtime")
        

    def send(self, obj: dict):
        if self._ws:
            with self._send_lock:
                self._ws.send(json.dumps(obj))

    # Expose current assistant item id for truncation calls
    @property
    def current_assistant_item_id(self) -> Optional[str]:
        return self._current_assistant_item_id

    def update_session(self):
        vad_eagerness = os.getenv("REALTIME_VAD_EAGERNESS", "high")
        voice = os.getenv("REALTIME_VOICE", "cedar")

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
                            "eagerness": vad_eagerness,
                            "create_response": True,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": {"type": "audio/pcmu"},
                        "voice": voice,
                        "speed": 1,
                    },
                },
            },
        }
            
        # Load system prompt from external Markdown file
        try:
            base_dir = os.path.dirname(__file__)
            instructions_path = os.path.join(base_dir, "system_prompt.md")

            with open(instructions_path, "r", encoding="utf-8") as f:
                instructions_text = f.read().strip()

            if instructions_text:
                event["session"]["instructions"] = instructions_text
        except Exception as e:
            self.log.warning("Failed to load instructions file", error=str(e))
        self.send(event)

    def send_audio_chunk(self, pcm16_mono_bytes: bytes):
        evt = {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16_mono_bytes).decode("ascii")}
        self.send(evt)

    def send_truncate(self, item_id: str, audio_end_ms: int):
        try:
            end_ms = int(max(0, audio_end_ms))
        except Exception:
            end_ms = 0
        evt = {
            "type": "conversation.item.truncate",
            "item_id": item_id,
            "content_index": 0,
            "audio_end_ms": end_ms,
        }
        self.log.info("Sending truncate", item_id=item_id, ms=str(end_ms))
        self.send(evt)

    def close(self):
        # Gracefully close the WebSocket to avoid leaking threads across calls
        try:
            if self._ws:
                with self._send_lock:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
        finally:
            
            # Best-effort join
            if self._ws_thread and self._ws_thread.is_alive():
                try:
                    self._ws_thread.join(timeout=1.0)
                except Exception:
                    pass
            self._ws = None
            self._ws_thread = None
            self.log.info("Realtime disconnected")