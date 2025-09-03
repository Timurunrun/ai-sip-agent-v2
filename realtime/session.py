import base64
import json
import os
import threading
from typing import Callable, Optional

import websocket
from utils.logging import get_logger, bind


class RealtimeClient:
    """Minimal WebSocket client for gpt-realtime.

    Streams audio deltas (g711 µ-law) with a small jitter buffer so
    playback can start immediately instead of waiting for response end.
    """

    def __init__(
        self,
        on_audio: Callable[[bytes, int], None],
        on_text: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        context: Optional[dict[str, str]] = None,
        on_audio_done: Optional[Callable[[], None]] = None,
    ):
        self.on_audio = on_audio
        self.on_text = on_text or (lambda t: None)
        self.on_error = on_error or (lambda e: None)
        self.on_audio_done = on_audio_done or (lambda: None)

        self._ws: Optional[websocket.WebSocketApp] = None
        self._open_evt = threading.Event()
        self._send_lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None
        # When streaming, emit audio as soon as deltas arrive. We still
        # keep a tiny scratch buffer to hold partial state between messages
        # if needed, but we do not wait for response.done.
        self._buffered_audio: bytearray = bytearray()
        # Default to 8 kHz since output is PCMU (G.711 µ-law)
        self._current_sr = 8000
        # Jitter (ms) before flushing deltas downstream; adjustable via env
        try:
            self._jitter_ms = int(os.getenv("BOT_AUDIO_JITTER_MS", "120"))
        except Exception:
            self._jitter_ms = 120
        self.log = get_logger("realtime")
        if context:
            self.log = bind(self.log, **context)

    def connect(self):
        url = "wss://api.openai.com/v1/realtime?model=gpt-realtime"
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
                # some servers may send binary events separately; ignore
                return
            t = data.get("type")
            # Collect audio delta variants (accept both 'audio' and 'delta' keys)
            if t in ("response.output_audio.delta", "response.audio.delta"):
                b64 = data.get("audio") or data.get("delta") or ""
                if b64:
                    chunk = base64.b64decode(b64)
                    if chunk:
                        # Emit immediately for streaming playback
                        self.on_audio(chunk, self._current_sr)
                # Sample rate may be provided as 'rate' or 'sample_rate'
                sr = data.get("rate") or data.get("sample_rate")
                if sr:
                    try:
                        self._current_sr = int(sr)
                    except Exception:
                        pass
            elif t in ("response.completed", "response.output_audio.done", "response.done"):
                # Signal that current response audio finished
                self.on_audio_done()
            elif t in ("response.output_text.delta", "response.text.delta"):
                self.on_text(data.get("delta", ""))
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
        # Start flusher once connected
        self._start_flush_thread()

    def _start_flush_thread(self):
        if self._flush_thread and self._flush_thread.is_alive():
            return
        self._stop_flush.clear()

        def _flush_loop():
            # Stream out in ~jitter_ms chunks; if data is sparse, flush anyway
            last_flush = 0.0
            while not self._stop_flush.is_set():
                with self._out_cond:
                    # Wait up to jitter_ms for new data
                    self._out_cond.wait(timeout=max(0.001, self._jitter_ms / 1000.0))
                    buf_len = len(self._out_buffer)
                    # Compute target bytes for jitter window
                    bytes_per_ms = max(1, int(self._current_sr / 1000))  # µ-law = 1 byte / sample
                    target = max(80, bytes_per_ms * self._jitter_ms)
                    if buf_len >= target or (buf_len > 0):
                        # Send up to one jitter window worth; if less available, send all
                        send_len = target if buf_len >= target else buf_len
                        to_send = bytes(self._out_buffer[:send_len])
                        # Remove sent portion
                        del self._out_buffer[:send_len]
                    else:
                        to_send = b""
                if to_send:
                    try:
                        self.on_audio(to_send, self._current_sr)
                    except Exception as e:
                        self.on_error(f"on_audio stream error: {e}")
            # Final drain
            with self._out_cond:
                if self._out_buffer:
                    data = bytes(self._out_buffer)
                    self._out_buffer.clear()
                else:
                    data = b""
            if data:
                try:
                    self.on_audio(data, self._current_sr)
                except Exception as e:
                    self.on_error(f"on_audio final drain error: {e}")

        self._flush_thread = threading.Thread(target=_flush_loop, daemon=True)
        self._flush_thread.start()

    def send(self, obj: dict):
        if self._ws:
            with self._send_lock:
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

    def send_audio_chunk(self, pcm16_mono_bytes: bytes):
        evt = {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm16_mono_bytes).decode("ascii")}
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
            # Stop the flusher thread
            try:
                self._stop_flush.set()
                with self._out_cond:
                    self._out_cond.notify_all()
                if self._flush_thread and self._flush_thread.is_alive():
                    self._flush_thread.join(timeout=1.0)
            except Exception:
                pass
            # Best-effort join
            if self._ws_thread and self._ws_thread.is_alive():
                try:
                    self._ws_thread.join(timeout=1.0)
                except Exception:
                    pass
            self._ws = None
            self._ws_thread = None
            self.log.info("Realtime disconnected")
