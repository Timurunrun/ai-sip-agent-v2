import os
import time
import threading
from typing import Optional
from collections import deque

import pjsua2 as pj

from audio.tail_wav import TailWavReader
from realtime.session import RealtimeClient
from utils.logging import get_logger, bind, exception


class Call(pj.Call):
    """Per-call handler that bridges audio to gpt-realtime."""

    def __init__(self, acc, call_id=pj.PJSUA_INVALID_ID):
        super().__init__(acc, call_id)
        self.acc = acc
        self._audio_media: Optional[pj.AudioMedia] = None
        self._recorder: Optional[pj.AudioMediaRecorder] = None
        self._player: Optional[pj.AudioMediaPlayer] = None
        self._recording_path: Optional[str] = None
        self._tail: Optional[TailWavReader] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_stream = threading.Event()
        self._rt: Optional[RealtimeClient] = None
        self._playing = False
        self._bot_streamer: Optional["BotAudioStreamer"] = None
        try:
            ci = self.getInfo()
            cid = getattr(ci, "callIdString", None)
        except Exception:
            cid = None
        self._call_id = str(cid or id(self))
        self.log = bind(get_logger("sip.call"), call_id=self._call_id)

    def _has_valid_port(self, media: Optional[pj.AudioMedia]) -> bool:
        try:
            return bool(media) and media.getPortId() >= 0
        except Exception:
            return False

    def _is_call_active(self) -> bool:
        try:
            ci = self.getInfo()
            return ci.state == pj.PJSIP_INV_STATE_CONFIRMED
        except Exception:
            return False

    # Called on SIP state change
    def onCallState(self, prm):
        ci = self.getInfo()
        self.log.info("State change", state=ci.stateText, code=str(ci.lastStatusCode))
        if ci.stateText == "DISCONNECTED":
            self._stop_stream.set()
            try:
                if self._bot_streamer:
                    self._bot_streamer.close()
            except Exception:
                exception(self.log, "Streamer close failed")
            self._cleanup_media()

            # Schedule deletion of this Call on the main thread and remove from account's tracking
            def _finalize():
                try:
                    try:
                        self.delete()
                    except Exception:
                        pass
                    try:
                        if hasattr(self.acc, 'calls') and self in self.acc.calls:
                            self.acc.calls.remove(self)
                    except Exception:
                        pass
                except Exception:
                    pass
            self.acc.cmdq.put(_finalize)

    # Called when media becomes active
    def onCallMediaState(self, prm):
        ci = self.getInfo()
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                self._audio_media = pj.AudioMedia.typecastFromMedia(self.getMedia(mi.index))
                # Start recorder to WAV file
                if self._recording_path:
                    self._recorder = pj.AudioMediaRecorder()
                    self._recorder.createRecorder(self._recording_path)
                    self._audio_media.startTransmit(self._recorder)
                    self.log.info("Recording started", path=str(self._recording_path))
                    # Start tailing thread to stream to realtime
                    self._start_streaming_thread()

    def prepare_recording(self, wav_path: str):
        self._recording_path = wav_path

    def _start_streaming_thread(self):
        if not self._recording_path:
            return
        self._stop_stream.clear()
        self._rt = RealtimeClient(
            on_audio=self._on_bot_audio,
            on_text=self._on_bot_text,
            on_error=lambda e: self.log.error("Realtime error", error=str(e)),
            context={"call_id": self._call_id},
            on_audio_done=self._on_bot_audio_done,
        )
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def _stream_loop(self):
        try:
            self._rt.connect()
            self._rt.update_session()                                               # Configure session with audio in/out
            self._tail = TailWavReader(self._recording_path, wait_for_header=True)  # Create tail reader after header exists
            
            # Send audio frames as they appear (TailWavReader will pick ~20ms frames)
            for chunk in self._tail.iter_chunks(stop_event=self._stop_stream):
                self._rt.send_audio_chunk(chunk)

        except Exception:
            exception(self.log, "Stream loop failure")
        finally:
            try:
                if self._rt:
                    self._rt.close()
            except Exception:
                pass

            # Close tail reader from the streaming thread to avoid race with reads
            try:
                if self._tail:
                    self._tail.close()
            except Exception:
                pass
            finally:
                self._tail = None

    def _on_bot_text(self, text: str):
        self.log.debug("Bot text", text=text)

    def _on_bot_audio(self, audio_bytes: bytes, sample_rate: int):
        # Stream µ-law audio bytes chunk-by-chunk via jittered segment queue
        if not self._bot_streamer:
            self._bot_streamer = BotAudioStreamer(self)
        try:
            self._bot_streamer.feed(audio_bytes, sample_rate or 8000)
        except Exception:
            exception(self.log, "Bot streamer feed failed")

    def _on_bot_audio_done(self):
        if self._bot_streamer:
            try:
                self._bot_streamer.on_done()
            except Exception:
                exception(self.log, "Bot streamer finalize failed")

    def _cleanup_media(self):
        def _do():
            # At call teardown, the conference bridge may already be destroyed.
            # Avoid any stopTransmit/connect calls here to prevent pjmedia_conf assertions.
            self._player = None
            self._recorder = None
            self._audio_media = None
            self.log.debug("Media cleaned up")
        self.acc.cmdq.put(_do)


class BotAudioStreamer:
    """Per-call jittered streamer that queues µ-law WAV segments for seamless playback.

    - Buffers incoming PCMU (G.711 µ-law) bytes and writes short WAV segments.
    - Starts playback only after a small configurable jitter buffer is accumulated.
    - Chains segments using a player with EOF callback to minimize gaps.
    """

    def __init__(self, call: Call):
        self.call = call
        self.cmdq = call.acc.cmdq
        self.log = bind(get_logger("sip.stream"), call_id=call._call_id)
        
        # Settings
        self.sample_rate = 8000
        self.segment_ms = int(os.getenv("BOT_SEGMENT_MS", "300"))               # Size of the server-sent chunk
        self.jitter_ms = int(os.getenv("BOT_JITTER_MS", "100"))                 # Jitter-like waiting
        self.overlap_ms = max(0, int(os.getenv("BOT_OVERLAP_MS", "10")))        # Start next segment slightly before current ends to avoid gaps
        self.segment_bytes = max(1, int(self.sample_rate * self.segment_ms / 1000))
        
        # State
        self._buf = bytearray()
        self._queue: list[tuple[str, int]] = []  # (path, duration_ms)
        self._queued_ms = 0
        self._started = False
        self._end_of_response = False
        
        self._player: Optional[pj.AudioMediaPlayer] = None      # Active player (currently transmitting)
        self._preloaded: Optional[pj.AudioMediaPlayer] = None   # Preloaded player prepared for seamless start
        self._current_end_ts: float = 0.0                       # Internal timing helper for overlap scheduling
        self._lock = threading.Lock()
        self._counter = 0

    def feed(self, ulaw_bytes: bytes, sample_rate: int):
        if not ulaw_bytes:
            return
        with self._lock:
            if sample_rate and sample_rate != self.sample_rate:
                self.sample_rate = sample_rate
                self.segment_bytes = max(1, int(self.sample_rate * self.segment_ms / 1000))
            self._buf.extend(ulaw_bytes)
            self._flush_segments_locked()
            self._maybe_start_locked()

    def on_done(self):
        with self._lock:
            # Flush remaining as a final small segment
            if self._buf:
                self._emit_segment_locked(bytes(self._buf), int(len(self._buf) * 1000 / self.sample_rate))
                self._buf.clear()
            self._end_of_response = True
            self._maybe_start_locked()      # If playback is ongoing and player is idle, try to start next

    def close(self):
        with self._lock:
            self._queue.clear()
            self._queued_ms = 0
            self._buf.clear()
            self._end_of_response = True

            # Stop player on main thread
            if self._player:
                p = self._player
                self._player = None
                def _stop():
                    try:
                        if self.call._has_valid_port(p) and self.call._has_valid_port(self.call._audio_media):
                            try:
                                p.stopTransmit(self.call._audio_media)
                            except Exception:
                                pass
                    finally:
                        try:
                            p.delete()
                        except Exception:
                            pass
                self.cmdq.put(_stop)

    # Internals
    def _flush_segments_locked(self):
        # Emit fixed-size segments for smoother playback
        while len(self._buf) >= self.segment_bytes:
            chunk = self._buf[:self.segment_bytes]
            del self._buf[:self.segment_bytes]
            self._emit_segment_locked(bytes(chunk), self.segment_ms)

    def _emit_segment_locked(self, ulaw_chunk: bytes, duration_ms: int):
        from audio.g711_wav import write_mulaw_wav

        # Use same directory as recording to stay on same filesystem
        base = self.call._recording_path or f"/tmp/pjsua_recordings_v2/call_{int(time.time())}.wav"
        path = base.replace('.wav', f"_stream_{self._counter}.wav")
        self._counter += 1
        try:
            write_mulaw_wav(path, ulaw_chunk, self.sample_rate)
            self._queue.append((path, duration_ms))
            self._queued_ms += duration_ms
        except Exception:
            exception(self.log, "Failed to write segment", file=path)

    def _maybe_start_locked(self):
        # Start playback once jitter buffer is filled, or continue chaining
        if not self._started:
            if self._queued_ms >= self.jitter_ms and self._queue:
                self._started = True
                self._start_next_locked()
        else:
            # If already started but no active player and we have queue, start next
            if not self._player and self._queue:
                self._start_next_locked()

    def _start_next_locked(self):
        if not self._queue:
            if self._end_of_response:
                # Finished current response
                self._started = False
                self._end_of_response = False
            return

        path, dur = self._queue.pop(0)
        self._queued_ms = max(0, self._queued_ms - dur)

        def _play_next():
            # Validate ports
            if not self.call._is_call_active() or not self.call._has_valid_port(self.call._audio_media):
                return

            try:
                p = self._create_player_for(path)
                if self.call._is_call_active() and self.call._has_valid_port(self.call._audio_media):
                    p.startTransmit(self.call._audio_media)
                with self._lock:
                    self._player = p
                    self._current_end_ts = time.monotonic() + max(0.0, float(dur) / 1000.0)     # Compute expected end timestamp for overlap scheduling
                
                # Try to preload the next segment (if any) to remove file open latency
                self.log.info("Segment playback", file=path, ms=str(dur))
                self._try_preload_next()
                self._schedule_overlap_start(dur)
            except Exception:
                exception(self.log, "Segment play failed", file=path)

        self.cmdq.put(_play_next)

    def _try_preload_next(self):
        # Prepare next player in advance without starting it
        with self._lock:
            if self._preloaded or not self._queue:
                return
            next_path, _ = self._queue[0]

        def _prep():
            if not self.call._is_call_active() or not self.call._has_valid_port(self.call._audio_media):
                return
            try:
                np = self._create_player_for(next_path)
                with self._lock:
                    self._preloaded = np
                self.log.debug("Preloaded next segment", file=next_path)
            except Exception:
                exception(self.log, "Preload failed", file=next_path)
        self.cmdq.put(_prep)

    def _create_player_for(self, path: str) -> pj.AudioMediaPlayer:
        # Define a subclass to hook EOF for cleanup and conditional chaining
        streamer = self
        call = self.call

        class _Player(pj.AudioMediaPlayer):
            def __init__(self_inner):
                super().__init__()
                self_inner._seg_path = path
            def onEof2(self_inner):
                def _advance():
                    try:
                        if call._has_valid_port(self_inner) and call._has_valid_port(call._audio_media):
                            try:
                                self_inner.stopTransmit(call._audio_media)
                            except Exception:
                                pass
                    finally:
                        try:
                            self_inner.delete()
                        except Exception:
                            pass
                        try:
                            os.remove(getattr(self_inner, '_seg_path', path))
                        except Exception:
                            pass
                    with streamer._lock:
                        was_active = (streamer._player is self_inner)
                        if was_active:
                            streamer._player = None
                        if was_active and streamer._queue:
                            streamer._start_next_locked()
                streamer.cmdq.put(_advance)

        p = _Player()
        p.createPlayer(path, pj.PJMEDIA_FILE_NO_LOOP)
        return p

    def _schedule_overlap_start(self, dur_ms: int):
        # Start preloaded ~overlap_ms before current ends; fall back to EOF chain otherwise
        if self.overlap_ms <= 0:
            return
        delay = max(0.0, (dur_ms - self.overlap_ms) / 1000.0)

        def _tick():
            # If preloaded ready, start it; else retry shortly if current still active
            with self._lock:
                pre = self._preloaded
                cur = self._player
                still_time = self._current_end_ts - time.monotonic()
            if pre and cur and still_time > -0.25:      # within reasonable window
                def _start_preloaded():
                    if not self.call._is_call_active() or not self.call._has_valid_port(self.call._audio_media):
                        return
                    try:
                        pre.startTransmit(self.call._audio_media)
                        with self._lock:
                            # Transition: new becomes the active player
                            self._player = pre
                            self._preloaded = None
                            # Remove the just-started path from queue since it's now playing
                            if self._queue:
                                # Pop the first queued item as it's now started
                                path_started, next_dur = self._queue.pop(0)
                                self._queued_ms = max(0, self._queued_ms - next_dur)
                                # Update expected end based on the new segment
                                self._current_end_ts = time.monotonic() + max(0.0, float(next_dur) / 1000.0)
                        # After starting, immediately try to preload the subsequent one
                        self._try_preload_next()
                        # And schedule overlap again for the now-active segment
                        # Use the duration of the started segment
                        self._schedule_overlap_start(next_dur)
                        self.log.debug("Overlap start", ms=str(self.overlap_ms))
                    except Exception:
                        exception(self.log, "Overlap start failed")
                self.cmdq.put(_start_preloaded)
            else:
                # If not ready yet and current hasn't finished, retry shortly
                with self._lock:
                    retry = (self._player is not None) and (self._current_end_ts - time.monotonic() > 0.01)
                if retry:
                    t = threading.Timer(0.01, _tick)
                    t.daemon = True
                    t.start()

        t = threading.Timer(delay, _tick)
        t.daemon = True
        t.start()