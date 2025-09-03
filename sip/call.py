import base64
import os
import time
import threading
from typing import Optional

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
        try:
            ci = self.getInfo()
            cid = getattr(ci, "callIdString", None)
        except Exception:
            cid = None
        self._call_id = str(cid or id(self))
        self.log = bind(get_logger("sip.call"), call_id=self._call_id)
        # Cached helper to avoid pjsua_conf_disconnect asserts on invalid ports
        # When media is torn down, conference slot id becomes -1; avoid stop/connect then.

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
        )
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def _stream_loop(self):
        try:
            self._rt.connect()
            # Configure session with audio in/out
            self._rt.update_session()
            # Create tail reader after header exists
            self._tail = TailWavReader(self._recording_path, wait_for_header=True)
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
        # Write µ-law WAV to temp and play it
        from audio.g711_wav import write_mulaw_wav
        tmp = self._recording_path.replace(".wav", f"_out_{int(time.time()*1000)}.wav")
        write_mulaw_wav(tmp, audio_bytes, sample_rate)
        self._play_file(tmp)

    # PJSUA2 calls must run on the main thread; marshal via account's cmdq
    def _play_file(self, path: str):
        def _do():
            try:
                if not self._is_call_active() or not self._has_valid_port(self._audio_media):
                    return
                if self._player:
                    try:
                        if self._is_call_active() and self._has_valid_port(self._player) and self._has_valid_port(self._audio_media):
                            self._player.stopTransmit(self._audio_media)
                    except Exception:
                        pass
                self._player = pj.AudioMediaPlayer()
                self._player.createPlayer(path, pj.PJMEDIA_FILE_NO_LOOP)
                if self._is_call_active() and self._has_valid_port(self._audio_media) and self._has_valid_port(self._player):
                    self._player.startTransmit(self._audio_media)
                self._playing = True
                self.log.info("Playback started", file=path)
            except Exception:
                exception(self.log, "Playback failed", file=path)
        self.acc.cmdq.put(_do)

    def _stop_playback(self):
        def _do():
            if self._is_call_active() and self._player and self._has_valid_port(self._player) and self._has_valid_port(self._audio_media):
                try:
                    self._player.stopTransmit(self._audio_media)
                except Exception:
                    pass
            self._playing = False
            # Release immediately to avoid repeated stop attempts
            self._player = None
            self.log.info("Playback stopped")
        self.acc.cmdq.put(_do)

    def _cleanup_media(self):
        def _do():
            # At call teardown, the conference bridge may already be destroyed.
            # Avoid any stopTransmit/connect calls here to prevent pjmedia_conf assertions.
            self._player = None
            self._recorder = None
            # Mark media as gone to avoid future attempts
            self._audio_media = None
            self.log.debug("Media cleaned up")
        self.acc.cmdq.put(_do)
