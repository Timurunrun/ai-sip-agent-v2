import base64
import os
import time
import threading
from typing import Optional

import pjsua2 as pj

from audio.tail_wav import TailWavReader
from realtime.session import RealtimeClient


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

    # Called on SIP state change
    def onCallState(self, prm):
        ci = self.getInfo()
        print(f"[CALL] State: {ci.stateText} code={ci.lastStatusCode}")
        if ci.stateText == "DISCONNECTED":
            self._stop_stream.set()
            try:
                if self._tail:
                    self._tail.close()
            except Exception:
                pass
            self._cleanup_media()

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
                    print(f"[CALL] Recording to {self._recording_path}")
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
            on_error=lambda e: print(f"[RT] error: {e}")
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

        except Exception as e:
            print(f"[CALL] stream loop error: {e}")
        finally:
            try:
                if self._rt:
                    self._rt.close()
            except Exception:
                pass

    def _on_bot_text(self, text: str):
        print(f"[BOT] {text}")

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
                if not self._audio_media:
                    return
                if self._player:
                    try:
                        self._player.stopTransmit(self._audio_media)
                    except Exception:
                        pass
                self._player = pj.AudioMediaPlayer()
                self._player.createPlayer(path, pj.PJMEDIA_FILE_NO_LOOP)
                self._player.startTransmit(self._audio_media)
                self._playing = True
            except Exception as e:
                print(f"[CALL] play error: {e}")
        self.acc.cmdq.put(_do)

    def _stop_playback(self):
        def _do():
            if self._player and self._audio_media:
                try:
                    self._player.stopTransmit(self._audio_media)
                except Exception:
                    pass
            self._playing = False
            # Release immediately to avoid repeated stop attempts
            self._player = None
        self.acc.cmdq.put(_do)

    def _cleanup_media(self):
        def _do():
            try:
                if self._player and self._audio_media:
                    self._player.stopTransmit(self._audio_media)
            except Exception:
                pass
            self._player = None
        self.acc.cmdq.put(_do)
