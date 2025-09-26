import threading
import wave
from typing import Optional


class ConversationRecorder:
    """Write synchronized stereo WAV with caller and assistant channels."""

    def __init__(self, path: str, sample_rate: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self._wave = wave.open(path, "wb")
        self._wave.setnchannels(2)
        self._wave.setsampwidth(2)  # PCM16
        self._wave.setframerate(sample_rate)
        self._lock = threading.Lock()
        self._buffers = [bytearray(), bytearray()]
        self._closed = False
        self._flush_block = 4096  # bytes per channel

    def write_caller(self, pcm16_bytes: bytes) -> None:
        self._write_channel(0, pcm16_bytes)

    def write_assistant(self, pcm16_bytes: bytes) -> None:
        self._write_channel(1, pcm16_bytes)

    def _write_channel(self, index: int, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            if self._closed:
                return
            buf = self._buffers[index]
            buf.extend(data)
            other = self._buffers[1 - index]
            diff = len(buf) - len(other)
            if diff > 0:
                other.extend(b"\x00" * diff)
            self._flush_locked()

    def _flush_locked(self) -> None:
        ready = min(len(self._buffers[0]), len(self._buffers[1]))
        if ready < 2:
            return
        ready -= ready % 2  # ensure whole samples
        while ready >= 2:
            take = min(ready, self._flush_block)
            take -= take % 2
            if take <= 0:
                break
            left = bytes(self._buffers[0][:take])
            right = bytes(self._buffers[1][:take])
            interleaved = self._interleave(left, right)
            self._wave.writeframes(interleaved)
            del self._buffers[0][:take]
            del self._buffers[1][:take]
            ready -= take

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            # Pad any remaining shorter channel with silence
            diff = len(self._buffers[0]) - len(self._buffers[1])
            if diff > 0:
                self._buffers[1].extend(b"\x00" * diff)
            elif diff < 0:
                self._buffers[0].extend(b"\x00" * (-diff))
            # Flush remaining data
            self._flush_locked()
            self._wave.close()
            self._closed = True

    @staticmethod
    def _interleave(left: bytes, right: bytes) -> bytes:
        # Both inputs must have equal length and be multiples of 2 bytes.
        frames = bytearray(len(left) * 2)
        frames[0::4] = left[0::2]
        frames[1::4] = left[1::2]
        frames[2::4] = right[0::2]
        frames[3::4] = right[1::2]
        return bytes(frames)

    def __enter__(self):  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - convenience
        self.close()
        return False
