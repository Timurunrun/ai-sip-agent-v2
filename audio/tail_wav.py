import os
import struct
import time
from typing import Iterator, Optional


class TailWavReader:
    """Tail a growing WAV file and yield raw PCM16 mono chunks.

    Assumes standard 44-byte header, PCM16. Skips header and yields appended bytes
    as they become available.
    """

    def __init__(self, path: str, wait_for_header: bool = True, header_timeout: float = 3.0, poll_interval: float = 0.01):
        self.path = path
        self._f = open(path, "rb", buffering=0)
        # Wait for minimal WAV header (44 bytes)
        start = time.time()
        hdr = b""
        while len(hdr) < 44:
            self._f.seek(0)
            hdr = self._f.read(44)
            if len(hdr) >= 44:
                break
            if not wait_for_header or (time.time() - start) > header_timeout:
                raise RuntimeError("Incomplete WAV header")
            time.sleep(poll_interval)
        # Channels at 22, Sample rate at 24, Bits per sample at 34
        self.channels = struct.unpack_from('<H', hdr, 22)[0]
        self.sample_rate = struct.unpack_from('<I', hdr, 24)[0]
        self.bits_per_sample = struct.unpack_from('<H', hdr, 34)[0]
        self.bytes_per_sample = max(1, self.bits_per_sample // 8)
        # Position after header
        self._offset = 44
        self._f.seek(self._offset)

    def iter_chunks(self, stop_event, frame_bytes: Optional[int] = None, poll_interval: float = 0.01) -> Iterator[bytes]:
        if frame_bytes is None:
            # default to ~20ms frames based on header
            frame_bytes = int(self.sample_rate * self.channels * self.bytes_per_sample * 0.02)
            frame_bytes = max(1, frame_bytes)
        while not stop_event.is_set():
            # Check current file size
            try:
                size = os.path.getsize(self.path)
            except FileNotFoundError:
                break
            # Read while we have at least one frame
            while size - self._offset >= frame_bytes:
                data = self._f.read(frame_bytes)
                if not data:
                    break
                self._offset += len(data)
                yield data      # Quick yield to caller
            time.sleep(poll_interval)

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
