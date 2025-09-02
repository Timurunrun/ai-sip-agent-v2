import struct


def write_mulaw_wav(path: str, ulaw_bytes: bytes, sample_rate: int = 8000):
    """Write µ-law bytes into a WAV container (PCM mu-law, fmt=7)."""
    num_channels = 1
    bits_per_sample = 8  # µ-law is 8-bit
    byte_rate = sample_rate * num_channels  # 1 byte per sample
    block_align = num_channels
    data_size = len(ulaw_bytes)
    riff_size = 36 + data_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack('<I', riff_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack('<I', 16))              # PCM fmt chunk size
        f.write(struct.pack('<H', 0x0007))          # WAVE_FORMAT_MULAW
        f.write(struct.pack('<H', num_channels))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', byte_rate))
        f.write(struct.pack('<H', block_align))
        f.write(struct.pack('<H', bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack('<I', data_size))
        f.write(ulaw_bytes)

