from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.logging import get_logger, exception


class DeepgramError(RuntimeError):
    """Raised when the Deepgram API returns an error response."""


class DeepgramClient:
    """HTTP client wrapper for Deepgram's transcription API."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-2",
        endpoint: str = "https://api.deepgram.com/v1/listen",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("Deepgram API key is required")

        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))
        self._log = get_logger("integrations.deepgram")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    @retry(
        retry=retry_if_exception_type((DeepgramError, httpx.HTTPError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def transcribe(self, wav_path: str) -> dict:
        """Transcribe a WAV audio file via Deepgram.

        Returns the parsed JSON response. Retries automatically on transient failures.
        """

        path = Path(wav_path)
        if not path.is_file():
            raise FileNotFoundError(f"Recording not found: {wav_path}")

        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "audio/wav",
        }
        params = {
            "model": self._model,
            "utterances": "true",
            "punctuate": "true",
            "language": "ru",
            "smart_format": "true",
            "profanity_filter": "false",
            "multichannel": "true",
            "diarize": "true",
        }

        self._log.info("Sending audio to Deepgram", path=str(path), size=str(path.stat().st_size))

        with path.open("rb") as audio_file:
            try:
                response = self._client.post(
                    self._endpoint,
                    headers=headers,
                    params=params,
                    data=audio_file,
                )
            except httpx.HTTPError as exc:
                exception(self._log, "Deepgram request failed", path=str(path))
                raise

        if response.status_code >= 400:
            detail: Optional[str]
            try:
                detail = response.json().get("error")
            except Exception:
                detail = response.text
            raise DeepgramError(f"Deepgram error {response.status_code}: {detail}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise DeepgramError("Failed to decode Deepgram response") from exc

        return payload

    @staticmethod
    def extract_utterances(payload: dict) -> list[dict]:
        """Return utterance-level transcripts from Deepgram response."""

        results = payload.get("results") or {}
        channels = results.get("channels") or []

        if channels:
            segments: list[dict] = []
            for idx, channel in enumerate(channels):
                alt = (channel.get("alternatives") or [{}])[0]
                words = alt.get("words") or []
                if idx == 0:
                    role = "Клиент"
                elif idx == 1:
                    role = "Ассистент"
                else:
                    role = f"Спикер {idx}"
                channel_segments = DeepgramClient._segments_from_words(words, role, idx, alt.get("confidence"))
                if not channel_segments:
                    transcript = (alt.get("transcript") or "").strip()
                    if transcript:
                        channel_segments.append(
                            {
                                "text": transcript,
                                "speaker": role,
                                "channel": idx,
                                "start": words[0].get("start") if words else None,
                                "end": words[-1].get("end") if words else None,
                                "confidence": alt.get("confidence"),
                            }
                        )
                segments.extend(channel_segments)

            segments.sort(key=lambda seg: seg.get("start") or 0.0)
            return segments

        utterances = results.get("utterances") or []
        normalized = []
        for item in utterances:
            text = item.get("transcript", "").strip()
            if not text:
                continue
            speaker = item.get("speaker")
            label = (
                "Клиент" if speaker == 0 else "Ассистент" if speaker == 1 else speaker
            )
            normalized.append(
                {
                    "text": text,
                    "speaker": label,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "confidence": item.get("confidence"),
                }
            )
        return normalized

    @staticmethod
    def flatten_utterances(utterances: Iterable[dict]) -> str:
        """Produce a readable text transcript from utterance list."""

        lines: list[str] = []
        for entry in utterances:
            speaker = entry.get("speaker")
            if speaker is None:
                label = "Caller"
            elif isinstance(speaker, str):
                label = speaker
            else:
                label = f"Speaker {speaker}"
            start = entry.get("start")
            ts = f"{start:.1f}s" if isinstance(start, (int, float)) else "?"
            text = entry.get("text", "")
            lines.append(f"[{label} @ {ts}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _segments_from_words(words: list[dict], role: str, channel_index: int, confidence: Optional[float] = None) -> list[dict]:
        if not words:
            return []

        segments: list[dict] = []
        buffer: list[str] = []
        start_time = None
        last_end = None
        GAP_THRESHOLD = 1.0

        for item in words:
            word = (item.get("word") or "").strip()
            if not word:
                continue
            w_start = item.get("start")
            w_end = item.get("end", w_start)
            if buffer and last_end is not None and w_start is not None and w_start - last_end > GAP_THRESHOLD:
                segments.append(
                    {
                        "text": DeepgramClient._join_words(buffer),
                        "speaker": role,
                        "channel": channel_index,
                        "start": start_time,
                        "end": last_end,
                        "confidence": confidence,
                    }
                )
                buffer = []
                start_time = None
                last_end = None

            if not buffer:
                start_time = w_start

            buffer.append(word)
            last_end = w_end if isinstance(w_end, (int, float)) else last_end

        if buffer:
            segments.append(
                {
                    "text": DeepgramClient._join_words(buffer),
                    "speaker": role,
                    "channel": channel_index,
                    "start": start_time,
                    "end": last_end,
                    "confidence": confidence,
                }
            )

        return segments

    @staticmethod
    def _join_words(words: list[str]) -> str:
        raw = " ".join(words)
        replacements = {
            " ,": ",",
            " .": ".",
            " !": "!",
            " ?": "?",
            " :": ":",
            " ;": ";",
        }
        for needle, repl in replacements.items():
            raw = raw.replace(needle, repl)
        raw = raw.replace(" n't", "n't")
        return raw.strip()

