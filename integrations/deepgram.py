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

        utterances = payload.get("results", {}).get("utterances") or []
        # Normalize keys we rely on downstream
        normalized = []
        for item in utterances:
            text = item.get("transcript", "").strip()
            if not text:
                continue
            normalized.append(
                {
                    "text": text,
                    "speaker": item.get("speaker"),
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
            label = f"Speaker {speaker}" if speaker is not None else "Caller"
            start = entry.get("start")
            ts = f"{start:.1f}s" if isinstance(start, (int, float)) else "?"
            text = entry.get("text", "")
            lines.append(f"[{label} @ {ts}] {text}")
        return "\n".join(lines)