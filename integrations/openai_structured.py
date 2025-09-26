from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.logging import get_logger, exception


class OpenAIExtractionError(RuntimeError):
    """Raised when the OpenAI structured extraction fails."""


class GPTStructuredExtractor:
    """Calls the OpenAI Responses API to obtain structured answers for AmoCRM fields."""

    RESPONSES_URL = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        api_key: str,
        questions: list[dict],
        *,
        model: str = "gpt-5-mini",
        timeout: float = 120.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        if not questions:
            raise ValueError("Questions metadata must be supplied")

        self._api_key = api_key
        self._model = model
        self._questions = questions
        self._client = httpx.Client(timeout=httpx.Timeout(timeout))
        self._log = get_logger("integrations.openai")
        self._question_text = self._render_questions_for_prompt(questions)
        self._system_prompt = self._load_system_prompt()
        self._schema = self._build_schema()

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _render_questions_for_prompt(self, sections: Iterable[dict]) -> str:
        lines: list[str] = []
        for section in sections:
            section_name = section.get("name") or "Unnamed"
            lines.append(f"Раздел: {section_name}")
            for q in section.get("questions", []):
                name = q.get("name") or "?"
                qtype = q.get("type") or "text"
                qid = q.get("id")
                comment = q.get("comment") or ""
                lines.append(f"- ID {qid}: {name} (type={qtype})")
                if comment:
                    lines.append(f"  комментарий: {comment}")
                enums = q.get("enums") or []
                if enums:
                    enum_parts = [f"{opt['id']} -> {opt['value']}" for opt in enums if 'id' in opt and 'value' in opt]
                    if enum_parts:
                        lines.append("  варианты для ответа: " + ", ".join(enum_parts))
        return "\n".join(lines)

    def _build_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "integer"},
                            "text": {"type": ["string", "null"]},
                            "enum_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 0,
                            },
                            "confidence": {"type": ["number", "null"]},
                        },
                        "required": ["question_id", "text", "enum_ids", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["answers"],
            "additionalProperties": False,
        }

    @retry(
        retry=retry_if_exception_type((OpenAIExtractionError, httpx.HTTPError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def extract_fields(self, transcript_text: str, utterances: Iterable[dict]) -> dict:
        """Return the structured answers extracted from transcript."""

        utterances_json = json.dumps(list(utterances), ensure_ascii=False)
        user_prompt = (
            "Questions to fill (each question_id matches AmoCRM field id):\n"
            f"{self._question_text}\n\n"
            "Transcript (flattened):\n"
            f"{transcript_text}\n\n"
            "Transcript utterances (JSON):\n"
            f"{utterances_json}"
        )

        payload = {
            "model": self._model,
            "input": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "amocrm_answers",
                    "strict": True,
                    "schema": self._schema,
                }
            },
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = self._client.post(self.RESPONSES_URL, json=payload, headers=headers)
        except httpx.HTTPError:
            exception(self._log, "OpenAI request failed")
            raise

        if response.status_code >= 400:
            raise OpenAIExtractionError(f"OpenAI error {response.status_code}: {response.text}")

        data = response.json()
        answers = self._extract_output_text(data)
        if answers is None:
            raise OpenAIExtractionError("No structured output returned by OpenAI")

        try:
            parsed = json.loads(answers)
        except json.JSONDecodeError as exc:
            raise OpenAIExtractionError("Failed to parse OpenAI structured output") from exc
        return parsed

    def _load_system_prompt(self, prompt_path: Optional[Path] = None) -> str:
        path = prompt_path or Path(__file__).with_name("openai_system_prompt.md")
        try:
            content = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(
                "System prompt file not found. Please create 'openai_system_prompt.md'."
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                "Failed to read system prompt file 'openai_system_prompt.md'."
            ) from exc

        if not content:
            raise RuntimeError("System prompt file 'openai_system_prompt.md' is empty.")

        return content

    def _extract_output_text(self, payload: Dict[str, Any]) -> str | None:
        """Traverse Responses API output and return the structured JSON text."""

        outputs = payload.get("output") or []
        for item in outputs:
            if item.get("type") != "message":
                continue
            for chunk in item.get("content", []):
                if chunk.get("type") == "refusal":
                    detail = chunk.get("refusal") or "Request refused"
                    raise OpenAIExtractionError(detail)
                if chunk.get("type") == "output_text":
                    return chunk.get("text")
        return None