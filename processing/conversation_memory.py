from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from utils.logging import get_logger

_LOG = get_logger("processing.memory")
_LOCK = Lock()
_BASE_DIR = Path(os.getenv("CONVERSATION_SUMMARY_DIR", "logs/conversation_summaries"))


def _summary_path(phone: str) -> Path:
    phone_slug = "".join(ch for ch in str(phone) if ch.isdigit()) or str(phone).strip()
    if not phone_slug:
        phone_slug = "unknown"
    return _BASE_DIR / f"{phone_slug}.json"


def _normalize_summary(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(summary, dict):
        return None

    highlights_raw = summary.get("highlights")
    if highlights_raw is None:
        highlights_raw = summary.get("text")
    highlights = str(highlights_raw or "").strip()

    answered_raw = (
        summary.get("answered_questions")
        or summary.get("answered_question_ids")
        or summary.get("answered_ids")
        or []
    )
    answered: list[str] = []
    for value in answered_raw:
        text = str(value or "").strip()
        if not text:
            continue
        if text not in answered:
            answered.append(text)

    return {
        "highlights": highlights,
        "answered_questions": answered,
    }


def save_summary(phone: str, summary: Dict[str, Any], *, call_id: Optional[str] = None) -> None:
    if not phone or not isinstance(summary, dict):
        return

    normalized = _normalize_summary(summary)
    if normalized is None:
        return

    try:
        _BASE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _LOG.warning("Failed to create summary directory", error=str(exc), path=str(_BASE_DIR))
        return

    payload = {
        "phone": phone,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": normalized,
    }
    if call_id:
        payload["call_id"] = call_id

    path = _summary_path(phone)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    with _LOCK:
        try:
            path.write_text(serialized, encoding="utf-8")
        except Exception as exc:
            _LOG.warning("Failed to persist conversation summary", error=str(exc), path=str(path))


def load_summary(phone: str) -> Optional[Dict[str, Any]]:
    if not phone:
        return None

    path = _summary_path(phone)
    if not path.is_file():
        return None

    with _LOCK:
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception as exc:
            _LOG.warning("Failed to read conversation summary", error=str(exc), path=str(path))
            return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        _LOG.warning("Failed to parse conversation summary", error=str(exc), path=str(path))
        return None

    summary = _normalize_summary(data.get("summary") or {})
    if summary is None:
        return None

    result = {
        "phone": data.get("phone") or phone,
        "updated_at": data.get("updated_at"),
        "call_id": data.get("call_id"),
        "summary": summary,
    }
    return result
