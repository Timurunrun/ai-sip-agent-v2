from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from utils.logging import get_logger

_LOG = get_logger("processing.memory")
_LOCK = Lock()
_BASE_DIR = Path(os.getenv("CONVERSATION_SUMMARY_DIR", "logs/conversation_summaries"))


def _slug_for_phone(phone: str) -> str:
    phone_slug = "".join(ch for ch in str(phone) if ch.isdigit())
    if not phone_slug:
        phone_slug = str(phone).strip()
    if not phone_slug:
        phone_slug = "unknown"
    return phone_slug


def _legacy_path(slug: str) -> Path:
    return _BASE_DIR / f"{slug}.json"


def _indexed_path(slug: str, index: int) -> Path:
    return _BASE_DIR / f"{slug}_{index}.json"


def _extract_index(slug: str, path: Path) -> Optional[int]:
    stem = path.stem
    if stem == slug:
        return 0
    prefix = f"{slug}_"
    if stem.startswith(prefix):
        suffix = stem[len(prefix) :]
        try:
            return int(suffix)
        except ValueError:
            return None
    return None


def _list_summary_files(slug: str) -> list[Tuple[int, Path]]:
    candidates: list[Tuple[int, Path]] = []
    legacy = _legacy_path(slug)
    if legacy.is_file():
        candidates.append((0, legacy))
    for path in _BASE_DIR.glob(f"{slug}_*.json"):
        if not path.is_file():
            continue
        idx = _extract_index(slug, path)
        if idx is None:
            continue
        candidates.append((idx, path))
    candidates.sort(key=lambda item: item[0])
    return candidates


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


def save_summary(phone: str, summary: Dict[str, Any], *, call_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not phone or not isinstance(summary, dict):
        return None

    normalized = _normalize_summary(summary)
    if normalized is None:
        return None

    previous_entry = load_summary(phone)
    if previous_entry:
        previous_answers = previous_entry.get("summary", {}).get("answered_questions") or []
        combined: list[str] = []
        for answer in list(previous_answers) + list(normalized["answered_questions"]):
            answer_text = str(answer or "").strip()
            if not answer_text:
                continue
            if answer_text not in combined:
                combined.append(answer_text)
        normalized["answered_questions"] = combined

    slug = _slug_for_phone(phone)

    try:
        _BASE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        _LOG.warning("Failed to create summary directory", error=str(exc), path=str(_BASE_DIR))
        return normalized

    payload = {
        "phone": phone,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": normalized,
    }
    if call_id:
        payload["call_id"] = call_id

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    with _LOCK:
        legacy = _legacy_path(slug)
        target_legacy = _indexed_path(slug, 0)
        if legacy.is_file() and not target_legacy.exists():
            try:
                legacy.rename(target_legacy)
            except Exception as exc:
                _LOG.warning("Failed to migrate legacy summary file", error=str(exc), path=str(legacy))

        existing = _list_summary_files(slug)
        next_index = existing[-1][0] + 1 if existing else 0
        target_path = _indexed_path(slug, next_index)

        try:
            target_path.write_text(serialized, encoding="utf-8")
        except Exception as exc:
            _LOG.warning("Failed to persist conversation summary", error=str(exc), path=str(target_path))

    return normalized


def load_summary(phone: str) -> Optional[Dict[str, Any]]:
    if not phone:
        return None

    slug = _slug_for_phone(phone)

    with _LOCK:
        if not _BASE_DIR.exists():
            return None

        candidates = _list_summary_files(slug)
        if not candidates:
            return None

        index, path = max(candidates, key=lambda item: item[0])

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
