from __future__ import annotations

import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import httpx

from utils.logging import bind, exception, get_logger

from integrations.amocrm import AmoCRMClient, AmoCRMError
from integrations.deepgram import DeepgramClient
from integrations.openai_structured import GPTStructuredExtractor


class CallProcessingPipeline:
    """Coordinates post-call processing: transcription, extraction, CRM update."""

    def __init__(
        self,
        *,
        deepgram: DeepgramClient,
        extractor: GPTStructuredExtractor,
        amocrm: AmoCRMClient,
        questions: list[dict],
        max_workers: int = 12,
        audit_log_path: str | Path | None = None,
    ) -> None:
        self._deepgram = deepgram
        self._extractor = extractor
        self._amocrm = amocrm
        self._questions = questions
        self._question_index: Dict[int, dict] = {
            int(q["id"]): q
            for section in questions
            for q in section.get("questions", [])
            if "id" in q
        }
        self._log = get_logger("processing.pipeline")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline")
        self._active_numbers: set[str] = set()
        self._lock = threading.Lock()
        self._audit_lock = threading.Lock()
        default_dir = Path("logs") / "postprocessing"
        raw_path = Path(audit_log_path) if audit_log_path else default_dir
        self._audit_prefix = "postprocessing"

        if raw_path.suffix:
            if raw_path.stem:
                self._audit_prefix = raw_path.stem
            audit_dir = raw_path.parent
        else:
            audit_dir = raw_path
            if raw_path.name:
                self._audit_prefix = raw_path.name

        self._audit_dir: Path | None = audit_dir
        try:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            self._log.warning(
                "Failed to ensure audit log directory",
                path=str(self._audit_dir),
            )
            self._audit_dir = None

    def is_processing(self, phone: str) -> bool:
        normalized = phone or ""
        with self._lock:
            return normalized in self._active_numbers

    def submit(self, phone: str, wav_path: str, *, call_id: Optional[str] = None) -> None:
        if not phone or not wav_path:
            self._log.warning("Skipping pipeline submission due to missing phone/path", phone=phone, path=wav_path)
            return

        normalized = phone
        with self._lock:
            if normalized in self._active_numbers:
                self._log.info("Pipeline already running for phone", phone=normalized)
                return
            self._active_numbers.add(normalized)

        log = bind(self._log, phone=normalized, call_id=str(call_id or "?"))
        log.info("Pipeline scheduled", path=wav_path)
        future = self._executor.submit(self._run_pipeline, normalized, wav_path, log)
        future.add_done_callback(lambda f: self._on_done(normalized, f))

    def _on_done(self, phone: str, future: Future) -> None:
        try:
            # Trigger exception re-raise if task failed
            future.result()
        except Exception:
            exception(self._log, "Pipeline task failed", phone=phone)
        finally:
            with self._lock:
                self._active_numbers.discard(phone)

    def _run_pipeline(self, phone: str, wav_path: str, log) -> None:
        transcript_text: str | None = None
        structured: dict | None = None
        custom_fields: list[dict] | None = None
        status = "started"
        lead_id: Optional[int] = None
        call_id = None
        prompt_context: str | None = None
        if hasattr(log, "extra"):
            try:
                call_id = log.extra.get("call_id")  # type: ignore[attr-defined]
            except Exception:
                call_id = None
        try:
            self._ensure_recording_ready(wav_path, log)
            payload = self._deepgram.transcribe(wav_path)
            utterances = self._deepgram.extract_utterances(payload)
            transcript_text = self._deepgram.flatten_utterances(utterances)
            prompt_context = (
                "Transcript (flattened):\n"
                f"{self._sanitize_transcript(transcript_text)}"
            )

            structured = self._extractor.extract_fields(transcript_text, utterances)
            custom_fields = self._convert_to_custom_fields(structured)
            if not custom_fields:
                log.warning("Structured output produced no fields; skipping CRM update")
                status = "no_fields"
                self._write_audit_entry(
                    phone,
                    call_id,
                    transcript_text,
                    structured,
                    custom_fields,
                    status=status,
                    prompt_context=prompt_context,
                )
                return

            try:
                contact = self._amocrm.find_contact_by_phone(phone)
                if not contact:
                    log.warning("No contact found for phone; aborting CRM update")
                    status = "contact_not_found"
                    self._write_audit_entry(
                        phone,
                        call_id,
                        transcript_text,
                        structured,
                        custom_fields,
                        status=status,
                        prompt_context=prompt_context,
                    )
                    return

                lead_id = self._amocrm.pick_lead_id(contact)
                if not lead_id:
                    log.warning("Contact has no associated lead; aborting CRM update")
                    status = "lead_not_found"
                    self._write_audit_entry(
                        phone,
                        call_id,
                        transcript_text,
                        structured,
                        custom_fields,
                        status=status,
                        prompt_context=prompt_context,
                    )
                    return

                self._amocrm.update_lead_custom_fields(lead_id, custom_fields)
                log.info("CRM update completed", lead_id=str(lead_id))
                status = "success"
                self._write_audit_entry(
                    phone,
                    call_id,
                    transcript_text,
                    structured,
                    custom_fields,
                    status=status,
                    lead_id=lead_id,
                    prompt_context=prompt_context,
                )
            except (AmoCRMError, httpx.HTTPError) as exc:
                log.warning(
                    "Skipping CRM update due to AmoCRM connectivity issue",
                    error=str(exc),
                )
                status = "crm_unreachable"
                self._write_audit_entry(
                    phone,
                    call_id,
                    transcript_text,
                    structured,
                    custom_fields,
                    status=status,
                    lead_id=lead_id,
                    error=str(exc),
                    prompt_context=prompt_context,
                )
                return
        except Exception as exc:
            exception(log, "Pipeline execution error")
            status = "error"
            self._write_audit_entry(
                phone,
                call_id,
                transcript_text,
                structured,
                custom_fields,
                status=status,
                lead_id=lead_id,
                error=str(exc),
                prompt_context=prompt_context,
            )
            raise

    def _write_audit_entry(
        self,
        phone: str,
        call_id: Optional[str],
        transcript: Optional[str],
        structured: Optional[dict],
        custom_fields: Optional[list[dict]],
        *,
        status: str,
        lead_id: Optional[int] = None,
        error: Optional[str] = None,
        prompt_context: Optional[str] = None,
    ) -> None:
        if not self._audit_dir:
            return

        timestamp = datetime.now(timezone.utc)
        entry = {
            "timestamp": timestamp.isoformat(),
            "phone": phone,
            "call_id": call_id,
            "status": status,
            "deepgram_transcript": transcript,
            "gpt_structured": structured,
            "amo_custom_fields": custom_fields,
            "lead_id": lead_id,
            "prompt_transcript_context": prompt_context,
        }
        if error:
            entry["error"] = error

        try:
            serialized = json.dumps(entry, ensure_ascii=False)
        except Exception:
            # Fallback: attempt to coerce problematic fields to string
            safe_entry = dict(entry)
            if structured is not None:
                safe_entry["gpt_structured"] = json.loads(json.dumps(structured, default=str))
            if custom_fields is not None:
                safe_entry["amo_custom_fields"] = json.loads(json.dumps(custom_fields, default=str))
            serialized = json.dumps(safe_entry, ensure_ascii=False)

        target_path = self._build_audit_path(timestamp, phone, call_id, status)
        if target_path is None:
            return

        with self._audit_lock:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with target_path.open("w", encoding="utf-8") as fp:
                    fp.write(serialized)
            except Exception:
                self._log.warning("Failed to write audit log", path=str(target_path))

    def _build_audit_path(
        self,
        timestamp: datetime,
        phone: Optional[str],
        call_id: Optional[str],
        status: str,
    ) -> Optional[Path]:
        if not self._audit_dir:
            return None

        prefix = self._audit_prefix or "postprocessing"
        prefix_slug = "".join(ch for ch in prefix if ch.isalnum() or ch in {"-", "_"}) or "postprocessing"
        ts_slug = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        phone_raw = phone or "unknown"
        phone_slug = "".join(ch for ch in str(phone_raw) if ch.isdigit())
        if not phone_slug:
            phone_slug = "unknown"
        slug_parts = [prefix_slug, ts_slug, phone_slug]

        if call_id:
            call_slug = "".join(ch for ch in str(call_id) if ch.isalnum())
            if call_slug:
                slug_parts.append(call_slug)

        status_slug = "".join(ch for ch in status if ch.isalnum() or ch in {"-", "_"}) or "status"
        slug_parts.append(status_slug)

        filename = "_".join(slug_parts) + ".json"
        return self._audit_dir / filename

    def _sanitize_transcript(self, transcript: Optional[str]) -> str:
        if not transcript:
            return ""

        pattern = re.compile(r"^\[(?P<label>[^\]@]+)(?:\s*@[^\]]+)?\]\s*(?P<text>.*)")
        cleaned_lines: list[str] = []
        for raw_line in transcript.splitlines():
            match = pattern.match(raw_line)
            if match:
                label = (match.group("label") or "").strip() or "Спикер"
                text = (match.group("text") or "").strip()
                cleaned_lines.append(f"{label}: {text}" if text else f"{label}:")
            else:
                cleaned_lines.append(raw_line.strip())
        return "\n".join(cleaned_lines)

    def _ensure_recording_ready(self, wav_path: str, log) -> None:
        path = Path(wav_path)
        deadline = time.time() + 10.0
        last_size = -1
        while time.time() < deadline:
            if path.is_file():
                size = path.stat().st_size
                if size > 0 and size == last_size:
                    return
                last_size = size
            time.sleep(0.3)
        log.warning("Recording may be incomplete", path=str(path), size=str(path.stat().st_size if path.exists() else 0))

    def _convert_to_custom_fields(self, structured: dict) -> list[dict]:
        answers = structured.get("answers") or []
        result: list[dict] = []
        for answer in answers:
            qid = answer.get("question_id")
            if qid is None:
                continue
            try:
                qid_int = int(qid)
            except (TypeError, ValueError):
                continue
            question = self._question_index.get(qid_int)
            if not question:
                continue

            qtype = (question.get("type") or "text").lower()
            text = answer.get("text")
            if isinstance(text, str):
                text = text.strip()
            enum_ids = answer.get("enum_ids") or []

            if qtype in {"text", "textarea"}:
                if not text:
                    continue
                result.append({
                    "field_id": qid_int,
                    "values": [{"value": text}],
                })
            elif qtype == "select":
                enum_id = None
                if enum_ids:
                    enum_id = enum_ids[0]
                try:
                    enum_value = int(enum_id) if enum_id is not None else None
                except (TypeError, ValueError):
                    enum_value = None
                if enum_value is None:
                    continue
                result.append({
                    "field_id": qid_int,
                    "values": [{"enum_id": enum_value}],
                })
            elif qtype == "multiselect":
                valid_enums: list[int] = []
                for raw in enum_ids:
                    try:
                        val = int(str(raw))
                    except (TypeError, ValueError):
                        continue
                    valid_enums.append(val)
                if not valid_enums:
                    continue
                result.append({
                    "field_id": qid_int,
                    "values": [{"enum_id": int(eid)} for eid in valid_enums],
                })
            else:
                # Fallback: treat as text if we have a value
                if not text:
                    continue
                result.append({
                    "field_id": qid_int,
                    "values": [{"value": text}],
                })
        return result

    def shutdown(self) -> None:
        self._log.info("Shutting down pipeline")
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self._deepgram.close()
        except Exception:
            pass
        try:
            self._extractor.close()
        except Exception:
            pass
        try:
            self._amocrm.close()
        except Exception:
            pass