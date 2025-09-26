from __future__ import annotations

import re
from typing import Iterable, Optional

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from utils.logging import get_logger, exception


class AmoCRMError(RuntimeError):
    """Raised when the AmoCRM API returns an error."""


def _normalize_phone_candidates(phone: str) -> list[str]:
    digits = re.sub(r"\D", "", phone or "")
    candidates: list[str] = []
    if digits:
        candidates.append(digits)
        if digits.startswith("8") and len(digits) == 11:
            candidates.append("+7" + digits[1:])
            candidates.append(digits[1:])
        elif digits.startswith("7") and len(digits) == 11:
            candidates.append("+" + digits)
            candidates.append(digits[1:])
        elif digits.startswith("9") and len(digits) == 10:
            candidates.append("+7" + digits)
    return list(dict.fromkeys(candidates))  # dedupe preserving order


class AmoCRMClient:
    """Minimal AmoCRM REST client focused on contacts and deals."""

    def __init__(
        self,
        base_url: str,
        access_token: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("AMOCRM_BASE_URL must be set")
        if not access_token:
            raise ValueError("AMOCRM_ACCESS_TOKEN must be set")

        base_url = base_url.strip()
        if not re.match(r"^https?://", base_url, re.IGNORECASE):
            base_url = f"https://{base_url}"

        parsed = httpx.URL(base_url)
        if parsed.host is None:
            raise ValueError(
                "AMOCRM_BASE_URL must include a valid hostname, e.g. 'example.amocrm.ru'"
            )

        base_url = f"{parsed.scheme}://{parsed.host}"
        if parsed.port:
            base_url = f"{base_url}:{parsed.port}"
        self._client = httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self._log = get_logger("integrations.amocrm")

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.HTTPError:
            exception(self._log, "AmoCRM request failed", method=method, url=url)
            raise

        if response.status_code >= 400:
            text = response.text
            raise AmoCRMError(f"AmoCRM error {response.status_code}: {text}")
        return response

    @retry(
        retry=retry_if_exception_type((AmoCRMError, httpx.HTTPError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def find_contact_by_phone(self, phone: str) -> Optional[dict]:
        """Return the first contact matching the phone number (with leads embedded)."""

        candidates = _normalize_phone_candidates(phone)
        params = {"limit": 1, "with": "leads"}

        for query in candidates:
            params["query"] = query
            response = self._request("GET", "/api/v4/contacts", params=params)
            data = response.json()
            contacts = data.get("_embedded", {}).get("contacts") or []
            if contacts:
                contact = contacts[0]
                self._log.info("Found contact", phone=phone, contact_id=str(contact.get("id")))
                return contact

        self._log.warning("Contact not found", phone=phone)
        return None

    def pick_lead_id(self, contact: dict) -> Optional[int]:
        """Select the most relevant lead id from a contact payload."""

        leads = contact.get("_embedded", {}).get("leads") or []
        if not leads:
            return None

        # Prefer lead marked as main, otherwise the most recently updated
        main_leads = [lead for lead in leads if lead.get("is_main")]
        chosen = main_leads[0] if main_leads else leads[0]
        lead_id = chosen.get("id")
        try:
            return int(lead_id) if lead_id is not None else None
        except (TypeError, ValueError):
            return None

    @retry(
        retry=retry_if_exception_type((AmoCRMError, httpx.HTTPError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def update_lead_custom_fields(self, lead_id: int, custom_fields: Iterable[dict]) -> None:
        fields = [cf for cf in custom_fields if cf.get("values")]
        if not fields:
            self._log.info("No fields to push to AmoCRM", lead_id=str(lead_id))
            return

        payload = {"custom_fields_values": fields}
        self._log.info("Updating lead fields", lead_id=str(lead_id), count=str(len(fields)))
        self._request("PATCH", f"/api/v4/leads/{lead_id}", json=payload)