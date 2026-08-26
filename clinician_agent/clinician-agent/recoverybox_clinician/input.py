"""Parse and bound the only data accepted by the clinician review app."""

from __future__ import annotations

import json
import re
from typing import Any

from .models import SessionSummary, ValidationError

_EMAIL_PATTERN = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\s().-]*){8,}(?!\d)")
_MAX_QUESTION_LENGTH = 500
_MAX_PAYLOAD_BYTES = 1_000_000
_MAX_SESSIONS = 1_000


def parse_question(value: object) -> str:
    """Validate a bounded prompt and reject obvious direct identifiers."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("agent.input must be a non-empty question")
    question = " ".join(value.split())
    if len(question) > _MAX_QUESTION_LENGTH:
        raise ValidationError(f"agent.input must be at most {_MAX_QUESTION_LENGTH} characters")
    if _EMAIL_PATTERN.search(question) or _PHONE_PATTERN.search(question):
        raise ValidationError("agent.input cannot contain email addresses or phone numbers")
    return question


def parse_session_summaries_json(value: object) -> tuple[SessionSummary, ...]:
    """Load strict summaries from a JSON array and reject duplicate session IDs."""
    if not isinstance(value, str):
        raise ValidationError("agent.session-summaries-json must be a JSON string")
    if len(value.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
        raise ValidationError("session summary payload is too large")

    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationError("agent.session-summaries-json must contain valid JSON") from exc
    if not isinstance(parsed, list):
        raise ValidationError("session summary payload must be a JSON array")
    if len(parsed) > _MAX_SESSIONS:
        raise ValidationError(f"at most {_MAX_SESSIONS} session summaries are accepted")

    sessions = tuple(SessionSummary.from_mapping(item) for item in parsed)
    session_ids = [session.session_id for session in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise ValidationError("session_id values must be unique")
    return sessions
