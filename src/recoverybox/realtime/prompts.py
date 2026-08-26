"""Reviewable prompt text for RecoveryBox Realtime sessions."""

from __future__ import annotations

import json

from recoverybox.core import ApprovedCue, ApprovedCueCatalog


def build_realtime_session_instructions(
    base_instructions: str,
    cue_catalog: ApprovedCueCatalog,
) -> str:
    """Append the exact active-exercise cue catalog to a session prompt.

    The catalog is visible to reviewers as ordinary text, not as pre-rendered
    audio files.  Listing cues here does not authorize the model to choose one:
    the local Guardian still selects the cue ID and a response-level prompt
    supplies the corresponding phrase.
    """

    base = base_instructions.strip()
    if not base:
        raise ValueError("base_instructions must not be blank")
    if not cue_catalog:
        raise ValueError("cue_catalog must not be empty")

    cue_lines = [
        f"- {cue.cue_id}: {json.dumps(cue.spoken_text, ensure_ascii=False)}"
        for cue in cue_catalog.values()
    ]
    cue_section = "\n".join(
        (
            "ACTIVE_EXERCISE PROMPT CUES",
            "The application, not the model, decides whether a cue is allowed.",
            "Only speak a cue when response instructions name its ID and exact phrase.",
            "Never invent, combine, extend, or paraphrase these phrases:",
            *cue_lines,
        )
    )
    return f"{base}\n\n{cue_section}"


def build_prompt_cue_response_instructions(cue: ApprovedCue) -> str:
    """Build the response prompt for one Guardian-selected cue."""

    cue_id = json.dumps(cue.cue_id, ensure_ascii=False)
    phrase = json.dumps(cue.spoken_text, ensure_ascii=False)
    return (
        f"The local Guardian selected cue ID {cue_id}. "
        "Speak exactly the following approved phrase and nothing else. "
        f"Do not add, remove, combine, or paraphrase any words: {phrase}"
    )


__all__ = [
    "build_prompt_cue_response_instructions",
    "build_realtime_session_instructions",
]
