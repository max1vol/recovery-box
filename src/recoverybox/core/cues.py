"""Clinician-reviewable phrases that the exercise loop is allowed to speak."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class CueId(StrEnum):
    """Stable identifiers for the default approved cue set."""

    SQUAT_SET_INTRO = "squat_set_intro"
    SQUAT_PERSON_DETECTED = "squat_person_detected"
    READY = "ready"
    MOVE_SLOWLY = "move_slowly"
    KNEE_ALIGNMENT = "knee_alignment"
    COMFORTABLE_RANGE = "comfortable_range"
    HOLD_POSITION = "hold_position"
    CAMERA_PAUSE = "camera_pause"
    PAIN_STOP = "pain_stop"
    SESSION_COMPLETE = "session_complete"
    SQUAT_REP_ONE = "squat_rep_one"
    SQUAT_REP_TWO = "squat_rep_two"
    SQUAT_REP_THREE = "squat_rep_three"
    SQUAT_REP_FOUR = "squat_rep_four"
    SQUAT_REP_FIVE = "squat_rep_five"
    SQUAT_REP_SIX = "squat_rep_six"
    SQUAT_REP_SEVEN = "squat_rep_seven"
    SQUAT_REP_EIGHT = "squat_rep_eight"
    SQUAT_REP_NINE = "squat_rep_nine"
    SQUAT_REP_TEN = "squat_rep_ten"
    ARMS_T_SHAPE = "arms_t_shape"


class CueKind(StrEnum):
    """How an approved phrase is used by the device."""

    INSTRUCTION = "instruction"
    CORRECTION = "correction"
    STATUS = "status"
    SAFETY = "safety"


@dataclass(frozen=True, slots=True)
class ApprovedCue:
    """A fixed phrase whose wording can be reviewed before deployment."""

    cue_id: str
    spoken_text: str
    kind: CueKind

    def __post_init__(self) -> None:
        cue_id = self.cue_id.strip()
        spoken_text = self.spoken_text.strip()
        if not cue_id:
            raise ValueError("cue_id must not be empty")
        if not spoken_text:
            raise ValueError("spoken_text must not be empty")
        object.__setattr__(self, "cue_id", cue_id)
        object.__setattr__(self, "spoken_text", spoken_text)


class ApprovedCueCatalog(Mapping[str, ApprovedCue]):
    """Immutable lookup of phrases allowed to enter the exercise audio path."""

    def __init__(self, cues: Iterable[ApprovedCue]) -> None:
        by_id: dict[str, ApprovedCue] = {}
        for cue in cues:
            if cue.cue_id in by_id:
                raise ValueError(f"duplicate cue_id: {cue.cue_id}")
            by_id[cue.cue_id] = cue
        self._cues = MappingProxyType(by_id)

    def __getitem__(self, cue_id: str) -> ApprovedCue:
        return self._cues[cue_id]

    def __iter__(self) -> Iterator[str]:
        return iter(self._cues)

    def __len__(self) -> int:
        return len(self._cues)

    def is_approved(self, cue_id: str) -> bool:
        """Return whether ``cue_id`` has approved, fixed wording."""

        return cue_id in self._cues

    @property
    def content_sha256(self) -> str:
        """Return a deterministic digest of every safety-relevant cue field.

        Entries are ordered by cue ID so the digest identifies catalog content,
        not construction order.  Keeping an expected digest in a review test
        makes any identifier, kind, or literal phrase change explicit.
        """

        payload = [
            {
                "cue_id": cue.cue_id,
                "kind": cue.kind.value,
                "spoken_text": cue.spoken_text,
            }
            for cue in sorted(self._cues.values(), key=lambda cue: cue.cue_id)
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


DEFAULT_CUE_CATALOG = ApprovedCueCatalog(
    (
        ApprovedCue(
            CueId.SQUAT_SET_INTRO,
            "Hi Max. Let's start with a set of three squats.",
            CueKind.INSTRUCTION,
        ),
        ApprovedCue(
            CueId.SQUAT_PERSON_DETECTED,
            "I can see you. Now do the squats.",
            CueKind.INSTRUCTION,
        ),
        ApprovedCue(
            CueId.READY,
            "When you are ready, begin the movement.",
            CueKind.INSTRUCTION,
        ),
        ApprovedCue(
            CueId.MOVE_SLOWLY,
            "Move slowly and with control.",
            CueKind.CORRECTION,
        ),
        ApprovedCue(
            CueId.KNEE_ALIGNMENT,
            "Keep your knee aligned with your toes.",
            CueKind.CORRECTION,
        ),
        ApprovedCue(
            CueId.COMFORTABLE_RANGE,
            "Stay within a comfortable range.",
            CueKind.CORRECTION,
        ),
        ApprovedCue(
            CueId.HOLD_POSITION,
            "Hold that position briefly.",
            CueKind.CORRECTION,
        ),
        ApprovedCue(
            CueId.CAMERA_PAUSE,
            "Please pause while I check the camera view.",
            CueKind.SAFETY,
        ),
        ApprovedCue(
            CueId.PAIN_STOP,
            "Stop the exercise and rest.",
            CueKind.SAFETY,
        ),
        ApprovedCue(
            CueId.SESSION_COMPLETE,
            "That completes this session.",
            CueKind.STATUS,
        ),
        ApprovedCue(CueId.SQUAT_REP_ONE, "One.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_TWO, "Slower.", CueKind.CORRECTION),
        ApprovedCue(
            CueId.SQUAT_REP_THREE,
            "Three. Excellent. Now bring your arms out into a T shape.",
            CueKind.STATUS,
        ),
        ApprovedCue(CueId.SQUAT_REP_FOUR, "Four. Nice and steady.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_FIVE, "That's five. Halfway there.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_SIX, "That's six.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_SEVEN, "That's seven.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_EIGHT, "Eight. Two more.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_NINE, "That's nine.", CueKind.STATUS),
        ApprovedCue(CueId.SQUAT_REP_TEN, "That's ten. Set complete.", CueKind.STATUS),
        ApprovedCue(
            CueId.ARMS_T_SHAPE,
            "Bring your arms back out to a T shape.",
            CueKind.CORRECTION,
        ),
    )
)


# These two cues are the only ones that may be selected without an assessable
# movement observation.  The Guardian still validates them against both this
# closed set and the active exercise plan before it issues a CUE decision.
SQUAT_SCRIPTED_SESSION_CUE_IDS = frozenset(
    {
        CueId.SQUAT_SET_INTRO.value,
        CueId.SQUAT_PERSON_DETECTED.value,
    }
)
