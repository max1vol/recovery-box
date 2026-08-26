"""Button-turn orchestration around the pure Realtime protocol modules."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from recoverybox.core import DEFAULT_CUE_CATALOG, ApprovedCueCatalog
from recoverybox.session import (
    DEFAULT_CUE_CATALOG_VERSION,
    ApprovedCuePlaybackAuthorization,
)

from .prompts import (
    build_prompt_cue_response_instructions,
    build_realtime_session_instructions,
)
from .protocol import (
    ParsedServerEvent,
    ServerEventKind,
    build_assistant_item_truncate,
    build_audio_append,
    build_audio_clear,
    build_audio_commit,
    build_function_call_output,
    build_prompt_cue_response_create,
    build_response_cancel,
    build_response_create,
    build_session_update,
    parse_server_event,
)
from .safety import (
    AudioGateError,
    ConversationMode,
    ModelAudioGate,
    ModelAudioPolicy,
    ReleasedModelAudio,
)
from .tools import (
    ToolRegistry,
    ToolValidationError,
    ValidatedToolCall,
    deduplicate_validated_tool_calls,
    extract_validated_tool_calls,
)
from .transport import RealtimeTransport


@dataclass(frozen=True, slots=True)
class RejectedToolCall:
    """Safe diagnostic; raw model arguments are intentionally not retained."""

    reason: str


@dataclass(frozen=True, slots=True)
class RealtimeClientResult:
    event: ParsedServerEvent
    released_audio: tuple[ReleasedModelAudio, ...] = ()
    validated_tool_calls: tuple[ValidatedToolCall, ...] = ()
    rejected_tool_call: RejectedToolCall | None = None
    response_authorized: bool = False


class RealtimeSession:
    """Synchronous session controller with an injectable JSON transport.

    Hardware owns capture/playback.  This class sends complete button-defined
    user turns and returns only audio approved by :class:`ModelAudioGate`.
    """

    def __init__(
        self,
        *,
        transport: RealtimeTransport,
        tools: ToolRegistry | None = None,
        audio_gate: ModelAudioGate | None = None,
        cue_catalog: ApprovedCueCatalog = DEFAULT_CUE_CATALOG,
        cue_catalog_version: str = DEFAULT_CUE_CATALOG_VERSION,
    ) -> None:
        if not cue_catalog_version.strip():
            raise ValueError("cue_catalog_version must not be blank")
        self.transport = transport
        self.tools = tools or ToolRegistry(())
        self.audio_gate = audio_gate or ModelAudioGate()
        self.cue_catalog = cue_catalog
        self.cue_catalog_version = cue_catalog_version.strip()
        self.current_response_id: str | None = None
        self.current_assistant_item_id: str | None = None
        self._state_lock = threading.RLock()
        self._seen_tool_call_ids: set[str] = set()
        self._submitted_tool_call_ids: set[str] = set()

    def configure(self, *, instructions: str, voice: str) -> None:
        self.transport.send_event(
            build_session_update(
                instructions=build_realtime_session_instructions(
                    instructions,
                    self.cue_catalog,
                ),
                voice=voice,
                tools=self.tools.wire_tools,
            )
        )

    def append_user_audio(self, pcm16_mono_24khz: bytes) -> None:
        self.transport.send_event(build_audio_append(pcm16_mono_24khz))

    def commit_user_turn(self) -> None:
        """Commit the button-captured buffer without automatically responding."""

        self.transport.send_event(build_audio_commit())

    def request_response(
        self,
        *,
        mode: ConversationMode,
        policy: ModelAudioPolicy = ModelAudioPolicy.TRANSCRIPT_QUARANTINE,
        authorized_text: str | None = None,
        instructions: str | None = None,
    ) -> None:
        """Request a response after establishing its local release policy.

        The strict quarantine lane automatically asks for the exact authorized
        sentence when no narrower response instruction is supplied.  The
        conversational streaming lane remains explicit at the call site.  A
        live request must reach a terminal response or be cancelled before a
        second live request is queued; cancellation tombstones do not block a
        later request.
        """

        with self._state_lock:
            event = self._prepare_response_request(
                mode=mode,
                policy=policy,
                authorized_text=authorized_text,
                instructions=instructions,
            )
            self.audio_gate.authorize_next_response(
                mode=mode,
                policy=policy,
                authorized_text=authorized_text,
            )
            try:
                self.transport.send_event(event)
            except Exception:
                # A send failure can mean the event was written before the
                # connection failed. Preserve a tombstone rather than letting
                # an uncertain late creation steal a future authorization.
                self.audio_gate.cancel_pending_authorizations()
                raise

    def request_approved_prompt_cue(
        self,
        authorization: ApprovedCuePlaybackAuthorization,
    ) -> None:
        """Request one Guardian-authorized catalog phrase as isolated audio.

        The caller supplies no text.  This method revalidates the typed cue
        capability, resolves the phrase from its own catalog, disables tools
        for the response, and opens only the active-exercise exact-transcript
        quarantine lane.
        """

        if not isinstance(authorization, ApprovedCuePlaybackAuthorization):
            raise TypeError("authorization must be ApprovedCuePlaybackAuthorization")
        if authorization.catalog_version != self.cue_catalog_version:
            raise AudioGateError("prompt cue catalog version does not match Realtime session")
        try:
            cue = self.cue_catalog[authorization.cue_id.value]
        except KeyError as exc:
            raise AudioGateError("prompt cue is not in the Realtime catalog") from exc
        if cue.kind is not authorization.cue_kind:
            raise AudioGateError("prompt cue kind does not match the Realtime catalog")

        instructions = build_prompt_cue_response_instructions(cue)
        event = build_prompt_cue_response_create(
            cue_id=cue.cue_id,
            catalog_version=self.cue_catalog_version,
            instructions=instructions,
        )
        with self._state_lock:
            if self.audio_gate.open_authorizations:
                raise AudioGateError(
                    "a response authorization is already awaiting a terminal event"
                )
            self.audio_gate.validate_next_response(
                mode=ConversationMode.ACTIVE_EXERCISE,
                policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
                authorized_text=cue.spoken_text,
            )
            self.audio_gate.authorize_next_response(
                mode=ConversationMode.ACTIVE_EXERCISE,
                policy=ModelAudioPolicy.PROMPT_CUE_QUARANTINE,
                authorized_text=cue.spoken_text,
            )
            try:
                self.transport.send_event(event)
            except Exception:
                self.audio_gate.cancel_pending_authorizations()
                raise

    def finish_user_turn_and_request(
        self,
        *,
        mode: ConversationMode,
        policy: ModelAudioPolicy = ModelAudioPolicy.TRANSCRIPT_QUARANTINE,
        authorized_text: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self.commit_user_turn()
        self.request_response(
            mode=mode,
            policy=policy,
            authorized_text=authorized_text,
            instructions=instructions,
        )

    def handle_event(self, raw: Mapping[str, Any]) -> RealtimeClientResult:
        """Validate one event, then route tools and model audio safely."""

        parsed = parse_server_event(raw)
        with self._state_lock:
            rejected: RejectedToolCall | None = None
            calls: tuple[ValidatedToolCall, ...] = ()
            if raw.get("type") in {
                "response.function_call_arguments.done",
                "response.done",
            }:
                try:
                    validated = extract_validated_tool_calls(raw, self.tools)
                    calls = deduplicate_validated_tool_calls(
                        validated,
                        self._seen_tool_call_ids,
                    )
                except ToolValidationError as exc:
                    # Do not echo raw model-supplied JSON into diagnostics.
                    rejected = RejectedToolCall(reason=str(exc))

            released = self.audio_gate.ingest(parsed)
            response_authorized = False
            if parsed.kind is ServerEventKind.RESPONSE_CREATED:
                response_authorized = self.audio_gate.is_response_authorized(parsed.response_id)
                if response_authorized:
                    self.current_response_id = parsed.response_id
            elif parsed.kind is ServerEventKind.AUDIO_DELTA:
                if self.audio_gate.is_response_authorized(parsed.response_id):
                    self.current_assistant_item_id = parsed.item_id
            elif parsed.kind is ServerEventKind.RESPONSE_DONE:
                if parsed.response_id == self.current_response_id:
                    self.current_response_id = None
                    self.current_assistant_item_id = None

            return RealtimeClientResult(
                event=parsed,
                released_audio=released,
                validated_tool_calls=calls,
                rejected_tool_call=rejected,
                response_authorized=response_authorized,
            )

    def receive_once(self) -> RealtimeClientResult:
        return self.handle_event(self.transport.receive_event())

    def cancel_and_truncate(
        self,
        *,
        played_audio_ms: int,
        response_id: str | None = None,
        assistant_item_id: str | None = None,
        content_index: int = 0,
    ) -> None:
        """Cancel generation and truncate history to what the speaker played.

        Physical playback must be stopped by the caller before invoking this
        method.  The local gate is closed before network events are sent so
        stale deltas cannot become audible during interruption.
        """

        response_id = response_id or self.current_response_id
        assistant_item_id = assistant_item_id or self.current_assistant_item_id
        self.cancel_response(response_id=response_id)
        if assistant_item_id is not None:
            self.truncate_assistant(
                item_id=assistant_item_id,
                audio_end_ms=played_audio_ms,
                content_index=content_index,
            )

    def cancel_response(self, *, response_id: str | None = None) -> None:
        """Cancel active generation and close its local release authorization."""

        with self._state_lock:
            response_id = response_id or self.current_response_id
            if response_id is not None and self.audio_gate.is_response_authorized(response_id):
                self.audio_gate.discard_response(response_id)
            else:
                # Before response.created, no server response ID exists. Keep
                # the cancelled request's FIFO position as a tombstone so a
                # late creation cannot bind to the following turn.
                self.audio_gate.cancel_pending_authorizations()
                if response_id is not None:
                    self.audio_gate.discard_response(response_id)
            self.current_response_id = None
            self.current_assistant_item_id = None
            self.transport.send_event(build_response_cancel(response_id=response_id))

    def revoke_pending_response_locally(self) -> int:
        """Tombstone unbound work without cancelling default conversation audio.

        An out-of-band prompt cue has no safe wire cancellation target until
        ``response.created`` supplies its response ID. Cue delivery revokes the
        pending gate first and sends an ID-scoped cancel after that late
        creation is correlated.
        """

        with self._state_lock:
            cancelled = self.audio_gate.cancel_pending_authorizations()
            self.current_response_id = None
            self.current_assistant_item_id = None
            return cancelled

    def truncate_assistant(
        self, *, item_id: str, audio_end_ms: int, content_index: int = 0
    ) -> None:
        """Truncate by assistant item ID, never by response ID."""

        self.transport.send_event(
            build_assistant_item_truncate(
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
        )

    def clear_input_audio(self) -> None:
        self.transport.send_event(build_audio_clear())

    def submit_tool_output(self, *, call_id: str, output: Mapping[str, Any]) -> None:
        """Submit output only for a call previously validated by this session.

        This method serializes caller-provided data; it never imports, resolves,
        or executes model-selected code.
        """

        with self._state_lock:
            event = self._prepare_tool_output(call_id=call_id, output=output)
            self.transport.send_event(event)
            self._submitted_tool_call_ids.add(call_id)

    def submit_tool_output_and_request(
        self,
        *,
        call_id: str,
        output: Mapping[str, Any],
        mode: ConversationMode,
        policy: ModelAudioPolicy = ModelAudioPolicy.TRANSCRIPT_QUARANTINE,
        authorized_text: str | None = None,
        instructions: str | None = None,
    ) -> None:
        """Complete a validated tool call and request its authorized follow-up.

        Tool execution deliberately remains outside this class.  The caller
        applies deterministic application logic, supplies a JSON-compatible
        output mapping, and selects the same local audio policy used by an
        ordinary response.  The wire order is function output, local
        authorization, then ``response.create``.
        """

        with self._state_lock:
            tool_event = self._prepare_tool_output(call_id=call_id, output=output)
            response_event = self._prepare_response_request(
                mode=mode,
                policy=policy,
                authorized_text=authorized_text,
                instructions=instructions,
            )

            self.transport.send_event(tool_event)
            self._submitted_tool_call_ids.add(call_id)
            self.audio_gate.authorize_next_response(
                mode=mode,
                policy=policy,
                authorized_text=authorized_text,
            )
            try:
                self.transport.send_event(response_event)
            except Exception:
                self.audio_gate.cancel_pending_authorizations()
                raise

    def close(self) -> None:
        with self._state_lock:
            self.audio_gate.discard_all()
            self.current_response_id = None
            self.current_assistant_item_id = None
            self.transport.close()

    def _prepare_response_request(
        self,
        *,
        mode: ConversationMode,
        policy: ModelAudioPolicy,
        authorized_text: str | None,
        instructions: str | None,
    ) -> dict[str, Any]:
        if policy is ModelAudioPolicy.PROMPT_CUE_QUARANTINE:
            raise AudioGateError("prompt cue responses require request_approved_prompt_cue")
        if self.audio_gate.open_authorizations:
            raise AudioGateError("a response authorization is already awaiting a terminal event")
        self.audio_gate.validate_next_response(
            mode=mode,
            policy=policy,
            authorized_text=authorized_text,
        )
        if instructions is None and policy is ModelAudioPolicy.TRANSCRIPT_QUARANTINE:
            instructions = _exact_text_instruction(authorized_text or "")
        elif instructions is None and policy is ModelAudioPolicy.NO_AUDIO:
            instructions = (
                "Use available function tools only. Do not produce spoken or audio output."
            )
        return build_response_create(instructions=instructions)

    def _prepare_tool_output(
        self,
        *,
        call_id: str,
        output: Mapping[str, Any],
    ) -> dict[str, Any]:
        if call_id not in self._seen_tool_call_ids:
            raise ToolValidationError("tool call id was not locally validated")
        if call_id in self._submitted_tool_call_ids:
            raise ToolValidationError("tool call output was already submitted")
        output_json = json.dumps(
            dict(output), separators=(",", ":"), sort_keys=True, allow_nan=False
        )
        return build_function_call_output(call_id=call_id, output_json=output_json)


def _exact_text_instruction(authorized_text: str) -> str:
    quoted = json.dumps(authorized_text, ensure_ascii=False)
    return (
        "Speak exactly the following locally authorized text. Do not add, remove, "
        f"or paraphrase any words: {quoted}"
    )
