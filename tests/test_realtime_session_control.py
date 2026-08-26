from __future__ import annotations

import pytest

from recoverybox.realtime.session_control import (
    FINISH_SESSION_TOOL,
    FINISH_SESSION_TOOL_NAME,
    SESSION_CONTROL_TOOL_REGISTRY,
    RuntimeAbortReason,
    SessionControlError,
    SessionEndController,
    SessionEndSignal,
    SessionEndSource,
    validate_finish_session_call,
)
from recoverybox.realtime.tools import (
    ToolValidationError,
    ValidatedToolCall,
)


def _validated_call(*, call_id: str = "call-finish-1") -> ValidatedToolCall:
    return validate_finish_session_call(call_id=call_id, arguments_json="{}")


def test_finish_session_is_the_only_session_control_tool() -> None:
    assert SESSION_CONTROL_TOOL_REGISTRY.wire_tools == (FINISH_SESSION_TOOL.to_wire(),)
    assert FINISH_SESSION_TOOL.to_wire() == {
        "type": "function",
        "name": FINISH_SESSION_TOOL_NAME,
        "description": (
            "End the audio session only after the user explicitly asks to finish, "
            "leave, or says goodbye."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


@pytest.mark.parametrize(
    "arguments_json",
    [
        "not-json",
        '"bye"',
        '{"reason":"bye"}',
        '{"reason":"bye","reason":"again"}',
    ],
)
def test_malformed_or_nonempty_arguments_never_end_session(arguments_json: str) -> None:
    controller = SessionEndController()

    with pytest.raises(ToolValidationError):
        validate_finish_session_call(
            call_id="call-malformed",
            arguments_json=arguments_json,
        )

    assert controller.ended is False
    assert controller.end_signal is None


def test_unknown_tool_is_rejected_before_controller() -> None:
    controller = SessionEndController()

    with pytest.raises(ToolValidationError, match="unknown tool"):
        SESSION_CONTROL_TOOL_REGISTRY.validate_call(
            name="close_socket",
            call_id="call-unknown",
            arguments_json="{}",
        )

    assert controller.ended is False


@pytest.mark.parametrize(
    "model_output",
    [
        "okay, bye",
        {"name": FINISH_SESSION_TOOL_NAME, "arguments": {}},
        None,
    ],
)
def test_free_form_or_raw_model_output_cannot_end_session(model_output: object) -> None:
    controller = SessionEndController()

    with pytest.raises(TypeError, match="locally validated"):
        controller.accept_validated_tool_call(model_output)  # type: ignore[arg-type]

    assert controller.ended is False


def test_controller_defensively_rejects_wrong_or_nonempty_typed_calls() -> None:
    controller = SessionEndController()

    with pytest.raises(SessionControlError, match="cannot end"):
        controller.accept_validated_tool_call(
            ValidatedToolCall(name="report_symptom", call_id="call-1", arguments={})
        )
    with pytest.raises(SessionControlError, match="empty object"):
        controller.accept_validated_tool_call(
            ValidatedToolCall(
                name=FINISH_SESSION_TOOL_NAME,
                call_id="call-2",
                arguments={"reason": "bye"},
            )
        )

    assert controller.ended is False


def test_validated_tool_call_emits_exactly_one_end_signal() -> None:
    observed: list[SessionEndSignal] = []
    controller = SessionEndController(observed.append)

    first = controller.accept_validated_tool_call(_validated_call())
    duplicate = controller.accept_validated_tool_call(_validated_call())
    later_call = controller.accept_validated_tool_call(_validated_call(call_id="call-finish-2"))

    assert first is not None
    assert first.source is SessionEndSource.VALIDATED_TOOL_CALL
    assert first.tool_call_id == "call-finish-1"
    assert first.abort_reason is None
    assert duplicate is None
    assert later_call is None
    assert controller.ended is True
    assert controller.end_signal is first
    assert observed == [first]


def test_physical_stop_uses_same_idempotent_local_end_path() -> None:
    observed: list[SessionEndSignal] = []
    controller = SessionEndController(observed.append)

    first = controller.request_physical_stop()
    duplicate_button_press = controller.request_physical_stop()
    later_tool_call = controller.accept_validated_tool_call(_validated_call())

    assert first is not None
    assert first.source is SessionEndSource.PHYSICAL_STOP
    assert first.tool_call_id is None
    assert first.abort_reason is None
    assert duplicate_button_press is None
    assert later_tool_call is None
    assert controller.end_signal is first
    assert observed == [first]


def test_runtime_abort_is_distinct_and_does_not_claim_physical_stop() -> None:
    controller = SessionEndController()

    signal = controller.request_runtime_abort(RuntimeAbortReason.SERVICE_SHUTDOWN)

    assert signal is not None
    assert signal.source is SessionEndSource.RUNTIME_ABORT
    assert signal.abort_reason is RuntimeAbortReason.SERVICE_SHUTDOWN
    assert signal.tool_call_id is None
    assert controller.request_physical_stop() is None


def test_end_signal_authority_is_bound_to_the_exact_controller() -> None:
    issuing_controller = SessionEndController()
    foreign_controller = SessionEndController()

    signal = issuing_controller.request_physical_stop()

    assert signal is not None
    assert issuing_controller.issued(signal) is True
    assert foreign_controller.issued(signal) is False
    assert issuing_controller.issued(object()) is False
    with pytest.raises(TypeError, match="issued by this"):
        foreign_controller._end(signal)
    assert foreign_controller.ended is False


def test_all_controller_end_paths_mint_controller_bound_signals() -> None:
    tool_controller = SessionEndController()
    physical_controller = SessionEndController()
    abort_controller = SessionEndController()

    tool_signal = tool_controller.accept_validated_tool_call(_validated_call())
    physical_signal = physical_controller.request_physical_stop()
    abort_signal = abort_controller.request_runtime_abort(RuntimeAbortReason.EXPLICIT_CLOSE)

    assert tool_signal is not None and tool_controller.issued(tool_signal)
    assert physical_signal is not None and physical_controller.issued(physical_signal)
    assert abort_signal is not None and abort_controller.issued(abort_signal)
    assert not tool_controller.issued(physical_signal)
    assert not physical_controller.issued(abort_signal)
    assert not abort_controller.issued(tool_signal)


def test_end_signals_cannot_be_publicly_forged() -> None:
    with pytest.raises(TypeError, match="only be issued"):
        SessionEndSignal(source=SessionEndSource.PHYSICAL_STOP)

    forged = object.__new__(SessionEndSignal)
    object.__setattr__(forged, "source", SessionEndSource.PHYSICAL_STOP)
    object.__setattr__(forged, "tool_call_id", None)
    object.__setattr__(forged, "abort_reason", None)
    object.__setattr__(forged, "_issuer", object())

    assert SessionEndController().issued(forged) is False


def test_callback_failure_does_not_reopen_an_ended_session() -> None:
    def fail_after_signal(_: SessionEndSignal) -> None:
        raise RuntimeError("composition cleanup failed")

    controller = SessionEndController(fail_after_signal)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        controller.accept_validated_tool_call(_validated_call())

    assert controller.ended is True
    assert controller.request_physical_stop() is None
