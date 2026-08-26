from __future__ import annotations

import pytest

from recoverybox.realtime.session_control import (
    FINISH_SESSION_TOOL,
    FINISH_SESSION_TOOL_NAME,
    SESSION_CONTROL_TOOL_REGISTRY,
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

    expected = SessionEndSignal(
        source=SessionEndSource.VALIDATED_TOOL_CALL,
        tool_call_id="call-finish-1",
    )
    assert first == expected
    assert duplicate is None
    assert later_call is None
    assert controller.ended is True
    assert controller.end_signal == expected
    assert observed == [expected]


def test_physical_stop_uses_same_idempotent_local_end_path() -> None:
    observed: list[SessionEndSignal] = []
    controller = SessionEndController(observed.append)

    first = controller.request_physical_stop()
    duplicate_button_press = controller.request_physical_stop()
    later_tool_call = controller.accept_validated_tool_call(_validated_call())

    expected = SessionEndSignal(source=SessionEndSource.PHYSICAL_STOP)
    assert first == expected
    assert duplicate_button_press is None
    assert later_tool_call is None
    assert controller.end_signal == expected
    assert observed == [expected]


def test_callback_failure_does_not_reopen_an_ended_session() -> None:
    def fail_after_signal(_: SessionEndSignal) -> None:
        raise RuntimeError("composition cleanup failed")

    controller = SessionEndController(fail_after_signal)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        controller.accept_validated_tool_call(_validated_call())

    assert controller.ended is True
    assert controller.request_physical_stop() is None
