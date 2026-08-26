from recoverybox.demo import demo_as_dicts, run_safety_demo


def test_demo_exercises_safe_happy_uncertain_and_pain_paths() -> None:
    events = run_safety_demo()

    assert [(event.step, event.outcome) for event in events] == [
        ("assessable_movement", "cue"),
        ("camera_uncertainty", "pause"),
        ("pain_report", "stop"),
        ("sanitized_feature_snapshot", "ready_for_local_flower_client"),
    ]
    assert events[0].evidence["approved_cue_id"] == "move_slowly"
    assert events[1].evidence["arbitrary_model_audio_allowed"] is False
    assert events[2].evidence["model_was_allowed_to_lower_caution"] is False
    assert events[3].evidence["contains_raw_media"] is False
    assert events[3].evidence["schema_version"] == "rehab-quality-features/v1"
    assert events[3].evidence["source_pose_schema_version"] == "recoverybox.pose.v1"


def test_demo_output_is_json_compatible() -> None:
    payload = demo_as_dicts()

    assert payload[0]["evidence"]["guardian_rule_version"] == "guardian-v1"
