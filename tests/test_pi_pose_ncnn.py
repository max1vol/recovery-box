from __future__ import annotations

from pathlib import Path

import pytest

from recoverybox.device.pi_pose import PiPoseConfigurationError, PiPoseRuntimeError
from recoverybox.device.pi_pose_ncnn import (
    NANODET_BOX_NAMES,
    NANODET_SCORE_NAMES,
    NcnnPersonPoseEstimator,
    NcnnPoseConfig,
    PersonBox,
    PoseCrop,
    decode_nanodet_person_boxes,
    decode_rtmpose_simcc,
    person_box_to_pose_crop,
    validate_pinned_file,
)


class _FakeMat:
    def __init__(self, width: int, height: int, values: list[float] | None = None):
        self.w = width
        self.h = height
        self.c = 1
        self.values = values if values is not None else [0.0] * (width * height)
        self.normalization: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def __getitem__(self, index: int) -> float:
        return self.values[index]

    def substract_mean_normalize(self, mean, norm) -> None:
        self.normalization = (tuple(mean), tuple(norm))


def test_config_never_weakens_guardian_half_second_limit() -> None:
    with pytest.raises(PiPoseConfigurationError, match=r"0\.5s"):
        NcnnPoseConfig(maximum_result_age_seconds=0.5001)


def test_config_scrubs_invalid_environment_number() -> None:
    with pytest.raises(PiPoseConfigurationError, match="environment"):
        NcnnPoseConfig.from_environment({"RECOVERYBOX_NCNN_THREADS": "private"})


def test_pinned_validator_rejects_symlink_without_hashing(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"private")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(PiPoseConfigurationError, match="non-symlink"):
        validate_pinned_file(
            link,
            expected_size=7,
            expected_sha256="0" * 64,
            asset_name="fixture",
        )


def test_pose_crop_applies_padding_and_aspect_and_rejects_clipping() -> None:
    crop = person_box_to_pose_crop(
        PersonBox(x1=250, y1=120, x2=390, y2=360, score=0.9),
        frame_width=640,
        frame_height=480,
    )

    assert crop is not None
    assert crop.width / crop.height == pytest.approx(192 / 256, rel=0.01)
    assert (crop.frame_width, crop.frame_height) == (640, 480)

    clipped = person_box_to_pose_crop(
        PersonBox(x1=0, y1=0, x2=200, y2=470, score=0.9),
        frame_width=640,
        frame_height=480,
    )
    assert clipped is None


def test_simcc_decoder_uses_contiguous_joint_rows_and_inverse_crop() -> None:
    x_values = [0.0] * (17 * 384)
    y_values = [0.0] * (17 * 512)
    for joint in range(17):
        x_values[joint * 384 + 192] = 0.8
        y_values[joint * 512 + 256] = 0.7
    crop = PoseCrop(
        x=100,
        y=40,
        width=300,
        height=400,
        frame_width=640,
        frame_height=480,
    )

    points = decode_rtmpose_simcc(
        _FakeMat(384, 17, x_values),
        _FakeMat(512, 17, y_values),
        crop=crop,
        person_score=0.6,
    )

    assert len(points) == 17
    assert points[0].x == pytest.approx(250 / 640)
    assert points[0].y == pytest.approx(240 / 480)
    assert points[0].score == pytest.approx(0.6)


def _nanodet_tensor_set(input_size: int = 64):
    score_tensors = []
    box_tensors = []
    for stride in (8, 16, 32):
        rows = (input_size // stride) ** 2
        score_tensors.append(_FakeMat(80, rows))
        values = [0.0] * (32 * rows)
        for row in range(rows):
            for side in range(4):
                values[row * 32 + side * 8 + 1] = 10.0
        box_tensors.append(_FakeMat(32, rows, values))
    return score_tensors, box_tensors


def test_nanodet_decoder_reads_person_class_only_and_returns_numeric_box() -> None:
    scores, boxes = _nanodet_tensor_set()
    scores[2].values[0] = 0.9
    scores[2].values[1] = 0.99  # bicycle must not count as a person

    decoded = decode_nanodet_person_boxes(
        scores,
        boxes,
        input_width=64,
        input_height=64,
        resized_scale=1.0,
        pad_left=0,
        pad_top=0,
        frame_width=64,
        frame_height=64,
        score_threshold=0.55,
        nms_iou_threshold=0.3,
    )

    assert len(decoded) == 1
    assert decoded[0].score == pytest.approx(0.9)
    assert 0 <= decoded[0].x1 < decoded[0].x2 <= 64
    assert 0 <= decoded[0].y1 < decoded[0].y2 <= 64


def test_nanodet_decoder_rejects_malformed_tensor_shape() -> None:
    scores, boxes = _nanodet_tensor_set()
    scores[0].w = 79

    with pytest.raises(PiPoseRuntimeError, match="shape"):
        decode_nanodet_person_boxes(
            scores,
            boxes,
            input_width=64,
            input_height=64,
            resized_scale=1.0,
            pad_left=0,
            pad_top=0,
            frame_width=64,
            frame_height=64,
            score_threshold=0.55,
            nms_iou_threshold=0.3,
        )


def test_nanodet_decoder_preserves_two_distinct_people_for_ambiguity_gate() -> None:
    scores, boxes = _nanodet_tensor_set()
    scores[2].values[0] = 0.9
    scores[2].values[3 * 80] = 0.8

    decoded = decode_nanodet_person_boxes(
        scores,
        boxes,
        input_width=64,
        input_height=64,
        resized_scale=1.0,
        pad_left=0,
        pad_top=0,
        frame_width=64,
        frame_height=64,
        score_threshold=0.55,
        nms_iou_threshold=0.3,
    )

    assert len(decoded) == 2


class _FakePixelType:
    PIXEL_BGRA2BGR = 1
    PIXEL_BGRA2RGB = 2


class _FakeMatFactory:
    PixelType = _FakePixelType

    @staticmethod
    def from_pixels_resize(frame, pixel_type, width, height, target_width, target_height):
        del frame, pixel_type, width, height
        return _FakeMat(target_width, target_height)

    @staticmethod
    def from_pixels_roi_resize(*args):
        del args
        return _FakeMat(192, 256)


class _FakeBorderType:
    BORDER_CONSTANT = 0


class _FakeOptions:
    num_threads = 0
    use_vulkan_compute = True
    use_fp16_packed = True
    use_fp16_storage = True
    use_fp16_arithmetic = True


class _FakeExtractor:
    def __init__(self, tensors: dict[str, _FakeMat]):
        self.tensors = tensors

    def input(self, name: str, mat: _FakeMat) -> int:
        del name, mat
        return 0

    def extract(self, name: str):
        return 0, self.tensors[name]


class _FakeNet:
    def __init__(self, tensors: dict[str, _FakeMat]):
        self.opt = _FakeOptions()
        self.tensors = tensors
        self.cleared = False

    def load_param(self, path: str) -> int:
        del path
        return 0

    def load_model(self, path: str) -> int:
        del path
        return 0

    def create_extractor(self) -> _FakeExtractor:
        return _FakeExtractor(self.tensors)

    def clear(self) -> None:
        self.cleared = True


class _FakeNcnn:
    __version__ = "1.0.20260526"
    __file__ = "/runtime/ncnn/__init__.py"
    Mat = _FakeMatFactory
    BorderType = _FakeBorderType

    def __init__(self):
        detector_scores, detector_boxes = _nanodet_tensor_set(320)
        tensors = {
            **dict(zip(NANODET_SCORE_NAMES, detector_scores, strict=True)),
            **dict(zip(NANODET_BOX_NAMES, detector_boxes, strict=True)),
        }
        self.networks = [_FakeNet(tensors), _FakeNet({})]

    def Net(self) -> _FakeNet:
        return self.networks.pop(0)

    @staticmethod
    def copy_make_border(mat, top, bottom, left, right, kind, value):
        del kind, value
        return _FakeMat(mat.w + left + right, mat.h + top + bottom)


def test_estimator_missing_person_returns_numeric_status_without_retaining_frame(
    tmp_path: Path,
) -> None:
    fake_runtime = _FakeNcnn()
    config = NcnnPoseConfig(
        runtime_path=tmp_path,
        rtmpose_param_path=tmp_path / "pose.param",
        rtmpose_bin_path=tmp_path / "pose.bin",
        nanodet_param_path=tmp_path / "detector.param",
        nanodet_bin_path=tmp_path / "detector.bin",
    )
    ticks = iter((1.0, 1.1, 1.2))
    estimator = NcnnPersonPoseEstimator(
        config,
        ncnn_loader=lambda path: fake_runtime,  # type: ignore[arg-type]
        model_validator=lambda path, **options: Path(path),
        clock=lambda: next(ticks),
    )
    marker = b"SENSITIVE_FRAME_MARKER_9f31"
    raw = marker + bytes(32 * 32 * 4 - len(marker))

    result = estimator.infer_bgra(
        raw,
        width=32,
        height=32,
        captured_monotonic=0.95,
    )

    assert result.points is None
    assert result.person_score is None
    assert result.detector_ms == pytest.approx(100.0)
    assert result.total_ms == pytest.approx(200.0)
    assert result.evidence_age_ms == pytest.approx(250.0)
    assert not result.timed_out
    assert not hasattr(result, "frame")
    assert marker not in repr(estimator.__dict__).encode()
    estimator.close()


def test_estimator_rejects_wrong_frame_size(tmp_path: Path) -> None:
    fake_runtime = _FakeNcnn()
    config = NcnnPoseConfig(
        runtime_path=tmp_path,
        rtmpose_param_path=tmp_path / "pose.param",
        rtmpose_bin_path=tmp_path / "pose.bin",
        nanodet_param_path=tmp_path / "detector.param",
        nanodet_bin_path=tmp_path / "detector.bin",
    )
    estimator = NcnnPersonPoseEstimator(
        config,
        ncnn_loader=lambda path: fake_runtime,  # type: ignore[arg-type]
        model_validator=lambda path, **options: Path(path),
    )

    with pytest.raises(PiPoseRuntimeError, match="wrong byte size"):
        estimator.infer_bgra(
            b"no",
            width=2,
            height=2,
            captured_monotonic=0.0,
        )


def test_estimator_rejects_evidence_at_exactly_500ms_without_inference(
    tmp_path: Path,
) -> None:
    fake_runtime = _FakeNcnn()
    config = NcnnPoseConfig(
        runtime_path=tmp_path,
        rtmpose_param_path=tmp_path / "pose.param",
        rtmpose_bin_path=tmp_path / "pose.bin",
        nanodet_param_path=tmp_path / "detector.param",
        nanodet_bin_path=tmp_path / "detector.bin",
    )
    estimator = NcnnPersonPoseEstimator(
        config,
        ncnn_loader=lambda path: fake_runtime,  # type: ignore[arg-type]
        model_validator=lambda path, **options: Path(path),
        clock=lambda: 1.0,
    )

    result = estimator.infer_bgra(
        bytes(16),
        width=2,
        height=2,
        captured_monotonic=0.5,
    )

    assert result.timed_out
    assert result.evidence_age_ms == pytest.approx(500.0)
    assert result.points is None
    assert result.detector_ms == 0.0


def test_estimator_rejects_backwards_monotonic_clock(tmp_path: Path) -> None:
    fake_runtime = _FakeNcnn()
    config = NcnnPoseConfig(
        runtime_path=tmp_path,
        rtmpose_param_path=tmp_path / "pose.param",
        rtmpose_bin_path=tmp_path / "pose.bin",
        nanodet_param_path=tmp_path / "detector.param",
        nanodet_bin_path=tmp_path / "detector.bin",
    )
    ticks = iter((1.0, 0.9))
    estimator = NcnnPersonPoseEstimator(
        config,
        ncnn_loader=lambda path: fake_runtime,  # type: ignore[arg-type]
        model_validator=lambda path, **options: Path(path),
        clock=lambda: next(ticks),
    )

    with pytest.raises(PiPoseRuntimeError, match="moved backwards"):
        estimator.infer_bgra(
            bytes(16),
            width=2,
            height=2,
            captured_monotonic=0.8,
        )


def test_estimator_stops_after_detector_when_deadline_is_spent(tmp_path: Path) -> None:
    fake_runtime = _FakeNcnn()
    config = NcnnPoseConfig(
        runtime_path=tmp_path,
        rtmpose_param_path=tmp_path / "pose.param",
        rtmpose_bin_path=tmp_path / "pose.bin",
        nanodet_param_path=tmp_path / "detector.param",
        nanodet_bin_path=tmp_path / "detector.bin",
    )
    ticks = iter((1.0, 1.5))
    estimator = NcnnPersonPoseEstimator(
        config,
        ncnn_loader=lambda path: fake_runtime,  # type: ignore[arg-type]
        model_validator=lambda path, **options: Path(path),
        clock=lambda: next(ticks),
    )

    result = estimator.infer_bgra(
        bytes(16),
        width=2,
        height=2,
        captured_monotonic=1.0,
    )

    assert result.timed_out
    assert result.detector_ms == pytest.approx(500.0)
    assert result.total_ms == pytest.approx(500.0)
    assert result.pose_ms is None
