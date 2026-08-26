"""Tests for the video export helpers and command module imports."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from core import sam3_video_to_yolo
from core.video_io import YOLOExporter


class StubVideoHandler:
    """Minimal video handler used to observe exporter lifecycle calls."""

    video_path = "/external/vessel-trip.mp4"

    def release(self):
        """Provide the exporter-required release operation without side effects."""


def test_video_command_imports_without_missing_modules():
    """Verify the video command imports without relying on removed modules."""
    from core import sam3_video_to_yolo

    assert sam3_video_to_yolo.DEFAULT_MODEL_PATH.name == "sam3.pt"


def test_video_workflow_uses_configured_default_frame_step():
    """The default frame interval should control the selected video frames."""
    args = sam3_video_to_yolo.parse_args(["--video", "source.mp4", "--output-dir", "run"])

    assert args.frame_step == 6
    assert sam3_video_to_yolo.frame_indices_for_video(300 * 12, args.frame_step, None) == list(
        range(0, 300 * 12, 6)
    )


def test_video_precision_uses_quantize_and_cpu_falls_back_to_fp32():
    """Video inference should use the current Ultralytics precision option."""
    args = sam3_video_to_yolo.parse_args([
        "--video",
        "source.mp4",
        "--output-dir",
        "run",
        "--device",
        "cpu",
    ])

    assert args.quantize == 16
    assert sam3_video_to_yolo.resolve_quantize(args.device, args.quantize) == 32


def test_build_predictor_passes_quantize_not_half(monkeypatch, tmp_path):
    """Ultralytics should receive the non-deprecated precision override."""
    received = {}

    class FakePredictor:
        def __init__(self, overrides):
            received.update(overrides)

    ultralytics_models = ModuleType("ultralytics.models")
    ultralytics_sam = ModuleType("ultralytics.models.sam")
    ultralytics_sam.SAM3SemanticPredictor = FakePredictor
    monkeypatch.setitem(sys.modules, "ultralytics.models", ultralytics_models)
    monkeypatch.setitem(sys.modules, "ultralytics.models.sam", ultralytics_sam)

    sam3_video_to_yolo.build_predictor(
        tmp_path / "model.pt",
        SimpleNamespace(conf=0.5, iou=0.5, imgsz=728, device=None, quantize=16),
    )

    assert received["quantize"] == 16
    assert "half" not in received


def test_yolo_exporter_writes_segmentation_dataset(tmp_path):
    """Verify a segmentation annotation writes every expected dataset artifact."""
    exporter = YOLOExporter(str(tmp_path), {0: "fish"})
    annotations = {
        12: {
            "frame_idx": 12,
            "detections": [{"class_id": 0, "mask_polygon": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}],
        }
    }

    exporter.export(StubVideoHandler(), annotations)

    assert (
        tmp_path / "labels" / "frame_000012.txt"
    ).read_text() == "0 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000\n"
    assert (
        json.loads((tmp_path / "annotations_json" / "frame_000012.json").read_text())
        == annotations[12]
    )
    assert "path: ." in (tmp_path / "data.yaml").read_text()


def test_interrupted_video_run_exports_completed_frames(monkeypatch, tmp_path):
    """An interrupt should export completed frames before propagating to the CLI."""
    video_path = tmp_path / "input.mp4"
    model_path = tmp_path / "model.pt"
    video_path.write_bytes(b"video")
    model_path.write_bytes(b"model")
    output_dir = tmp_path / "output"
    exported = {}
    metadata = {}
    run_config = {}

    class StubVideoHandler:
        def __init__(self, video_path):
            self.video_path = video_path
            self.info = SimpleNamespace(
                width=100,
                height=50,
                fps=10.0,
                total_frames=20,
                duration_seconds=2.0,
            )

        def release(self):
            pass

    class StubPredictor:
        def __init__(self):
            self.calls = 0

        def set_image(self, image_path):
            pass

        def __call__(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return [object()]

    class StubExporter:
        def __init__(self, output_dir, labels):
            pass

        def export(self, video_handler, annotations, **kwargs):
            exported.update(annotations)

    args = SimpleNamespace(
        video=video_path,
        model=model_path,
        output_dir="run",
        frame_step=1,
        sample_every_seconds=None,
        max_frames=None,
        classes=["fish"],
        overwrite_frames=False,
    )
    monkeypatch.setattr(sam3_video_to_yolo, "resolve_output_folder", lambda _: output_dir)
    monkeypatch.setattr(sam3_video_to_yolo, "VideoHandler", StubVideoHandler)
    monkeypatch.setattr(sam3_video_to_yolo, "build_predictor", lambda *_: StubPredictor())
    monkeypatch.setattr(sam3_video_to_yolo, "frame_indices_for_video", lambda **_: [0, 1])
    monkeypatch.setattr(
        sam3_video_to_yolo, "ensure_frame_image", lambda **_: tmp_path / "frame.jpg"
    )
    monkeypatch.setattr(
        sam3_video_to_yolo,
        "result_to_annotation",
        lambda _, frame_index, __: {"frame_idx": frame_index, "detections": []},
    )
    monkeypatch.setattr(sam3_video_to_yolo, "YOLOExporter", StubExporter)
    monkeypatch.setattr(
        sam3_video_to_yolo,
        "enrich_metadata_file",
        lambda **kwargs: metadata.update(kwargs) or {},
    )
    monkeypatch.setattr(sam3_video_to_yolo, "log_export_metadata", lambda _: None)
    monkeypatch.setattr(
        sam3_video_to_yolo,
        "write_run_config",
        lambda **kwargs: run_config.update(kwargs),
    )

    with pytest.raises(KeyboardInterrupt):
        sam3_video_to_yolo.run(args)

    assert exported == {0: {"frame_idx": 0, "detections": []}}
    assert metadata["sampled_indices"] == [0]
    assert run_config["sampled_frames"] == 1
