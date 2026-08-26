"""Tests for the video export helpers and command module imports."""

import json

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
