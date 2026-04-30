import numpy as np

import core.io_manager as io_manager
from core.io_manager import (
    auto_save_labels,
    load_config,
    load_persisted_classes,
    load_tracking_data,
    persist_classes,
    save_config,
    save_tracking_data,
)
from core.state import LabelingState


def test_load_save_config(tmp_path):
    """Test saving and loading config."""
    io_manager.CONFIG_FILE = tmp_path / "sam3_config.json"
    save_config("test_in", "test_out")
    assert io_manager.CONFIG_FILE.exists()
    config = load_config()
    assert config["images_folder"] == "test_in"
    assert config["output_folder"] == "test_out"


def test_persist_classes(tmp_path):
    """Test persisting classes."""
    io_manager.CLASSES_STORE = tmp_path / "sam3_classes.txt"
    persist_classes(["fish", "turtle"])
    classes = load_persisted_classes()
    assert classes == ["fish", "turtle"]


def test_save_and_load_tracking_data(tmp_path):
    """Tracking assignments and summaries should round-trip through JSON."""
    save_tracking_data(
        tmp_path,
        {"frame_0001.jpg": [1, None, 3]},
        {1: {"track_id": 1, "frame_count": 2}, 3: {"track_id": 3, "frame_count": 1}},
        {"iou_gate": 0.4},
    )

    tracking_data = load_tracking_data(tmp_path)

    assert tracking_data["frame_track_ids"] == {"frame_0001.jpg": [1, None, 3]}
    assert tracking_data["tracks"][1]["frame_count"] == 2
    assert tracking_data["config"]["iou_gate"] == 0.4


def test_auto_save_labels_only_writes_annotations_visible_in_filters(tmp_path):
    """Saving should persist only the annotations currently shown by the threshold filters."""
    state = LabelingState()
    state.classes = ["fish"]
    state.class_thresholds = {"fish": 0.9}
    state.output_folder = tmp_path / "out"
    state.current_image = np.zeros((8, 8, 3), dtype=np.uint8)
    state.current_image_path = tmp_path / "frame_0001.jpg"
    state.current_image_path.write_bytes(b"test-image")
    state.current_labels = [
        (
            0,
            [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4],
            [0.1, 0.1, 0.4, 0.1, 0.4, 0.4, 0.1, 0.4],
            None,
            0.95,
        ),
        (
            0,
            [0.5, 0.5, 0.8, 0.5, 0.8, 0.8, 0.5, 0.8],
            [0.5, 0.5, 0.8, 0.5, 0.8, 0.8, 0.5, 0.8],
            None,
            0.5,
        ),
    ]

    message = auto_save_labels(state)
    saved_lines = (state.output_folder / "labels_seg" / "frame_0001.txt").read_text().splitlines()

    assert message == "Saved 1 annotations (Seg)"
    assert len(saved_lines) == 1
