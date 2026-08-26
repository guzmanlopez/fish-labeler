"""UI tests for the main labeling window."""

from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import QAbstractItemView, QPushButton

from ui.canvas import LABEL_COLORS, has_canvas_label_icon
from ui.main_window import ANNOTATION_COLOR_ROLE, MainWindow, has_annotation_icon


class DummySamEngine:
    """Minimal SAM engine stub used to keep UI tests fast and deterministic."""

    def __init__(self, model_path="sam3.pt"):
        self.model_path = model_path


def build_window(monkeypatch, qtbot):
    """Create a main window with a lightweight SAM stub for UI assertions."""
    monkeypatch.setattr("ui.main_window.SAMEngine", DummySamEngine)
    monkeypatch.setattr(
        "ui.main_window.load_persisted_classes",
        lambda: ["person", "fish", "buoy"],
    )
    monkeypatch.setattr("ui.main_window.load_config", lambda: {})

    window = MainWindow()
    qtbot.addWidget(window)
    return window


def find_button(window, text):
    """Return the first push button with the given visible text."""
    for button in window.findChildren(QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"Button not found: {text}")


def test_mainwindow_init_uses_annotation_delegate(monkeypatch, qtbot):
    """The annotation list should use the custom delegate built for colored rows."""
    window = build_window(monkeypatch, qtbot)

    assert window.sam3_sec.toggle_btn.isChecked()
    assert window.label_list.objectName() == "annotationList"
    assert window.label_list.itemDelegate() is not None
    assert window.label_list.spacing() == 2
    assert window.label_list.minimumHeight() == 360
    assert window.label_list.maximumHeight() == 640
    assert window.mask_opacity_slider.value() == 62
    assert window.mask_opacity_label.text() == "62%"
    assert window.canvas.mask_opacity == 0.62


def test_mainwindow_starts_with_secondary_sections_collapsed(monkeypatch, qtbot):
    """Settings and detection filters should start collapsed to reduce startup clutter."""
    window = build_window(monkeypatch, qtbot)

    assert not window.sam3_sec.content_area.isHidden()
    assert not window.tracking_sec.toggle_btn.isChecked()
    assert window.tracking_sec.content_area.isHidden()
    assert window.track_linking_sec.toggle_btn.isChecked()
    assert not window.track_linking_sec.content_area.isHidden()
    assert not window.track_stitching_sec.toggle_btn.isChecked()
    assert window.track_stitching_sec.content_area.isHidden()
    assert window.track_manager_sec.toggle_btn.isChecked()
    assert not window.track_manager_sec.content_area.isHidden()
    assert not window.settings_sec.toggle_btn.isChecked()
    assert window.settings_sec.content_area.isHidden()
    assert not window.filters_sec.toggle_btn.isChecked()
    assert window.filters_sec.content_area.isHidden()


def test_expanded_sections_use_higher_contrast_header_text(monkeypatch, qtbot):
    """Expanded section headers should use brighter styling than collapsed ones."""
    window = build_window(monkeypatch, qtbot)

    assert "#F5F7FF" in window.sam3_sec.toggle_btn.styleSheet()
    assert "#8B949E" in window.tracking_sec.toggle_btn.styleSheet()


def test_tracking_section_contains_controls(monkeypatch, qtbot):
    """The tracking section should expose parameters and track management widgets."""
    window = build_window(monkeypatch, qtbot)

    assert window.track_linking_sec.parent() is window.tracking_sec.content_area
    assert window.track_stitching_sec.parent() is window.tracking_sec.content_area
    assert window.track_manager_sec.parent() is window.tracking_sec.content_area
    assert window.track_list.objectName() == "trackList"
    assert window.track_list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    assert window.track_list.itemDelegate() is not None
    assert window.track_conf_slider.value() == 55
    assert window.stitch_gap_slider.value() == 12
    assert window.track_id_input.placeholderText() == "Track ID"
    assert find_button(window, "Merge").toolTip()


def test_buttons_and_tracking_controls_expose_tooltips(monkeypatch, qtbot):
    """Key actions and tracking controls should explain themselves via tooltips."""
    window = build_window(monkeypatch, qtbot)

    assert "frame number" in window.jump_input.toolTip().lower()
    assert "tracking" in find_button(window, "Run tracking").toolTip().lower()
    assert "track id" in window.track_id_input.toolTip().lower()
    assert "selected annotations" in find_button(window, "Apply track").toolTip().lower()
    assert "visible annotations" in find_button(window, "Save").toolTip().lower()
    assert "loaded sequence" in find_button(window, "Clear tracks").toolTip().lower()


def test_sam3_section_collapses_all_segmentation_controls(monkeypatch, qtbot):
    """Collapsing the top-level SAM3 section should hide all segmentation options."""
    window = build_window(monkeypatch, qtbot)

    assert window.text_seg_sec.parent() is window.sam3_sec.content_area
    assert window.visual_seg_sec.parent() is window.sam3_sec.content_area
    assert window.settings_sec.parent() is window.sam3_sec.content_area

    window.sam3_sec.set_expanded(False)

    assert window.sam3_sec.content_area.isHidden()


def test_label_colors_are_bright_and_diverse():
    """The canvas palette should provide enough vivid colors for many classes."""
    assert len(LABEL_COLORS) >= 16
    assert len({color.name() for color in LABEL_COLORS}) == len(LABEL_COLORS)
    assert all(max(color.red(), color.green(), color.blue()) >= 191 for color in LABEL_COLORS)


def test_has_annotation_icon_skips_suppressed_classes():
    """Person and buoy should not show annotation list icons even if assets exist."""
    assert has_annotation_icon("person") is False
    assert has_annotation_icon("buoy") is False
    assert has_annotation_icon("fish") is True


def test_canvas_label_icon_skips_suppressed_classes():
    """Person and buoy should not show on-canvas label icons even when assets exist."""
    assert has_canvas_label_icon("person") is False
    assert has_canvas_label_icon("buoy") is False
    assert has_canvas_label_icon("fish") is True


def test_refresh_labels_ui_sets_colors_and_icons(monkeypatch, qtbot):
    """Refreshing labels should assign row colors and only keep allowed icons."""
    window = build_window(monkeypatch, qtbot)
    window.state.class_thresholds = {"person": 0.25, "fish": 0.25, "buoy": 0.25}
    window.state.current_labels = [
        [0, [], [], None, 0.95],
        [1, [], [], None, 0.90],
        [2, [], [], None, 0.85],
    ]

    window._refresh_labels_ui()

    assert window.label_list.count() == 3
    assert window.label_list.item(0).text() == "ID 1 - person - 0.95"
    assert window.label_list.item(0).data(ANNOTATION_COLOR_ROLE) == LABEL_COLORS[0]
    assert window.label_list.item(0).icon().isNull()
    assert not window.label_list.item(1).icon().isNull()
    assert window.label_list.item(2).icon().isNull()


def test_refresh_labels_ui_hides_items_below_threshold(monkeypatch, qtbot):
    """Rows under the class threshold should stay in the list but remain hidden."""
    window = build_window(monkeypatch, qtbot)
    window.state.class_thresholds = {"person": 0.25, "fish": 0.95, "buoy": 0.25}
    window.state.current_labels = [
        [1, [], [], None, 0.50],
        [0, [], [], None, 0.99],
    ]

    window._refresh_labels_ui()

    assert window.label_list.count() == 2
    assert window.label_list.item(0).isHidden()
    assert not window.label_list.item(1).isHidden()


def test_mask_opacity_slider_updates_canvas(monkeypatch, qtbot):
    """Changing the mask opacity slider should immediately update canvas rendering state."""
    window = build_window(monkeypatch, qtbot)

    window.mask_opacity_slider.setValue(85)

    assert window.state.mask_opacity == 0.85
    assert window.canvas.mask_opacity == 0.85
    assert window.mask_opacity_label.text() == "85%"


def test_point_prompt_controls_follow_visual_mode(monkeypatch, qtbot):
    """Point prompt controls should be visible only in point mode."""
    window = build_window(monkeypatch, qtbot)

    assert not window.point_prompt_sec.isHidden()

    window._set_mode(1)

    assert window.point_prompt_sec.isHidden()

    window._set_mode(0)

    assert not window.point_prompt_sec.isHidden()


def test_point_clicks_queue_positive_and_negative_prompts(monkeypatch, qtbot):
    """Canvas point clicks should be queued until the user runs point segmentation."""
    window = build_window(monkeypatch, qtbot)
    window.state.current_image = "dummy-image"

    window._on_point_click(10, 20)
    window.point_negative_btn.click()
    window._on_point_click(30, 40)

    assert window.state.positive_prompt_points == [(10, 20)]
    assert window.state.negative_prompt_points == [(30, 40)]
    assert window.point_prompt_counts_label.text() == "Positive: 1 | Negative: 1"
    assert window.canvas._positive_prompt_points == [(10, 20)]
    assert window.canvas._negative_prompt_points == [(30, 40)]


def test_point_prompt_persistence_across_frames_is_configurable(monkeypatch, qtbot):
    """Point prompts should persist across frame changes only for the enabled prompt types."""
    window = build_window(monkeypatch, qtbot)
    window.state.positive_prompt_points = [(10, 20)]
    window.state.negative_prompt_points = [(30, 40)]

    window.keep_positive_points_cb.setChecked(True)
    window.keep_negative_points_cb.setChecked(False)
    window._apply_point_prompt_persistence()

    assert window.state.positive_prompt_points == [(10, 20)]
    assert window.state.negative_prompt_points == []
    assert window.canvas._positive_prompt_points == [(10, 20)]
    assert window.canvas._negative_prompt_points == []

    window.keep_negative_points_cb.setChecked(True)
    window.state.negative_prompt_points = [(50, 60)]
    window._apply_point_prompt_persistence()

    assert window.state.positive_prompt_points == [(10, 20)]
    assert window.state.negative_prompt_points == [(50, 60)]


def test_load_current_image_replots_kept_point_prompts(monkeypatch, qtbot):
    """Loading another frame should re-apply kept point prompts onto the canvas."""
    window = build_window(monkeypatch, qtbot)
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    monkeypatch.setattr("ui.main_window.cv2.imread", lambda path: image.copy())
    monkeypatch.setattr("ui.main_window.load_existing_labels", lambda *args: [])

    window.state.output_folder = Path("output")
    window.state.image_list = [Path("frame_0001.jpg"), Path("frame_0002.jpg")]
    window.state.current_index = 1
    window.state.positive_prompt_points = [(10, 20)]
    window.state.negative_prompt_points = [(30, 15)]
    window.keep_positive_points_cb.setChecked(True)
    window.keep_negative_points_cb.setChecked(True)

    window._load_current_image()

    assert window.state.current_image_path == Path("frame_0002.jpg")
    assert window.canvas._positive_prompt_points == [(10, 20)]
    assert window.canvas._negative_prompt_points == [(30, 15)]
    assert window.point_prompt_counts_label.text() == "Positive: 1 | Negative: 1"


def test_successful_point_inference_keeps_prompt_markers_visible(monkeypatch, qtbot):
    """Successful SAM point inference should not clear queued prompts automatically."""
    window = build_window(monkeypatch, qtbot)
    window.state.positive_prompt_points = [(10, 20)]
    window.state.negative_prompt_points = [(30, 15)]
    window._refresh_point_prompt_ui()

    label = (1, [], [], None, 1.0)
    window._on_point_prompts_done((label, "SAM detected object"), "")

    assert window.state.current_labels[-1] == label
    assert window.state.positive_prompt_points == [(10, 20)]
    assert window.state.negative_prompt_points == [(30, 15)]
    assert window.canvas._positive_prompt_points == [(10, 20)]
    assert window.canvas._negative_prompt_points == [(30, 15)]
    assert window.point_prompt_counts_label.text() == "Positive: 1 | Negative: 1"


def test_sync_current_frame_track_ids_uses_only_visible_labels(monkeypatch, qtbot):
    """Track-id persistence should follow the same filtered subset used for saving."""
    window = build_window(monkeypatch, qtbot)
    window.state.current_image_path = Path("frame_0001.jpg")
    window.state.class_thresholds = {"person": 0.25, "fish": 0.95, "buoy": 0.25}
    window.state.current_labels = [
        (1, [], [], None, 0.5, 7),
        (0, [], [], None, 0.99, 3),
    ]

    window._sync_current_frame_track_ids()

    assert window.state.frame_track_ids == {"frame_0001.jpg": [3]}


def test_refresh_track_list_uses_class_colors(monkeypatch, qtbot):
    """Track rows should reuse the same label colors shown in the canvas and annotation list."""
    window = build_window(monkeypatch, qtbot)
    window.state.track_summaries = {
        4: {
            "track_id": 4,
            "class_id": 1,
            "frame_count": 3,
            "start_frame": 0,
            "end_frame": 2,
        }
    }

    window._refresh_track_list()

    assert window.track_list.count() == 1
    assert window.track_list.item(0).data(ANNOTATION_COLOR_ROLE) == LABEL_COLORS[1]


def test_merge_selected_tracks_rewrites_ids_and_summaries(monkeypatch, qtbot):
    """Merging selected tracks should keep the lowest id and remap all selected tracks to it."""
    window = build_window(monkeypatch, qtbot)
    window.state.current_image_path = Path("frame_0001.jpg")
    window.state.current_labels = [
        (1, [], [], None, 0.92, 7),
        (1, [], [], None, 0.91, 2),
        (1, [], [], None, 0.89, 7),
    ]
    window.state.frame_track_ids = {
        "frame_0001.jpg": [7, 2, 7],
        "frame_0002.jpg": [2, None, 7],
    }
    window.state.track_summaries = {
        2: {
            "track_id": 2,
            "class_id": 1,
            "frame_count": 2,
            "start_frame": 0,
            "end_frame": 1,
            "detection_count": 2,
        },
        7: {
            "track_id": 7,
            "class_id": 1,
            "frame_count": 2,
            "start_frame": 0,
            "end_frame": 1,
            "detection_count": 3,
        },
    }

    window._refresh_track_list()
    window.track_list.item(0).setSelected(True)
    window.track_list.item(1).setSelected(True)

    window._merge_selected_tracks()

    assert window.state.selected_track_ids == {2}
    assert window.track_id_input.text() == "2"
    assert window.state.frame_track_ids["frame_0001.jpg"] == [2, 2, 2]
    assert window.state.frame_track_ids["frame_0002.jpg"] == [2, None, 2]
    assert all(label[5] == 2 for label in window.state.current_labels)
    assert set(window.state.track_summaries) == {2}
