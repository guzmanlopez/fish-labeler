"""UI tests for the main labeling window."""

from ui.canvas import LABEL_COLORS
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
    assert not window.settings_sec.toggle_btn.isChecked()
    assert window.settings_sec.content_area.isHidden()
    assert not window.filters_sec.toggle_btn.isChecked()
    assert window.filters_sec.content_area.isHidden()


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
    assert all(
        max(color.red(), color.green(), color.blue()) >= 191 for color in LABEL_COLORS
    )


def test_has_annotation_icon_skips_suppressed_classes():
    """Person and buoy should not show annotation list icons even if assets exist."""
    assert has_annotation_icon("person") is False
    assert has_annotation_icon("buoy") is False
    assert has_annotation_icon("fish") is True


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
