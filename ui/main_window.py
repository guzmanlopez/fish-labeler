"""Main window and sidebar UI for the semi-automatic labeling workflow."""

import traceback
from pathlib import Path

import cv2
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

from core.io_manager import (
    auto_save_labels,
    get_visible_labels,
    load_config,
    load_existing_labels,
    load_persisted_classes,
    load_progress,
    load_tracking_data,
    persist_classes,
    save_config,
    save_progress,
    save_tracking_data,
)
from core.sam_engine import SAMEngine
from core.state import LabelingState
from core.tracker import TrackingConfig, run_offline_tracker
from ui.canvas import LABEL_COLORS, AnnotationCanvas, get_class_icon, icon_asset_exists

# -- helpers -----------------------------------------------------------

ANNOTATION_COLOR_ROLE = int(Qt.ItemDataRole.UserRole) + 1
TRACK_ITEM_ROLE = int(Qt.ItemDataRole.UserRole) + 2
ANNOTATION_ICON_EXCLUSIONS = {"person", "buoy"}


def has_annotation_icon(class_name: str) -> bool:
    """Return whether the annotation list should render an icon for a class."""
    norm_name = class_name.lower().strip().replace(" ", "_")
    if norm_name in ANNOTATION_ICON_EXCLUSIONS:
        return False
    return icon_asset_exists(class_name)


def get_label_track_id(label):
    """Return the track id stored on a label tuple, if any."""
    return label[5] if len(label) > 5 else None


def set_label_track_id(label, track_id):
    """Return a label tuple updated with the provided track id."""
    base = tuple(label[:5])
    return base + (track_id,)


def apply_track_ids_to_labels(labels, track_ids):
    """Attach persisted track ids to a frame's label sequence by index."""
    applied = []
    for index, label in enumerate(labels):
        track_id = track_ids[index] if index < len(track_ids) else None
        applied.append(set_label_track_id(label, track_id))
    return applied


def get_visible_frame_labels(labels, classes, class_thresholds):
    """Return the subset of frame labels currently shown by the threshold filters."""
    return get_visible_labels(labels, classes, class_thresholds)


class AnnotationItemDelegate(QStyledItemDelegate):
    """Paint annotation rows with readable text and per-class color highlights."""

    def paint(self, painter, option, index):
        """Paint a rounded colored row with an optional icon and high-contrast text."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        row_rect = option.rect.adjusted(4, 1, -4, -1)
        bg_value = index.data(ANNOTATION_COLOR_ROLE)
        bg_color = QColor(bg_value) if bg_value is not None else QColor(54, 58, 79)
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        fill_color = QColor(bg_color)
        fill_color.setAlpha(205 if is_selected else 168)
        if is_hovered and not is_selected:
            fill_color = fill_color.lighter(108)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill_color)
        painter.drawRoundedRect(row_rect, 8, 8)

        if is_selected:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(183, 189, 248), 1.5))
            painter.drawRoundedRect(row_rect, 8, 8)

        content_rect = row_rect.adjusted(8, 0, -8, 0)
        icon = index.data(Qt.ItemDataRole.DecorationRole)
        icon_size = 16
        text_left = content_rect.left()
        if isinstance(icon, QIcon) and not icon.isNull():
            icon_top = content_rect.top() + (content_rect.height() - icon_size) // 2
            icon_rect = content_rect.adjusted(0, icon_top - content_rect.top(), 0, 0)
            icon.paint(
                painter,
                icon_rect.left(),
                icon_top,
                icon_size,
                icon_size,
                Qt.AlignmentFlag.AlignCenter,
            )
            text_left += icon_size + 8

        text_rect = content_rect.adjusted(text_left - content_rect.left(), 0, 0, 0)
        text_font = QFont(option.font)
        text_font.setBold(True)
        painter.setFont(text_font)
        painter.setPen(QColor(15, 15, 26))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            index.data(Qt.ItemDataRole.DisplayRole) or "",
        )
        painter.restore()

    def sizeHint(self, option, index):
        """Return a stable row height so colored entries do not overlap."""
        base = super().sizeHint(option, index)
        return QSize(base.width(), max(32, base.height() + 6))


class CollapsibleSection(QWidget):
    """Compact section container used across the left and right sidebars."""

    def __init__(self, title, parent=None, expanded=True):
        """Create a section with a toggle button and a collapsible content area."""
        super().__init__(parent)
        self._title = title
        self.toggle_btn = QPushButton(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setToolTip(f"Show or hide the {title.lower()} section.")
        self.toggle_btn.clicked.connect(self._toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 8, 0, 0)
        self.content_layout.setSpacing(8)

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 10, 0, 4)
        main_lay.setSpacing(2)

        main_lay.addWidget(self.toggle_btn)

        self.line = QFrame()
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        main_lay.addWidget(self.line)

        main_lay.addWidget(self.content_area)
        self.set_expanded(expanded)

    def _toggle(self, checked):
        """Show or hide the section body while keeping the title row visible."""
        self.set_expanded(checked)

    def set_expanded(self, expanded):
        """Update the section arrow and content visibility."""
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setText(("▼ " if expanded else "▶ ") + self._title.upper())
        self.content_area.setVisible(expanded)
        if expanded:
            self.toggle_btn.setStyleSheet(
                "text-align: left; background: transparent; border: none; color: #F5F7FF; font-size: 11px; font-weight: 800; text-transform: uppercase;"
            )
            self.line.setStyleSheet("background-color: #A8B1FF; max-height: 1px; border: none;")
        else:
            self.toggle_btn.setStyleSheet(
                "text-align: left; background: transparent; border: none; color: #8B949E; font-size: 11px; font-weight: 700; text-transform: uppercase;"
            )
            self.line.setStyleSheet("background-color: #363A4F; max-height: 1px; border: none;")

    def addWidget(self, widget, stretch=0):
        self.content_layout.addWidget(widget, stretch)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)


# -- SAM Worker --------------------------------------------------------


class SAMWorker(QThread):
    """Run SAM inference off the UI thread and return either a result or an error."""

    finished = pyqtSignal(object, str)

    def __init__(self, func, *args):
        """Store the callable that will be executed in the worker thread."""
        super().__init__()
        self._func = func
        self._args = args

    def run(self):
        """Execute the SAM task and emit a normalized success or failure payload."""
        try:
            result = self._func(*self._args)
            self.finished.emit(result, "")
        except Exception as e:
            traceback.print_exc()
            self.finished.emit(None, str(e))


# -- MainWindow --------------------------------------------------------


class MainWindow(QMainWindow):
    """Main application window for browsing images and refining annotations."""

    def __init__(self, sam_model_path="sam3.pt"):
        """Build the labeling UI and wire the canvas, sidebars, and shortcuts."""
        super().__init__()
        self.setWindowTitle("Fish Labeler")
        self.setMinimumSize(1400, 900)
        self.state = LabelingState()
        self.state.classes = load_persisted_classes()
        self.sam = SAMEngine(model_path=sam_model_path)
        self._worker = None
        config = load_config()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ======= Navigation bar =======
        nav = QFrame()
        nav.setObjectName("navBar")
        nav_lay = QHBoxLayout(nav)
        nav_lay.setContentsMargins(8, 6, 8, 6)
        # nav_lay.setSpacing(12)
        fit_btn = QPushButton("🖵 Fit image")
        fit_btn.setToolTip("Fit image to window (F)")
        fit_btn.clicked.connect(lambda: self.canvas.fit_view())
        nav_lay.addWidget(fit_btn)

        self.selection_btn = QPushButton("Select object")
        self.selection_btn.setToolTip("Allows to select an object in the canvas (3)")
        self.selection_btn.setCheckable(True)
        self.selection_btn.setProperty("mode", "select")
        nav_lay.addWidget(self.selection_btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        nav_lay.addWidget(spacer)

        # -- Frame selection --
        frame_nav_box = QHBoxLayout()
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("#")
        self.jump_input.setToolTip("Jump directly to a frame number in the loaded sequence.")
        # self.jump_input.setFixedWidth(52)
        self.jump_input.returnPressed.connect(self._jump_to)
        frame_nav_box.addWidget(self.jump_input)

        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet("color:#8892b0; font-size:12px;")
        frame_nav_box.addWidget(self.frame_label)

        jump_btn = QPushButton("Go")
        jump_btn.setToolTip("Jump to the frame number entered in the field.")
        jump_btn.clicked.connect(self._jump_to)
        frame_nav_box.addWidget(jump_btn)

        prev_btn = QPushButton("◀ Prev")
        prev_btn.setToolTip("Load the previous frame in the sequence. Left arrow shortcut.")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("Next ▶")
        next_btn.setToolTip("Load the next frame in the sequence. Right arrow shortcut.")
        next_btn.clicked.connect(self._next_image)
        frame_nav_box.addWidget(prev_btn)
        frame_nav_box.addWidget(next_btn)
        nav_lay.addLayout(frame_nav_box)

        root.addWidget(nav)

        # ======= Body (segmentation tools | canvas | annoatations) =======
        body = QSplitter(Qt.Orientation.Horizontal)

        # -- Segmentation panel --
        segmentation_w = QFrame()
        segmentation_w.setObjectName("segmentationPanel")
        segmentation_w.setMinimumWidth(210)
        sl = QVBoxLayout(segmentation_w)
        sl.setContentsMargins(12, 12, 12, 12)
        sl.setSpacing(8)

        self.sam3_sec = CollapsibleSection("SAM3 Segmentation", expanded=True)

        # -- Text segmentation --
        self.text_seg_sec = CollapsibleSection("Text prompt:")
        self.text_prompt = QLineEdit(self.state.classes[0] if self.state.classes else "")
        self.text_prompt.setPlaceholderText("e.g.: person,fish,glove,buoy,blood,stick")
        self.text_prompt.setToolTip(
            "Enter one or more text prompts for PCS segmentation, separated by commas."
        )
        self.text_prompt.returnPressed.connect(self._segment_text)
        self.text_seg_sec.addWidget(self.text_prompt)
        seg_btn = QPushButton("▶ Run PCS")
        seg_btn.setToolTip(
            "Run Promptable Concept Segmentation (PCS). The SAM3 model will attempt to segment all instances in the image of the visual concepts specified by the provided text prompt."
        )
        seg_btn.setObjectName("primaryBtn")
        seg_btn.clicked.connect(self._segment_text)
        self.text_seg_sec.addWidget(seg_btn)
        self.sam3_sec.addWidget(self.text_seg_sec)

        # -- Visual segmentation modes --
        self.visual_seg_sec = CollapsibleSection("Visual prompt:")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.selection_btn, 2)
        for i, (label, mode, tip) in enumerate(
            [
                ("Point", "point", "Single object instance using a point (1)"),
                ("Box", "box", "Single object instance using a box (2)"),
            ]
        ):
            rb = QRadioButton(label)
            rb.setProperty("mode", mode)
            rb.setToolTip(tip)
            rb.setStyleSheet("font-size: 16px; padding: 6px 2px;")
            if i == 0:
                rb.setChecked(True)
            self.mode_group.addButton(rb, i)
            self.visual_seg_sec.addWidget(rb)
        self.mode_group.buttonClicked.connect(self._mode_changed)

        self.visual_seg_sec.addWidget(QLabel("Class for visual:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.state.classes)
        self.class_combo.setToolTip("Choose which class to assign when using point or box prompts.")
        self.visual_seg_sec.addWidget(self.class_combo)
        class_sel_box = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("New class")
        self.new_class_input.setToolTip("Add a new class name to the project label list.")
        self.new_class_input.returnPressed.connect(self._add_class)
        class_sel_box.addWidget(self.new_class_input)
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(32)
        add_btn.setToolTip("Add the class typed in the field to the available classes.")
        add_btn.clicked.connect(self._add_class)
        class_sel_box.addWidget(add_btn)
        del_cls_btn = QPushButton("−")
        del_cls_btn.setFixedWidth(32)
        del_cls_btn.setToolTip("Delete current class")
        del_cls_btn.clicked.connect(self._delete_class)
        class_sel_box.addWidget(del_cls_btn)
        self.visual_seg_sec.addLayout(class_sel_box)

        self.point_prompt_sec = CollapsibleSection("Point prompts", expanded=True)
        point_prompt_hint = QLabel("Click the canvas to queue points, then run SAM.")
        point_prompt_hint.setStyleSheet("color: #8B949E; font-size: 11px;")
        self.point_prompt_sec.addWidget(point_prompt_hint)

        point_type_row = QHBoxLayout()
        self.point_target_group = QButtonGroup(self)
        self.point_positive_btn = QRadioButton("Positive")
        self.point_positive_btn.setChecked(True)
        self.point_positive_btn.setProperty("pointTarget", "positive")
        self.point_positive_btn.setToolTip(
            "Add a green positive point to indicate the object region to keep."
        )
        self.point_target_group.addButton(self.point_positive_btn)
        point_type_row.addWidget(self.point_positive_btn)
        self.point_negative_btn = QRadioButton("Negative")
        self.point_negative_btn.setProperty("pointTarget", "negative")
        self.point_negative_btn.setToolTip(
            "Add a red negative point to tell SAM which nearby region to avoid."
        )
        self.point_target_group.addButton(self.point_negative_btn)
        point_type_row.addWidget(self.point_negative_btn)
        self.point_prompt_sec.addLayout(point_type_row)
        self.point_target_group.buttonClicked.connect(self._point_prompt_target_changed)

        self.point_prompt_counts_label = QLabel("Positive: 0 | Negative: 0")
        self.point_prompt_counts_label.setToolTip(
            "Shows how many positive and negative point prompts are queued for the next SAM run."
        )
        self.point_prompt_sec.addWidget(self.point_prompt_counts_label)

        self.keep_positive_points_cb = QCheckBox("Keep positive across frames")
        self.keep_positive_points_cb.setChecked(self.state.keep_positive_points_across_frames)
        self.keep_positive_points_cb.setToolTip(
            "Keep queued positive points at the same coordinates when moving to another frame."
        )
        self.keep_positive_points_cb.toggled.connect(self._toggle_keep_positive_points)
        self.point_prompt_sec.addWidget(self.keep_positive_points_cb)

        self.keep_negative_points_cb = QCheckBox("Keep negative across frames")
        self.keep_negative_points_cb.setChecked(self.state.keep_negative_points_across_frames)
        self.keep_negative_points_cb.setToolTip(
            "Keep queued negative points at the same coordinates when moving to another frame."
        )
        self.keep_negative_points_cb.toggled.connect(self._toggle_keep_negative_points)
        self.point_prompt_sec.addWidget(self.keep_negative_points_cb)

        point_action_row = QHBoxLayout()
        self.run_points_btn = QPushButton("Run points")
        self.run_points_btn.setObjectName("primaryBtn")
        self.run_points_btn.setToolTip(
            "Run SAM point-based segmentation using all queued positive and negative clicks."
        )
        self.run_points_btn.clicked.connect(self._run_point_prompts)
        point_action_row.addWidget(self.run_points_btn)
        self.clear_points_btn = QPushButton("Clear points")
        self.clear_points_btn.setToolTip(
            "Remove all queued positive and negative point prompts for the current frame."
        )
        self.clear_points_btn.clicked.connect(self._clear_point_prompts)
        point_action_row.addWidget(self.clear_points_btn)
        self.point_prompt_sec.addLayout(point_action_row)
        self.visual_seg_sec.addWidget(self.point_prompt_sec)
        self.sam3_sec.addWidget(self.visual_seg_sec)

        # -- Settings --
        self.settings_sec = CollapsibleSection("Settings", expanded=False)
        self.settings_sec.addWidget(QLabel("Object visualization:"))
        dm_r = QHBoxLayout()
        dm_r.setSpacing(2)
        self.dm_group = QButtonGroup(self)
        for label, mode in [("Both", "both"), ("Mask", "mask"), ("Outline", "outline")]:
            rb = QRadioButton(label)
            rb.setProperty("dm", mode)
            if mode == "both":
                rb.setChecked(True)
            rb.setToolTip(f"Display annotations using {label.lower()} mode.")
            self.dm_group.addButton(rb)
            dm_r.addWidget(rb)
        self.dm_group.buttonClicked.connect(self._display_mode_changed)
        self.settings_sec.addLayout(dm_r)

        self.aabb_cb = QCheckBox("Plot Axis-Aligned Bounding Boxes")
        self.aabb_cb.setChecked(self.state.display_aabb)
        self.aabb_cb.setToolTip(
            "Draw simple axis-aligned boxes instead of oriented quadrilaterals."
        )

        def _on_aabb(v):
            """Docstring for _on_aabb."""
            self.state.display_aabb = bool(v)
            self.canvas.display_aabb = bool(v)
            self.canvas.update()

        self.aabb_cb.toggled.connect(_on_aabb)
        self.settings_sec.addWidget(self.aabb_cb)

        self.settings_sec.addWidget(QLabel("Mask opacity:"))
        self.mask_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.mask_opacity_slider.setRange(15, 100)
        self.mask_opacity_slider.setValue(int(self.state.mask_opacity * 100))
        self.mask_opacity_slider.setToolTip(
            "Adjust the canvas mask overlay strength for better contrast against bright images."
        )
        self.mask_opacity_label = QLabel(f"{int(self.state.mask_opacity * 100)}%")
        mask_r = QHBoxLayout()
        mask_r.addWidget(self.mask_opacity_slider)
        mask_r.addWidget(self.mask_opacity_label)
        self.mask_opacity_slider.valueChanged.connect(self._mask_opacity_changed)
        self.settings_sec.addLayout(mask_r)

        # Polygon simplification slider
        self.settings_sec.addWidget(QLabel("Polygon simplify:"))
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(1, 20)
        self.epsilon_slider.setValue(5)
        self.epsilon_slider.setToolTip("Lower = more precise (0.001~0.020)")
        self.epsilon_label = QLabel("0.005")
        eps_r = QHBoxLayout()
        eps_r.addWidget(self.epsilon_slider)
        eps_r.addWidget(self.epsilon_label)
        self.epsilon_slider.valueChanged.connect(self._epsilon_changed)
        self.settings_sec.addLayout(eps_r)

        # Overlap threshold slider
        self.settings_sec.addWidget(QLabel("Overlap threshold:"))
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlap_slider.setRange(0, 50)
        self.overlap_slider.setValue(10)
        self.overlap_slider.setToolTip(
            "Limit how much a new mask is allowed to overlap existing annotations before being rejected."
        )
        self.overlap_label = QLabel("10%")
        ov_r = QHBoxLayout()
        ov_r.addWidget(self.overlap_slider)
        ov_r.addWidget(self.overlap_label)
        self.overlap_slider.valueChanged.connect(self._overlap_changed)
        self.settings_sec.addLayout(ov_r)

        self.fallback_cb = QCheckBox("Fallback to box if SAM fails")
        self.fallback_cb.setChecked(False)
        self.fallback_cb.setToolTip(
            "Keep a box-based annotation when segmentation cannot produce a usable mask."
        )
        self.settings_sec.addWidget(self.fallback_cb)

        # Output formats
        self.settings_sec.addWidget(QLabel("Output formats:"))
        self.fmt_seg = QCheckBox("YOLO-Seg (Polygon)")
        self.fmt_seg.setChecked(True)
        self.fmt_mask = QCheckBox("PNG Mask")
        self.fmt_coco = QCheckBox("COCO JSON")
        self.fmt_obb = QCheckBox("OBB (Oriented Bounding Box)")
        self.fmt_seg.setToolTip("Save visible annotations as YOLO segmentation polygons.")
        self.fmt_mask.setToolTip("Save visible annotations as a per-pixel indexed mask image.")
        self.fmt_coco.setToolTip("Save visible annotations in COCO-style JSON format when enabled.")
        self.fmt_obb.setToolTip("Save visible annotations as oriented bounding boxes.")
        for cb in (self.fmt_seg, self.fmt_mask, self.fmt_coco, self.fmt_obb):
            cb.stateChanged.connect(self._fmt_changed)
            self.settings_sec.addWidget(cb)

        self.sam3_sec.addWidget(self.settings_sec)
        sl.addWidget(self.sam3_sec)

        self.tracking_sec = CollapsibleSection("Tracking", expanded=False)
        run_tracking_btn = QPushButton("Run tracking")
        run_tracking_btn.setObjectName("primaryBtn")
        run_tracking_btn.setToolTip(
            "Run offline multi-object tracking across the loaded frame sequence using the current tracking settings."
        )
        run_tracking_btn.clicked.connect(self._run_tracking)
        self.tracking_sec.addWidget(run_tracking_btn)

        self.track_linking_sec = CollapsibleSection("Tracklet Linking", expanded=True)

        self.track_conf_slider, self.track_conf_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "Detection confidence:",
            0,
            100,
            int(self.state.tracking_config["confidence_threshold"] * 100),
            "Ignore very weak detections before building tracklets.",
            "confidence_threshold",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.track_iou_slider, self.track_iou_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "IoU gate:",
            0,
            100,
            int(self.state.tracking_config["iou_gate"] * 100),
            "Minimum box overlap for frame-to-frame matching.",
            "iou_gate",
            100,
            lambda value: f"{value:.2f}",
        )
        self.track_center_slider, self.track_center_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "Max center jump:",
            1,
            100,
            int(self.state.tracking_config["max_center_distance"] * 100),
            "Maximum normalized center displacement between nearby frames.",
            "max_center_distance",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.track_missed_slider, self.track_missed_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "Max missed frames:",
            0,
            10,
            int(self.state.tracking_config["max_missed_frames"]),
            "How long a tracklet can survive without a match before ending.",
            "max_missed_frames",
            1,
            lambda value: f"{int(value)}",
            int,
        )
        self.track_size_slider, self.track_size_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "Size change gate:",
            0,
            200,
            int(self.state.tracking_config["max_size_change"] * 100),
            "Reject links that change box scale too abruptly.",
            "max_size_change",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.track_aspect_slider, self.track_aspect_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "Aspect ratio gate:",
            0,
            100,
            int(self.state.tracking_config["max_aspect_change"] * 100),
            "Reject links with strong box shape changes.",
            "max_aspect_change",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.track_velocity_slider, self.track_velocity_label = self._add_tracking_slider_control(
            self.track_linking_sec,
            "Velocity weight:",
            0,
            100,
            int(self.state.tracking_config["velocity_weight"] * 100),
            "How much constant-velocity prediction influences tracklet matching.",
            "velocity_weight",
            100,
            lambda value: f"{value:.2f}",
        )

        self.tracking_sec.addWidget(self.track_linking_sec)

        self.track_stitching_sec = CollapsibleSection("Offline Stitching", expanded=False)

        self.stitch_gap_slider, self.stitch_gap_label = self._add_tracking_slider_control(
            self.track_stitching_sec,
            "Max stitch gap:",
            0,
            30,
            int(self.state.tracking_config["max_stitch_gap"]),
            "Maximum frame gap allowed when merging tracklets offline.",
            "max_stitch_gap",
            1,
            lambda value: f"{int(value)} fr",
            int,
        )
        self.stitch_center_slider, self.stitch_center_label = self._add_tracking_slider_control(
            self.track_stitching_sec,
            "Stitch center gate:",
            1,
            120,
            int(self.state.tracking_config["stitch_center_distance"] * 100),
            "Maximum predicted center mismatch for offline stitching.",
            "stitch_center_distance",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.stitch_size_slider, self.stitch_size_label = self._add_tracking_slider_control(
            self.track_stitching_sec,
            "Stitch size gate:",
            0,
            200,
            int(self.state.tracking_config["stitch_size_change"] * 100),
            "Maximum relative size drift allowed for stitching tracklets.",
            "stitch_size_change",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.stitch_aspect_slider, self.stitch_aspect_label = self._add_tracking_slider_control(
            self.track_stitching_sec,
            "Stitch aspect gate:",
            0,
            100,
            int(self.state.tracking_config["stitch_aspect_change"] * 100),
            "Maximum aspect-ratio drift allowed during stitching.",
            "stitch_aspect_change",
            100,
            lambda value: f"{int(round(value * 100))}%",
        )
        self.stitch_penalty_slider, self.stitch_penalty_label = self._add_tracking_slider_control(
            self.track_stitching_sec,
            "Gap penalty:",
            0,
            100,
            int(self.state.tracking_config["gap_penalty"] * 100),
            "Penalty applied to longer gaps when selecting stitch candidates.",
            "gap_penalty",
            100,
            lambda value: f"{value:.2f}",
        )

        self.tracking_sec.addWidget(self.track_stitching_sec)

        self.track_manager_sec = CollapsibleSection("Track Manager", expanded=True)

        self.track_manager_sec.addWidget(QLabel("Tracks:"))
        self.track_list = QListWidget()
        self.track_list.setObjectName("trackList")
        self.track_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.track_list.setMinimumHeight(360)
        self.track_list.setMaximumHeight(640)
        self.track_list.setSpacing(2)
        self.track_list.setUniformItemSizes(True)
        self.track_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.track_list.setToolTip(
            "Review tracked identities, select one to highlight its detections, or manage tracks manually."
        )
        self.track_list.setItemDelegate(AnnotationItemDelegate(self.track_list))
        self.track_list.itemSelectionChanged.connect(self._on_track_list_selection)
        self.track_manager_sec.addWidget(self.track_list)

        track_assign_row = QHBoxLayout()
        self.track_id_input = QLineEdit()
        self.track_id_input.setPlaceholderText("Track ID")
        self.track_id_input.setToolTip(
            "Enter an existing or new track ID to assign to the selected annotations in this frame."
        )
        track_assign_row.addWidget(self.track_id_input)
        assign_track_btn = QPushButton("Apply track")
        assign_track_btn.setToolTip(
            "Assign the typed or selected track ID to the currently selected annotations."
        )
        assign_track_btn.clicked.connect(self._apply_track_to_selection)
        track_assign_row.addWidget(assign_track_btn)
        self.track_manager_sec.addLayout(track_assign_row)

        track_actions_row = QHBoxLayout()
        merge_tracks_btn = QPushButton("Merge")
        merge_tracks_btn.setToolTip(
            "Merge the selected tracks into the lowest selected track ID and keep a single identity."
        )
        merge_tracks_btn.clicked.connect(self._merge_selected_tracks)
        track_actions_row.addWidget(merge_tracks_btn)
        delete_track_btn = QPushButton("Delete")
        delete_track_btn.setToolTip(
            "Remove the selected track ID from every frame without deleting the detections themselves."
        )
        delete_track_btn.clicked.connect(self._delete_selected_track)
        clear_tracks_btn = QPushButton("Clear")
        clear_tracks_btn.setToolTip("Remove all track assignments from the loaded sequence.")
        clear_tracks_btn.clicked.connect(self._clear_tracks)
        track_actions_row.addWidget(delete_track_btn)
        track_actions_row.addWidget(clear_tracks_btn)
        self.track_manager_sec.addLayout(track_actions_row)

        self.tracking_sec.addWidget(self.track_manager_sec)

        sl.addWidget(self.tracking_sec)
        sl.addStretch()

        body.addWidget(segmentation_w)

        # -- Canvas --
        self.canvas = AnnotationCanvas()
        self.canvas.set_mask_opacity(self.state.mask_opacity)
        self.canvas.point_clicked.connect(self._on_point_click)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.label_selected.connect(self._on_label_selected)
        self.canvas.set_prompt_points([], [])
        body.addWidget(self.canvas)

        # -- Annotation panel --
        annotation_sidebar_scroll = QScrollArea()
        annotation_sidebar_scroll.setObjectName("rightPanel")
        annotation_sidebar_scroll.setWidgetResizable(True)
        annotation_sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        annotation_sidebar_scroll.setMinimumWidth(250)
        annotation_sidebar = QWidget()
        al = QVBoxLayout(annotation_sidebar)
        al.setContentsMargins(12, 12, 12, 12)
        al.setSpacing(8)

        # -- Folder loading --
        self.folder_sec = CollapsibleSection("Folder loading")
        img_folder_box = QHBoxLayout()
        self.folder_input = QLineEdit(config.get("images_folder", ""))
        self.folder_input.setPlaceholderText("Image folder")
        self.folder_input.setToolTip("Path to the folder containing the image sequence to label.")
        img_folder_box.addWidget(self.folder_input, 4)
        img_browse_btn = QPushButton("🗁")
        img_browse_btn.setFixedWidth(36)
        img_browse_btn.setToolTip("Browse images' folder")
        img_browse_btn.clicked.connect(self._browse_image_folder)
        img_folder_box.addWidget(img_browse_btn)
        self.folder_sec.addLayout(img_folder_box)

        output_folder_box = QHBoxLayout()
        self.output_input = QLineEdit(config.get("output_folder", ""))
        self.output_input.setPlaceholderText("Output folder")
        self.output_input.setToolTip(
            "Folder where visible annotations, masks, and tracking data will be saved."
        )
        output_folder_box.addWidget(self.output_input, 4)
        output_browse_btn = QPushButton("🖿")
        output_browse_btn.setFixedWidth(36)
        output_browse_btn.setToolTip("Browse output folder")
        output_browse_btn.clicked.connect(self._browse_output_folder)
        output_folder_box.addWidget(output_browse_btn)
        self.folder_sec.addLayout(output_folder_box)

        load_btn = QPushButton("Load")
        load_btn.setObjectName("primaryBtn")
        load_btn.setToolTip(
            "Load the image sequence and previously saved labels from the selected folders."
        )
        load_btn.clicked.connect(self._load_folder)
        self.folder_sec.addWidget(load_btn)
        al.addWidget(self.folder_sec)

        # -- Filter options --
        self.filters_sec = CollapsibleSection("Detection Filters", expanded=False)
        self.filters_containerLayout = QVBoxLayout()
        self.filters_sec.addLayout(self.filters_containerLayout)
        self.filters_widget = None
        al.addWidget(self.filters_sec)

        # -- Annotation list --
        self.anno_sec = CollapsibleSection("Annotations")
        self.label_list = QListWidget()
        self.label_list.setObjectName("annotationList")
        self.label_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.label_list.setMinimumHeight(360)
        self.label_list.setMaximumHeight(640)
        self.label_list.setSpacing(2)
        self.label_list.setUniformItemSizes(True)
        self.label_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.label_list.setToolTip(
            "List of annotations currently visible under the active detection filters."
        )
        self.label_list.setItemDelegate(AnnotationItemDelegate(self.label_list))
        self.label_list.itemSelectionChanged.connect(self._on_list_selection)
        self.anno_sec.addWidget(self.label_list)

        # Action row
        op1 = QHBoxLayout()
        op1.setSpacing(4)
        self.change_combo = QComboBox()
        self.change_combo.addItems(self.state.classes)
        self.change_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.change_combo.setToolTip("Choose the replacement class for the selected annotations.")
        op1.addWidget(self.change_combo)
        chg_btn = QPushButton("Apply class")
        chg_btn.setToolTip(
            "Change class of selected annotations to the currently selected class in the dropdown. Useful for correcting wrong classifications or reassigning classes."
        )
        chg_btn.clicked.connect(self._change_selected_class)
        op1.addWidget(chg_btn)
        self.anno_sec.addLayout(op1)

        op2 = QHBoxLayout()
        op2.setSpacing(4)
        del_btn = QPushButton("Delete")
        del_btn.setToolTip("Delete the currently selected annotations from this frame.")
        del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear All")
        clr_btn.setToolTip("Remove all annotations from the current frame.")
        clr_btn.clicked.connect(self._clear_all)
        op2.addWidget(del_btn)
        op2.addWidget(clr_btn)
        self.anno_sec.addLayout(op2)

        # Save
        save_btn = QPushButton("Save")
        save_btn.setObjectName("primaryBtn")
        save_btn.setToolTip(
            "Save only the annotations currently visible under the active filters. Ctrl+S shortcut."
        )
        save_btn.clicked.connect(self._save_labels)
        self.anno_sec.addWidget(save_btn)

        al.addWidget(self.anno_sec)

        al.addStretch()
        annotation_sidebar_scroll.setWidget(annotation_sidebar)
        body.addWidget(annotation_sidebar_scroll)

        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setStretchFactor(2, 0)
        root.addWidget(body, 1)

        # ======= Status bar =======
        self.status_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(120)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        sb = self.statusBar()
        if sb is not None:
            sb.addWidget(self.status_label, 1)
            sb.addPermanentWidget(self.progress_bar)

        # ======= Shortcuts =======
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, self._prev_image)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, self._next_image)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_selected)
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_labels)
        QShortcut(QKeySequence("1"), self, lambda: self._set_mode(0))
        QShortcut(QKeySequence("2"), self, lambda: self._set_mode(1))
        QShortcut(QKeySequence("3"), self, lambda: self._set_mode(2))
        QShortcut(QKeySequence("Ctrl+A"), self, self._select_all)
        QShortcut(QKeySequence("Escape"), self, self._deselect_all)
        QShortcut(QKeySequence("F"), self, self.canvas.fit_view)
        QShortcut(QKeySequence("R"), self, self._segment_text)

        # ======= Style =======
        theme_path = Path("themes/dark.qss")
        if theme_path.exists():
            with open(theme_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())

        self._set_mode(0)
        self._refresh_class_combos()
        self._refresh_track_list()
        self._refresh_point_prompt_ui()

    def _add_tracking_slider_control(
        self,
        section,
        title,
        minimum,
        maximum,
        value,
        tooltip,
        key,
        scale,
        formatter,
        cast=float,
    ):
        """Create a labeled tracking slider bound to a tracker configuration key."""
        section.addWidget(QLabel(title))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setToolTip(tooltip)
        label = QLabel("")
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(label)
        section.addLayout(row)

        def _on_value_changed(raw_value):
            converted = cast(raw_value if scale == 1 else raw_value / scale)
            self.state.tracking_config[key] = converted
            label.setText(formatter(converted))

        slider.valueChanged.connect(_on_value_changed)
        _on_value_changed(value)
        return slider, label

    def _current_frame_key(self):
        """Return the filename key used to persist track ids for the current frame."""
        if self.state.current_image_path is None:
            return None
        return self.state.current_image_path.name

    def _load_tracking_state(self):
        """Load persisted tracking assignments and merge tracker configuration defaults."""
        tracking_data = load_tracking_data(self.state.output_folder)
        self.state.frame_track_ids = tracking_data.get("frame_track_ids", {}) or {}
        self.state.track_summaries = tracking_data.get("tracks", {}) or {}
        loaded_config = tracking_data.get("config", {}) or {}
        self.state.tracking_config = {**self.state.tracking_config, **loaded_config}
        if hasattr(self, "track_conf_slider"):
            self.track_conf_slider.setValue(
                int(self.state.tracking_config["confidence_threshold"] * 100)
            )
            self.track_iou_slider.setValue(int(self.state.tracking_config["iou_gate"] * 100))
            self.track_center_slider.setValue(
                int(self.state.tracking_config["max_center_distance"] * 100)
            )
            self.track_missed_slider.setValue(int(self.state.tracking_config["max_missed_frames"]))
            self.track_size_slider.setValue(
                int(self.state.tracking_config["max_size_change"] * 100)
            )
            self.track_aspect_slider.setValue(
                int(self.state.tracking_config["max_aspect_change"] * 100)
            )
            self.track_velocity_slider.setValue(
                int(self.state.tracking_config["velocity_weight"] * 100)
            )
            self.stitch_gap_slider.setValue(int(self.state.tracking_config["max_stitch_gap"]))
            self.stitch_center_slider.setValue(
                int(self.state.tracking_config["stitch_center_distance"] * 100)
            )
            self.stitch_size_slider.setValue(
                int(self.state.tracking_config["stitch_size_change"] * 100)
            )
            self.stitch_aspect_slider.setValue(
                int(self.state.tracking_config["stitch_aspect_change"] * 100)
            )
            self.stitch_penalty_slider.setValue(
                int(self.state.tracking_config["gap_penalty"] * 100)
            )

    def _save_tracking_state(self):
        """Persist current track ids, summaries, and tracker parameters to disk."""
        self._sync_current_frame_track_ids()
        save_tracking_data(
            self.state.output_folder,
            self.state.frame_track_ids,
            self.state.track_summaries,
            self.state.tracking_config,
        )

    def _load_labels_for_frame(self, image_path):
        """Load labels for a frame, reusing in-memory labels for the active frame."""
        if (
            self.state.current_image_path is not None
            and image_path == self.state.current_image_path
        ):
            return [
                tuple(label[:5])
                for label in get_visible_frame_labels(
                    self.state.current_labels,
                    self.state.classes,
                    self.state.class_thresholds,
                )
            ]

        img = cv2.imread(str(image_path))
        if img is None:
            return []
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_output_path = self.state.output_folder / "labels" / f"{image_path.stem}.txt"
        label_seg_output_path = self.state.output_folder / "labels_seg" / f"{image_path.stem}.txt"
        return load_existing_labels(label_output_path, label_seg_output_path, rgb)

    def _load_track_aware_labels_for_frame(self, image_path):
        """Load a frame's labels and attach any persisted track ids by index."""
        labels = self._load_labels_for_frame(image_path)
        track_ids = self.state.frame_track_ids.get(image_path.name, [])
        return apply_track_ids_to_labels(labels, track_ids)

    def _sync_current_frame_track_ids(self):
        """Update persisted track ids from the current frame labels in memory."""
        frame_key = self._current_frame_key()
        if frame_key is None:
            return
        visible_labels = get_visible_frame_labels(
            self.state.current_labels,
            self.state.classes,
            self.state.class_thresholds,
        )
        track_ids = [get_label_track_id(label) for label in visible_labels]
        if any(track_id is not None for track_id in track_ids):
            self.state.frame_track_ids[frame_key] = track_ids
        else:
            self.state.frame_track_ids.pop(frame_key, None)

    def _apply_current_frame_track_ids(self):
        """Attach persisted track ids to the current frame after loading annotations."""
        frame_key = self._current_frame_key()
        if frame_key is None:
            return
        self.state.current_labels = apply_track_ids_to_labels(
            self.state.current_labels,
            self.state.frame_track_ids.get(frame_key, []),
        )

    def _tracking_config_object(self):
        """Build a tracker configuration dataclass from UI-backed state."""
        cfg = self.state.tracking_config
        return TrackingConfig(
            confidence_threshold=float(cfg["confidence_threshold"]),
            iou_gate=float(cfg["iou_gate"]),
            max_center_distance=float(cfg["max_center_distance"]),
            max_missed_frames=int(cfg["max_missed_frames"]),
            max_size_change=float(cfg["max_size_change"]),
            max_aspect_change=float(cfg["max_aspect_change"]),
            velocity_weight=float(cfg["velocity_weight"]),
            max_stitch_gap=int(cfg["max_stitch_gap"]),
            stitch_center_distance=float(cfg["stitch_center_distance"]),
            stitch_size_change=float(cfg["stitch_size_change"]),
            stitch_aspect_change=float(cfg["stitch_aspect_change"]),
            gap_penalty=float(cfg["gap_penalty"]),
        )

    def _collect_tracking_frames(self):
        """Load all frame labels in sequence for offline tracking."""
        return [self._load_labels_for_frame(image_path) for image_path in self.state.image_list]

    def _rebuild_track_summaries(self):
        """Recompute track summaries from persisted frame assignments."""
        summaries = {}
        image_list = self.state.image_list or (
            [self.state.current_image_path] if self.state.current_image_path is not None else []
        )
        for frame_index, image_path in enumerate(image_list):
            if image_path is None:
                continue
            labels = self._load_track_aware_labels_for_frame(image_path)
            for label in labels:
                track_id = get_label_track_id(label)
                if track_id is None:
                    continue
                class_id = label[0]
                score = label[4] if len(label) > 4 and label[4] is not None else 1.0
                summary = summaries.setdefault(
                    track_id,
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "start_frame": frame_index,
                        "end_frame": frame_index,
                        "frame_count": 0,
                        "detection_count": 0,
                        "mean_confidence": 0.0,
                        "max_gap": 0,
                        "_frames": [],
                    },
                )
                summary["class_id"] = class_id
                summary["start_frame"] = min(summary["start_frame"], frame_index)
                summary["end_frame"] = max(summary["end_frame"], frame_index)
                summary["detection_count"] += 1
                summary["mean_confidence"] += score
                summary["_frames"].append(frame_index)

        for track_id, summary in summaries.items():
            frames = sorted(set(summary.pop("_frames", [])))
            summary["frame_count"] = len(frames)
            summary["mean_confidence"] = summary["mean_confidence"] / max(
                1, summary["detection_count"]
            )
            summary["max_gap"] = max(
                (frames[i + 1] - frames[i] - 1 for i in range(len(frames) - 1)),
                default=0,
            )
        self.state.track_summaries = summaries

    def _refresh_track_list(self):
        """Refresh the tracking sidebar list from stored track summaries."""
        if not hasattr(self, "track_list"):
            return
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for track_id in sorted(self.state.track_summaries):
            summary = self.state.track_summaries[track_id]
            class_id = summary.get("class_id", 0)
            class_name = (
                self.state.classes[class_id]
                if class_id < len(self.state.classes)
                else f"c{class_id}"
            )
            color = LABEL_COLORS[class_id % len(LABEL_COLORS)]
            item = QListWidgetItem(
                f"T{track_id} | {class_name} | {summary.get('frame_count', 0)} fr | "
                f"{summary.get('start_frame', 0) + 1}-{summary.get('end_frame', 0) + 1}"
            )
            item.setData(TRACK_ITEM_ROLE, track_id)
            item.setData(ANNOTATION_COLOR_ROLE, QColor(color))
            if has_annotation_icon(class_name):
                item.setIcon(get_class_icon(class_name, color))
            if track_id in self.state.selected_track_ids:
                item.setSelected(True)
            self.track_list.addItem(item)
        self.track_list.blockSignals(False)

    def _next_track_id(self):
        """Return the next available positive track id."""
        if not self.state.track_summaries and not self.state.frame_track_ids:
            return 1
        known_ids = set(self.state.track_summaries)
        for track_ids in self.state.frame_track_ids.values():
            known_ids.update(track_id for track_id in track_ids if track_id is not None)
        return max(known_ids, default=0) + 1

    def _selected_or_typed_track_id(self):
        """Resolve a target track id from the input field or selected track list entry."""
        text = self.track_id_input.text().strip() if hasattr(self, "track_id_input") else ""
        if text:
            try:
                value = int(text)
                if value > 0:
                    return value
            except ValueError:
                return None
        selected_items = self.track_list.selectedItems() if hasattr(self, "track_list") else []
        if selected_items:
            return selected_items[0].data(TRACK_ITEM_ROLE)
        return None

    def _run_tracking(self):
        """Run offline tracking across the loaded frame sequence."""
        if not self.state.image_list:
            self._set_status("Load a frame sequence before running tracking")
            return
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        print(
            f"[tracking] requested run for {len(self.state.image_list)} frames in {self.state.output_folder}",
            flush=True,
        )
        auto_save_labels(self.state)
        self._sync_current_frame_track_ids()
        self._start_busy()
        self._set_status("Running offline tracking across all frames...")

        def _do():
            """Collect frame detections and run the offline tracker."""
            frames = self._collect_tracking_frames()
            return run_offline_tracker(frames, self._tracking_config_object())

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_tracking_done)
        self._worker.start()

    def _on_tracking_done(self, result, err):
        """Store tracking results and refresh the current frame with track ids."""
        self._end_busy()
        if err:
            print(f"[tracking] error: {err}", flush=True)
            self._set_status(f"Tracking error: {err}")
            return
        if result is None:
            print("[tracking] no result returned", flush=True)
            self._set_status("Tracking produced no result")
            return

        frame_track_ids = {}
        for frame_index, track_ids in result.frame_track_ids.items():
            if frame_index < len(self.state.image_list):
                frame_track_ids[self.state.image_list[frame_index].name] = track_ids
        self.state.frame_track_ids = frame_track_ids
        self.state.track_summaries = result.tracks
        self.state.selected_track_ids.clear()
        self._apply_current_frame_track_ids()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._refresh_track_list()
        print(
            f"[tracking] completed successfully with {len(self.state.track_summaries)} tracks",
            flush=True,
        )
        self._set_status(f"Tracking complete: {len(self.state.track_summaries)} tracks")

    def _on_track_list_selection(self):
        """Select detections in the current frame that belong to the chosen track ids."""
        self.state.selected_track_ids = {
            item.data(TRACK_ITEM_ROLE) for item in self.track_list.selectedItems()
        }
        selected_rows = {
            index
            for index, label in enumerate(self.state.current_labels)
            if get_label_track_id(label) in self.state.selected_track_ids
        }
        self.state.selected_labels = selected_rows
        self.canvas.set_selected(selected_rows)
        self.label_list.blockSignals(True)
        self.label_list.clearSelection()
        for row in selected_rows:
            item = self.label_list.item(row)
            if item is not None:
                item.setSelected(True)
        self.label_list.blockSignals(False)
        if len(self.state.selected_track_ids) == 1:
            track_id = next(iter(self.state.selected_track_ids))
            summary = self.state.track_summaries.get(track_id)
            if summary is not None:
                self._set_status(
                    f"Track T{track_id}: {summary.get('frame_count', 0)} frames, "
                    f"{summary.get('detection_count', 0)} detections"
                )
        elif len(self.state.selected_track_ids) > 1:
            self._set_status(
                f"Selected {len(self.state.selected_track_ids)} tracks for compare or merge"
            )

    def _apply_track_to_selection(self):
        """Assign the typed or next track id to selected annotations in the current frame."""
        if not self.state.selected_labels:
            self._set_status("Select annotations before assigning a track")
            return
        track_id = self._selected_or_typed_track_id() or self._next_track_id()
        for row in self.state.selected_labels:
            if row < len(self.state.current_labels):
                self.state.current_labels[row] = set_label_track_id(
                    self.state.current_labels[row], track_id
                )
        self.state.selected_track_ids = {track_id}
        self._sync_current_frame_track_ids()
        self._rebuild_track_summaries()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._refresh_track_list()
        self._set_status(f"Assigned T{track_id} to {len(self.state.selected_labels)} annotations")

    def _delete_selected_track(self):
        """Remove a track id from all frames without deleting the underlying detections."""
        track_id = self._selected_or_typed_track_id()
        if not track_id:
            self._set_status("Select or enter a track id to delete")
            return
        for frame_key, track_ids in list(self.state.frame_track_ids.items()):
            updated = [None if value == track_id else value for value in track_ids]
            if any(value is not None for value in updated):
                self.state.frame_track_ids[frame_key] = updated
            else:
                self.state.frame_track_ids.pop(frame_key, None)
        self.state.current_labels = [
            set_label_track_id(label, None) if get_label_track_id(label) == track_id else label
            for label in self.state.current_labels
        ]
        self.state.selected_track_ids.discard(track_id)
        self._rebuild_track_summaries()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._refresh_track_list()
        self._set_status(f"Deleted track T{track_id}")

    def _merge_selected_tracks(self):
        """Merge the selected track ids into the lowest selected track id."""
        selected_track_ids = sorted(
            track_id for track_id in self.state.selected_track_ids if track_id
        )
        if len(selected_track_ids) < 2:
            self._set_status("Select at least two tracks to merge")
            return

        target_track_id = selected_track_ids[0]
        merged_track_ids = set(selected_track_ids[1:])

        for frame_key, track_ids in list(self.state.frame_track_ids.items()):
            self.state.frame_track_ids[frame_key] = [
                target_track_id if value in merged_track_ids else value for value in track_ids
            ]

        self.state.current_labels = [
            set_label_track_id(
                label,
                target_track_id
                if get_label_track_id(label) in merged_track_ids
                else get_label_track_id(label),
            )
            if get_label_track_id(label) in merged_track_ids
            else label
            for label in self.state.current_labels
        ]
        self.state.selected_track_ids = {target_track_id}
        self.track_id_input.setText(str(target_track_id))
        self._rebuild_track_summaries()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._refresh_track_list()
        self._set_status(f"Merged {len(selected_track_ids)} tracks into T{target_track_id}")

    def _clear_tracks(self):
        """Remove all track ids from the loaded sequence and current frame."""
        self.state.frame_track_ids.clear()
        self.state.track_summaries.clear()
        self.state.selected_track_ids.clear()
        self.state.current_labels = [
            set_label_track_id(label, None) for label in self.state.current_labels
        ]
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._refresh_track_list()
        self._set_status("Cleared all tracks")

    # ==================================================================
    # Folder / image management
    # ==================================================================

    def _browse_image_folder(self):
        """Docstring for _browse_image_folder."""
        d = QFileDialog.getExistingDirectory(self, "Select image folder")
        if d:
            self.folder_input.setText(d)

    def _browse_output_folder(self):
        """Docstring for _browse_output_folder."""
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_input.setText(d)

    def _load_folder(self):
        """Docstring for _load_folder."""
        fp = self.folder_input.text().strip()
        op = self.output_input.text().strip() or "output"
        if not fp:
            self._set_status("Please enter a folder path")
            return
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = sorted(p for p in Path(fp).iterdir() if p.suffix.lower() in exts)
        if not imgs:
            self._set_status("No images found in folder")
            return
        self.state.image_list = imgs
        self.state.output_folder = Path(op)
        save_config(fp, op)
        self._load_tracking_state()
        self._refresh_track_list()
        last = load_progress(fp)
        if last >= len(imgs):
            last = 0
        self.state.current_index = last
        self._load_current_image()

    def _load_current_image(self):
        """Docstring for _load_current_image."""
        if not self.state.image_list:
            return
        img_path = self.state.image_list[self.state.current_index]
        img = cv2.imread(str(img_path))
        if img is None:
            self._set_status(f"Cannot read: {img_path.name}")
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.state.current_image = rgb
        self.state.current_image_path = img_path
        self._apply_point_prompt_persistence()
        self.state.current_labels = []
        self.state.selected_labels.clear()

        label_output_path = self.state.output_folder / "labels" / f"{img_path.stem}.txt"
        label_seg_output_path = self.state.output_folder / "labels_seg" / f"{img_path.stem}.txt"
        self.state.current_labels = load_existing_labels(
            label_output_path, label_seg_output_path, rgb
        )
        self._apply_current_frame_track_ids()

        self.canvas.set_image(rgb)
        self._refresh_labels_ui()
        self._refresh_point_prompt_ui()
        n = len(self.state.image_list)
        ci = self.state.current_index + 1
        nl = len(self.state.current_labels)
        self._set_status(f"Frame {ci} of {n} | {nl} annotations  |  {img_path.name}")
        self.frame_label.setText(f"/ {n}")
        self.jump_input.setText(str(ci))

    def _nav(self, delta):
        """Docstring for _nav."""
        if not self.state.image_list:
            return
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        self._sync_current_frame_track_ids()
        auto_save_labels(self.state)
        self._save_tracking_state()
        ni = self.state.current_index + delta
        if 0 <= ni < len(self.state.image_list):
            self.state.current_index = ni
            save_progress(self.state.image_list[0].parent, ni, self.state.image_list)
            self._load_current_image()

    def _prev_image(self):
        """Docstring for _prev_image."""
        self._nav(-1)

    def _next_image(self):
        """Docstring for _next_image."""
        self._nav(1)

    def _jump_to(self):
        """Docstring for _jump_to."""
        try:
            idx = int(self.jump_input.text()) - 1
        except ValueError:
            return
        if not self.state.image_list or not (0 <= idx < len(self.state.image_list)):
            return
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        self._sync_current_frame_track_ids()
        auto_save_labels(self.state)
        self._save_tracking_state()
        self.state.current_index = idx
        save_progress(self.state.image_list[0].parent, idx, self.state.image_list)
        self._load_current_image()

    # ==================================================================
    # SAM segmentation (background thread)
    # ==================================================================

    def _get_class_id(self):
        """Docstring for _get_class_id."""
        c = self.class_combo.currentText()
        return self.state.classes.index(c) if c in self.state.classes else 0

    def _start_busy(self):
        """Docstring for _start_busy."""
        self.canvas.set_busy(True)
        self.progress_bar.show()

    def _end_busy(self):
        """Docstring for _end_busy."""
        self.canvas.set_busy(False)
        self.progress_bar.hide()

    def _segment_text(self):
        """Docstring for _segment_text."""
        if self.state.current_image is None:
            return
        prompts = [p.strip() for p in self.text_prompt.text().split(",") if p.strip()]
        if not prompts:
            return
        self._start_busy()
        self._set_status("Running text segmentation...")

        def _do():
            """Docstring for _do."""
            return self.sam.segment_text(
                self.state.current_image,
                prompts,
                self.state.classes,
                self.state.current_labels,
                self.state.polygon_epsilon,
                self.state.overlap_threshold,
            )

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_text_seg_done)
        self._worker.start()

    def _on_text_seg_done(self, result, err):
        """Docstring for _on_text_seg_done."""
        self._end_busy()
        if err:
            self._set_status(f"Error: {err}")
            return
        if result is None:
            self._set_status("No results")
            return
        new_labels, added, skipped, new_classes = result
        if new_classes:
            self.state.classes.extend(new_classes)
            persist_classes(self.state.classes)
            self._refresh_class_combos()
        self.state.current_labels.extend(new_labels)
        self._refresh_labels_ui()
        self._set_status(f"Detection complete: added {added}, skipped {skipped}")

    def _on_point_click(self, x, y):
        """Queue a positive or negative point prompt from the canvas."""
        if self.state.current_image is None:
            return
        point = (int(x), int(y))
        if self.state.point_prompt_target == "negative":
            self.state.negative_prompt_points.append(point)
        else:
            self.state.positive_prompt_points.append(point)
        self._refresh_point_prompt_ui()
        self._set_status(
            f"Queued {self.state.point_prompt_target} point at ({x}, {y}) | "
            f"Positive: {len(self.state.positive_prompt_points)} | "
            f"Negative: {len(self.state.negative_prompt_points)}"
        )

    def _run_point_prompts(self):
        """Run SAM segmentation from all queued positive and negative point prompts."""
        if self.state.current_image is None:
            return
        if not self.state.positive_prompt_points:
            self._set_status("Add at least one positive point before running SAM")
            return

        cid = self._get_class_id()
        points = self.state.positive_prompt_points + self.state.negative_prompt_points
        point_labels = [1] * len(self.state.positive_prompt_points) + [0] * len(
            self.state.negative_prompt_points
        )
        self._start_busy()
        self._set_status(
            f"Running SAM with {len(self.state.positive_prompt_points)} positive and "
            f"{len(self.state.negative_prompt_points)} negative point(s)..."
        )

        def _do():
            """Run multi-point SAM inference using the queued canvas prompts."""
            return self.sam.segment_points(
                self.state.current_image,
                points,
                point_labels,
                cid,
                self.state.current_labels,
                self.state.polygon_epsilon,
                self.state.overlap_threshold,
            )

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_point_prompts_done)
        self._worker.start()

    def _on_point_prompts_done(self, result, err):
        """Store point-prompt segmentation results and keep prompts until cleared manually."""
        self._end_busy()
        if err:
            self._set_status(f"Error: {err}")
            return
        if result is None:
            self._set_status("No results")
            return
        label, msg = result
        if label:
            self.state.current_labels.append(label)
            self._refresh_labels_ui()
            self._refresh_point_prompt_ui()
        self._set_status(msg)

    def _on_point_seg_done(self, result, err):
        """Docstring for _on_point_seg_done."""
        self._end_busy()
        if err:
            self._set_status(f"Error: {err}")
            return
        if result is None:
            self._set_status("No results")
            return
        label, msg = result
        if label:
            self.state.current_labels.append(label)
            self._refresh_labels_ui()
        self._set_status(msg)

    def _on_box_drawn(self, x1, y1, x2, y2):
        """Docstring for _on_box_drawn."""
        if self.state.current_image is None:
            return
        cid = self._get_class_id()
        self._start_busy()
        self._set_status("Box segmenting...")

        def _do():
            """Docstring for _do."""
            return self.sam.segment_box(
                self.state.current_image,
                x1,
                y1,
                x2,
                y2,
                cid,
                self.state.current_labels,
                self.state.polygon_epsilon,
                self.state.overlap_threshold,
                self.fallback_cb.isChecked(),
            )

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_point_seg_done)  # same structure
        self._worker.start()

    # ==================================================================
    # UI sync
    # ==================================================================

    def _set_status(self, msg):
        """Docstring for _set_status."""
        self.status_label.setText(msg)

    def _refresh_labels_ui(self):
        """Rebuild the annotation list with per-class colors and optional icons."""
        self.canvas.class_thresholds = self.state.class_thresholds

        self.canvas.set_labels(
            self.state.current_labels, self.state.classes, self.state.selected_labels
        )
        self.label_list.blockSignals(True)
        self.label_list.clear()
        for idx, lb in enumerate(self.state.current_labels):
            cid = lb[0]
            cn = self.state.classes[cid] if cid < len(self.state.classes) else f"c{cid}"
            score = lb[4] if len(lb) > 4 else None
            track_id = get_label_track_id(lb)
            score_str = f" - {score:.2f}" if score is not None else ""
            co = LABEL_COLORS[cid % len(LABEL_COLORS)]
            track_str = f"T{track_id} | " if track_id is not None else ""
            item = QListWidgetItem(f"{track_str}ID {idx + 1} - {cn}{score_str}")
            item.setData(ANNOTATION_COLOR_ROLE, QColor(co))
            if has_annotation_icon(cn):
                item.setIcon(get_class_icon(cn, co))

            thr = self.state.class_thresholds.get(cn, 0.25)
            self.label_list.addItem(item)
            if score is not None and score < thr:
                item.setHidden(True)
            if idx in self.state.selected_labels:
                item.setSelected(True)
        self.label_list.blockSignals(False)
        self._refresh_track_list()
        if self.state.selected_track_ids:
            self._on_track_list_selection()

    def _refresh_class_combos(self):
        """Docstring for _refresh_class_combos."""
        for cb in (self.class_combo, self.change_combo):
            cur = cb.currentText()
            cb.clear()
            cb.addItems(self.state.classes)
            if cur in self.state.classes:
                cb.setCurrentText(cur)
        self._refresh_threshold_sliders()

    def _refresh_threshold_sliders(self):
        """Docstring for _refresh_threshold_sliders."""
        # Recreate the container widget
        if hasattr(self, "filters_widget") and self.filters_widget is not None:
            self.filters_widget.deleteLater()

        self.filters_widget = QWidget()
        self.filters_layout = QVBoxLayout(self.filters_widget)
        self.filters_layout.setContentsMargins(0, 4, 0, 4)
        self.filters_layout.setSpacing(12)
        self.filters_containerLayout.addWidget(self.filters_widget)

        for cn in self.state.classes:
            if cn not in self.state.class_thresholds:
                self.state.class_thresholds[cn] = 0.25
            val = self.state.class_thresholds[cn]

            t_row = QHBoxLayout()
            t_row.setContentsMargins(0, 0, 0, 0)
            t_row.setSpacing(8)

            lbl = QLabel(f"{cn}: {val:.2f}")
            lbl.setFixedWidth(85)
            lbl.setStyleSheet("color: #8B949E; font-size: 11px;")

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 100)
            slider.setValue(int(val * 100))

            def make_on_thresh(classname, label):
                """Docstring for make_on_thresh."""

                def _on_thresh(v):
                    """Docstring for _on_thresh."""
                    new_val = v / 100.0
                    self.state.class_thresholds[classname] = new_val
                    label.setText(f"{classname}: {new_val:.2f}")
                    # Update without redefining the whole list to prevent losing focus if desired
                    self._refresh_labels_ui()

                return _on_thresh

            slider.valueChanged.connect(make_on_thresh(cn, lbl))
            t_row.addWidget(lbl)
            t_row.addWidget(slider)
            self.filters_layout.addLayout(t_row)

    def _on_label_selected(self, idx):
        """Docstring for _on_label_selected."""
        self.state.selected_labels = self.canvas._selected.copy()
        self.label_list.blockSignals(True)
        self.label_list.clearSelection()
        for i in self.state.selected_labels:
            if i < self.label_list.count():
                item = self.label_list.item(i)
                if item is not None:
                    item.setSelected(True)
        self.label_list.blockSignals(False)
        n = len(self.state.selected_labels)
        if n:
            self._set_status(f"{n} annotations selected")

    def _on_list_selection(self):
        """Docstring for _on_list_selection."""
        sel = set()
        for it in self.label_list.selectedItems():
            row = self.label_list.row(it)
            if row != -1:
                sel.add(row)
        self.state.selected_labels = sel
        self.canvas.set_selected(sel)

    def _mode_changed(self, btn):
        """Docstring for _mode_changed."""
        self.canvas.set_mode(btn.property("mode"))
        self._refresh_point_prompt_ui()

    def _set_mode(self, idx):
        """Docstring for _set_mode."""
        btn = self.mode_group.button(idx)
        if btn:
            btn.setChecked(True)
            self.canvas.set_mode(btn.property("mode"))
            self._refresh_point_prompt_ui()

    def _current_visual_mode(self):
        """Return the currently selected visual prompt mode."""
        btn = self.mode_group.checkedButton()
        return btn.property("mode") if btn is not None else "point"

    def _point_prompt_target_changed(self, btn):
        """Store whether new point clicks should be treated as positive or negative."""
        self.state.point_prompt_target = btn.property("pointTarget") or "positive"

    def _toggle_keep_positive_points(self, checked):
        """Store whether positive prompts should survive frame navigation."""
        self.state.keep_positive_points_across_frames = bool(checked)

    def _toggle_keep_negative_points(self, checked):
        """Store whether negative prompts should survive frame navigation."""
        self.state.keep_negative_points_across_frames = bool(checked)

    def _apply_point_prompt_persistence(self):
        """Clear only the prompt types that are not configured to persist."""
        if not self.state.keep_positive_points_across_frames:
            self.state.positive_prompt_points.clear()
        if not self.state.keep_negative_points_across_frames:
            self.state.negative_prompt_points.clear()

    def _clear_point_prompts(self, announce=True):
        """Clear queued positive and negative point prompts for the current frame."""
        self.state.positive_prompt_points.clear()
        self.state.negative_prompt_points.clear()
        self._refresh_point_prompt_ui()
        if announce:
            self._set_status("Cleared queued point prompts")

    def _refresh_point_prompt_ui(self):
        """Update point-prompt controls and the canvas markers based on current state."""
        if not hasattr(self, "point_prompt_sec"):
            return

        if self.state.point_prompt_target == "negative":
            self.point_negative_btn.setChecked(True)
        else:
            self.point_positive_btn.setChecked(True)

        self.keep_positive_points_cb.blockSignals(True)
        self.keep_negative_points_cb.blockSignals(True)
        self.keep_positive_points_cb.setChecked(self.state.keep_positive_points_across_frames)
        self.keep_negative_points_cb.setChecked(self.state.keep_negative_points_across_frames)
        self.keep_positive_points_cb.blockSignals(False)
        self.keep_negative_points_cb.blockSignals(False)

        positive_count = len(self.state.positive_prompt_points)
        negative_count = len(self.state.negative_prompt_points)
        self.point_prompt_counts_label.setText(
            f"Positive: {positive_count} | Negative: {negative_count}"
        )

        is_point_mode = self._current_visual_mode() == "point"
        self.point_prompt_sec.setVisible(is_point_mode)
        if hasattr(self, "canvas"):
            if is_point_mode:
                self.canvas.set_prompt_points(
                    self.state.positive_prompt_points,
                    self.state.negative_prompt_points,
                )
            else:
                self.canvas.set_prompt_points([], [])

    def _display_mode_changed(self, btn):
        """Docstring for _display_mode_changed."""
        dm = btn.property("dm")
        self.state.display_mode = dm
        self.canvas.set_display_mode(dm)

    def _epsilon_changed(self, val):
        """Docstring for _epsilon_changed."""
        v = val / 1000.0
        self.state.polygon_epsilon = v
        self.epsilon_label.setText(f"{v:.3f}")

    def _overlap_changed(self, val):
        """Docstring for _overlap_changed."""
        v = val / 100.0
        self.state.overlap_threshold = v
        self.overlap_label.setText("Off" if val == 0 else f"{val}%")

    def _mask_opacity_changed(self, val):
        """Update the canvas mask opacity used to render overlay fills."""
        opacity = val / 100.0
        self.state.mask_opacity = opacity
        self.mask_opacity_label.setText(f"{val}%")
        self.canvas.set_mask_opacity(opacity)

    def _fmt_changed(self):
        """Docstring for _fmt_changed."""
        self.state.output_formats["obb"] = self.fmt_obb.isChecked()
        self.state.output_formats["seg"] = self.fmt_seg.isChecked()
        self.state.output_formats["mask"] = self.fmt_mask.isChecked()
        self.state.output_formats["coco"] = self.fmt_coco.isChecked()

    # ==================================================================
    # Annotation operations
    # ==================================================================

    def _delete_selected(self):
        """Docstring for _delete_selected."""
        if not self.state.selected_labels:
            return
        for i in sorted(self.state.selected_labels, reverse=True):
            if i < len(self.state.current_labels):
                del self.state.current_labels[i]
        self.state.selected_labels.clear()
        self._sync_current_frame_track_ids()
        self._rebuild_track_summaries()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._set_status("Deleted selected annotations")

    def _clear_all(self):
        """Docstring for _clear_all."""
        self.state.current_labels.clear()
        self.state.selected_labels.clear()
        self._sync_current_frame_track_ids()
        self._rebuild_track_summaries()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._set_status("Cleared all annotations")

    def _select_all(self):
        """Docstring for _select_all."""
        self.state.selected_labels = set(range(len(self.state.current_labels)))
        self.canvas.set_selected(self.state.selected_labels)
        self.label_list.selectAll()

    def _deselect_all(self):
        """Docstring for _deselect_all."""
        self.state.selected_labels.clear()
        self.canvas.set_selected(set())
        self.label_list.clearSelection()

    def _change_selected_class(self):
        """Docstring for _change_selected_class."""
        nc = self.change_combo.currentText()
        if not nc or nc not in self.state.classes:
            return
        nid = self.state.classes.index(nc)
        for i in self.state.selected_labels:
            if i < len(self.state.current_labels):
                lb = self.state.current_labels[i]
                self.state.current_labels[i] = (nid,) + lb[1:]
        self._sync_current_frame_track_ids()
        self._rebuild_track_summaries()
        self._save_tracking_state()
        self._refresh_labels_ui()
        self._set_status(f"Changed to {nc}")

    def _add_class(self):
        """Docstring for _add_class."""
        n = self.new_class_input.text().strip()
        if not n or n in self.state.classes:
            return
        self.state.classes.append(n)
        persist_classes(self.state.classes)
        self._refresh_class_combos()
        self.new_class_input.clear()
        self._set_status(f"Added class: {n}")

    def _delete_class(self):
        """Docstring for _delete_class."""
        c = self.class_combo.currentText()
        if not c or len(self.state.classes) <= 1:
            return
        cid = self.state.classes.index(c)
        using = sum(1 for lb in self.state.current_labels if lb[0] == cid)
        if using:
            self._set_status(f"{using} annotations use this class, cannot delete")
            return
        self.state.classes.remove(c)
        persist_classes(self.state.classes)
        self._refresh_class_combos()
        self._set_status(f"Deleted class: {c}")

    def _save_labels(self):
        """Docstring for _save_labels."""
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        self._sync_current_frame_track_ids()
        msg = auto_save_labels(self.state)
        self._save_tracking_state()
        self._set_status(msg or "Saved successfully")
