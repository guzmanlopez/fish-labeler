"""Main window and sidebar UI for the semi-automatic labeling workflow."""

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
    load_config,
    load_existing_labels,
    load_persisted_classes,
    load_progress,
    persist_classes,
    save_config,
    save_progress,
)
from core.sam_engine import SAMEngine
from core.state import LabelingState
from ui.canvas import LABEL_COLORS, AnnotationCanvas, get_class_icon, icon_asset_exists

# -- helpers -----------------------------------------------------------

ANNOTATION_COLOR_ROLE = int(Qt.ItemDataRole.UserRole) + 1
ANNOTATION_ICON_EXCLUSIONS = {"person", "buoy"}


def has_annotation_icon(class_name: str) -> bool:
    """Return whether the annotation list should render an icon for a class."""
    norm_name = class_name.lower().strip().replace(" ", "_")
    if norm_name in ANNOTATION_ICON_EXCLUSIONS:
        return False
    return icon_asset_exists(class_name)


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
        self.toggle_btn.clicked.connect(self._toggle)

        # Style inline to override global button styles simply
        self.toggle_btn.setStyleSheet(
            "text-align: left; background: transparent; border: none; font-size: 11px; font-weight: 700; text-transform: uppercase;"
        )

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 8, 0, 0)
        self.content_layout.setSpacing(8)

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 10, 0, 4)
        main_lay.setSpacing(2)

        main_lay.addWidget(self.toggle_btn)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #363A4F; max-height: 1px; border: none;")
        main_lay.addWidget(line)

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
        # self.jump_input.setFixedWidth(52)
        self.jump_input.returnPressed.connect(self._jump_to)
        frame_nav_box.addWidget(self.jump_input)

        self.frame_label = QLabel("")
        self.frame_label.setStyleSheet("color:#8892b0; font-size:12px;")
        frame_nav_box.addWidget(self.frame_label)

        jump_btn = QPushButton("Go")
        jump_btn.clicked.connect(self._jump_to)
        frame_nav_box.addWidget(jump_btn)

        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self._prev_image)
        next_btn = QPushButton("Next ▶")
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
        self.text_prompt = QLineEdit(
            self.state.classes[0] if self.state.classes else ""
        )
        self.text_prompt.setPlaceholderText("e.g.: person,fish,glove,buoy,blood,stick")
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
        self.visual_seg_sec.addWidget(self.class_combo)
        class_sel_box = QHBoxLayout()
        self.new_class_input = QLineEdit()
        self.new_class_input.setPlaceholderText("New class")
        self.new_class_input.returnPressed.connect(self._add_class)
        class_sel_box.addWidget(self.new_class_input)
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(32)
        add_btn.clicked.connect(self._add_class)
        class_sel_box.addWidget(add_btn)
        del_cls_btn = QPushButton("−")
        del_cls_btn.setFixedWidth(32)
        del_cls_btn.setToolTip("Delete current class")
        del_cls_btn.clicked.connect(self._delete_class)
        class_sel_box.addWidget(del_cls_btn)
        self.visual_seg_sec.addLayout(class_sel_box)
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
            self.dm_group.addButton(rb)
            dm_r.addWidget(rb)
        self.dm_group.buttonClicked.connect(self._display_mode_changed)
        self.settings_sec.addLayout(dm_r)

        self.aabb_cb = QCheckBox("Plot Axis-Aligned Bounding Boxes")
        self.aabb_cb.setChecked(self.state.display_aabb)

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
        self.overlap_label = QLabel("10%")
        ov_r = QHBoxLayout()
        ov_r.addWidget(self.overlap_slider)
        ov_r.addWidget(self.overlap_label)
        self.overlap_slider.valueChanged.connect(self._overlap_changed)
        self.settings_sec.addLayout(ov_r)

        self.fallback_cb = QCheckBox("Fallback to box if SAM fails")
        self.fallback_cb.setChecked(False)
        self.settings_sec.addWidget(self.fallback_cb)

        # Output formats
        self.settings_sec.addWidget(QLabel("Output formats:"))
        self.fmt_seg = QCheckBox("YOLO-Seg (Polygon)")
        self.fmt_seg.setChecked(True)
        self.fmt_mask = QCheckBox("PNG Mask")
        self.fmt_coco = QCheckBox("COCO JSON")
        self.fmt_obb = QCheckBox("OBB (Oriented Bounding Box)")
        for cb in (self.fmt_seg, self.fmt_mask, self.fmt_coco, self.fmt_obb):
            cb.stateChanged.connect(self._fmt_changed)
            self.settings_sec.addWidget(cb)

        self.sam3_sec.addWidget(self.settings_sec)
        sl.addWidget(self.sam3_sec)
        sl.addStretch()

        body.addWidget(segmentation_w)

        # -- Canvas --
        self.canvas = AnnotationCanvas()
        self.canvas.set_mask_opacity(self.state.mask_opacity)
        self.canvas.point_clicked.connect(self._on_point_click)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.label_selected.connect(self._on_label_selected)
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
        output_folder_box.addWidget(self.output_input, 4)
        output_browse_btn = QPushButton("🖿")
        output_browse_btn.setFixedWidth(36)
        output_browse_btn.setToolTip("Browse output folder")
        output_browse_btn.clicked.connect(self._browse_output_folder)
        output_folder_box.addWidget(output_browse_btn)
        self.folder_sec.addLayout(output_folder_box)

        load_btn = QPushButton("Load")
        load_btn.setObjectName("primaryBtn")
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
        self.label_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.label_list.setMinimumHeight(360)
        self.label_list.setMaximumHeight(640)
        self.label_list.setSpacing(2)
        self.label_list.setUniformItemSizes(True)
        self.label_list.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.label_list.setItemDelegate(AnnotationItemDelegate(self.label_list))
        self.label_list.itemSelectionChanged.connect(self._on_list_selection)
        self.anno_sec.addWidget(self.label_list)

        # Action row
        op1 = QHBoxLayout()
        op1.setSpacing(4)
        self.change_combo = QComboBox()
        self.change_combo.addItems(self.state.classes)
        self.change_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
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
        del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear All")
        clr_btn.clicked.connect(self._clear_all)
        op2.addWidget(del_btn)
        op2.addWidget(clr_btn)
        self.anno_sec.addLayout(op2)

        # Save
        save_btn = QPushButton("Save")
        save_btn.setObjectName("primaryBtn")
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

        self._refresh_class_combos()

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
        self.state.current_labels = []
        self.state.selected_labels.clear()

        label_output_path = self.state.output_folder / "labels" / f"{img_path.stem}.txt"
        label_seg_output_path = (
            self.state.output_folder / "labels_seg" / f"{img_path.stem}.txt"
        )
        self.state.current_labels = load_existing_labels(
            label_output_path, label_seg_output_path, rgb
        )

        self.canvas.set_image(rgb)
        self._refresh_labels_ui()
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
        auto_save_labels(self.state)
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
        auto_save_labels(self.state)
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
        """Docstring for _on_point_click."""
        if self.state.current_image is None:
            return
        cid = self._get_class_id()
        self._start_busy()
        self._set_status(f"Segmenting ({x},{y})...")

        def _do():
            """Docstring for _do."""
            return self.sam.segment_point(
                self.state.current_image,
                x,
                y,
                cid,
                self.state.current_labels,
                self.state.polygon_epsilon,
                self.state.overlap_threshold,
            )

        self._worker = SAMWorker(_do)
        self._worker.finished.connect(self._on_point_seg_done)
        self._worker.start()

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
            score_str = f" - {score:.2f}" if score is not None else ""
            co = LABEL_COLORS[cid % len(LABEL_COLORS)]
            item = QListWidgetItem(f"ID {idx + 1} - {cn}{score_str}")
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

    def _set_mode(self, idx):
        """Docstring for _set_mode."""
        btn = self.mode_group.button(idx)
        if btn:
            btn.setChecked(True)
            self.canvas.set_mode(btn.property("mode"))

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
        self._refresh_labels_ui()
        self._set_status("Deleted selected annotations")

    def _clear_all(self):
        """Docstring for _clear_all."""
        self.state.current_labels.clear()
        self.state.selected_labels.clear()
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
        msg = auto_save_labels(self.state)
        self._set_status(msg or "Saved successfully")
