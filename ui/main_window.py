"""
MainWindow v2 — Full features + optimized UI

Features: output format settings, slider controls, background inference, class delete, number key shortcuts, progress display
"""

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
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
from ui.canvas import LABEL_COLORS, AnnotationCanvas

# -- helpers -----------------------------------------------------------


def color_icon(color: QColor, size=12):
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, 3, 3)
    p.end()
    return QIcon(pm)


def _section_label(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        "color:#8892b0; font-size:12px; font-weight:600; padding:6px 0 2px 0;"
    )
    return lbl


# -- SAM Worker --------------------------------------------------------


class SAMWorker(QThread):
    finished = pyqtSignal(object, str)

    def __init__(self, func, *args):
        super().__init__()
        self._func = func
        self._args = args

    def run(self):
        try:
            result = self._func(*self._args)
            self.finished.emit(result, "")
        except Exception as e:
            self.finished.emit(None, str(e))


# -- MainWindow --------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self, sam_model_path="sam3.pt"):
        super().__init__()
        self.setWindowTitle("SAM3 Labeler")
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

        selection_btn = QPushButton("Select object")
        selection_btn.setToolTip("Allows to select an object in the canvas (3)")
        # selection_btn.clicked.connect(lambda: self.canvas.select_object())
        nav_lay.addWidget(selection_btn)

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
        nav_lay.addLayout(frame_nav_box)

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
        # segmentation_w.setObjectName("segmentationPanel")
        segmentation_w.setMinimumWidth(200)
        sl = QVBoxLayout(segmentation_w)
        sl.setContentsMargins(6, 6, 6, 6)
        sl.setSpacing(4)

        logo = QLabel("SAM3")
        logo.setStyleSheet("font-size:15px; font-weight:800; color:#7c8cf5;")
        sl.addWidget(logo)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        spacer.setFixedHeight(20)
        sl.addWidget(spacer)

        # -- Text segmentation --
        sl.addWidget(_section_label("Text prompt:"))
        self.text_prompt = QLineEdit(
            self.state.classes[0] if self.state.classes else ""
        )
        self.text_prompt.setPlaceholderText("e.g. person,fish,glove,buoy,blood,stick")
        self.text_prompt.returnPressed.connect(self._segment_text)
        sl.addWidget(self.text_prompt)
        seg_btn = QPushButton("▶ Run PCS")
        seg_btn.setToolTip(
            "Run Promptable Concept Segmentation (PCS). The SAM3 model will attempt to segment all instances in the image of the visual concepts specified by the provided text prompt."
        )
        seg_btn.setObjectName("primaryBtn")
        seg_btn.clicked.connect(self._segment_text)
        sl.addWidget(seg_btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        spacer.setFixedHeight(20)
        sl.addWidget(spacer)

        # -- Visual segmentation modes --
        sl.addWidget(_section_label("Visual prompt:"))
        self.mode_group = QButtonGroup(self)
        for i, (label, mode, tip) in enumerate(
            [
                ("Point", "point", "Single object instance using a point (1)"),
                ("Box", "box", "Single object instance using a box (2)"),
                # ("Select", "select", "Select annotations (3)"),
            ]
        ):
            rb = QRadioButton(label)
            rb.setProperty("mode", mode)
            rb.setToolTip(tip)
            rb.setStyleSheet("font-size: 16px; padding: 6px 2px;")
            if i == 0:
                rb.setChecked(True)
            self.mode_group.addButton(rb, i)
            sl.addWidget(rb)
        self.mode_group.buttonClicked.connect(self._mode_changed)

        sl.addWidget(_section_label("Class for visual:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.state.classes)
        sl.addWidget(self.class_combo)
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
        sl.addLayout(class_sel_box)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        spacer.setFixedHeight(40)
        sl.addWidget(spacer)

        # -- Settings --
        sl.addWidget(_section_label("Settings"))
        sl.addWidget(QLabel("Object visualization:"))
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
        sl.addLayout(dm_r)

        # Polygon simplification slider
        sl.addWidget(QLabel("Polygon simplify:"))
        self.epsilon_slider = QSlider(Qt.Orientation.Horizontal)
        self.epsilon_slider.setRange(1, 20)
        self.epsilon_slider.setValue(5)
        self.epsilon_slider.setToolTip("Lower = more precise (0.001~0.020)")
        self.epsilon_label = QLabel("0.005")
        eps_r = QHBoxLayout()
        eps_r.addWidget(self.epsilon_slider)
        eps_r.addWidget(self.epsilon_label)
        self.epsilon_slider.valueChanged.connect(self._epsilon_changed)
        sl.addLayout(eps_r)

        # Overlap threshold slider
        sl.addWidget(QLabel("Overlap threshold:"))
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlap_slider.setRange(0, 50)
        self.overlap_slider.setValue(10)
        self.overlap_label = QLabel("10%")
        ov_r = QHBoxLayout()
        ov_r.addWidget(self.overlap_slider)
        ov_r.addWidget(self.overlap_label)
        self.overlap_slider.valueChanged.connect(self._overlap_changed)
        sl.addLayout(ov_r)

        self.fallback_cb = QCheckBox("Fallback to box if SAM fails")
        self.fallback_cb.setChecked(False)
        sl.addWidget(self.fallback_cb)

        # Output formats
        sl.addWidget(QLabel("Output formats:"))
        self.fmt_seg = QCheckBox("YOLO-Seg (Polygon)")
        self.fmt_seg.setChecked(True)
        self.fmt_mask = QCheckBox("PNG Mask")
        self.fmt_coco = QCheckBox("COCO JSON")
        self.fmt_obb = QCheckBox("OBB (Oriented Bounding Box)")
        for cb in (self.fmt_seg, self.fmt_mask, self.fmt_coco, self.fmt_obb):
            cb.stateChanged.connect(self._fmt_changed)
            sl.addWidget(cb)

        body.addWidget(segmentation_w)
        sl.addStretch()

        # -- Canvas --
        self.canvas = AnnotationCanvas()
        self.canvas.point_clicked.connect(self._on_point_click)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.label_selected.connect(self._on_label_selected)
        body.addWidget(self.canvas)

        # -- Annotation panel --
        annotation_sidebar_scroll = QScrollArea()
        annotation_sidebar_scroll.setWidgetResizable(True)
        annotation_sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        annotation_sidebar_scroll.setMinimumWidth(200)
        annotation_sidebar = QWidget()
        al = QVBoxLayout(annotation_sidebar)
        al.setContentsMargins(6, 6, 6, 6)
        al.setSpacing(4)

        # -- Folder loading --
        al.addWidget(_section_label("Folder loading"))
        img_folder_box = QHBoxLayout()
        self.folder_input = QLineEdit(config.get("images_folder", ""))
        self.folder_input.setPlaceholderText("Image folder")
        img_folder_box.addWidget(self.folder_input, 4)
        img_browse_btn = QPushButton("🗁")
        img_browse_btn.setFixedWidth(36)
        img_browse_btn.setToolTip("Browse images' folder")
        img_browse_btn.clicked.connect(self._browse_image_folder)
        img_folder_box.addWidget(img_browse_btn)
        al.addLayout(img_folder_box)

        output_folder_box = QHBoxLayout()
        self.output_input = QLineEdit(config.get("output_folder", ""))
        self.output_input.setPlaceholderText("Output folder")
        output_folder_box.addWidget(self.output_input, 4)
        output_browse_btn = QPushButton("🖿")
        output_browse_btn.setFixedWidth(36)
        output_browse_btn.setToolTip("Browse output folder")
        output_browse_btn.clicked.connect(self._browse_output_folder)
        output_folder_box.addWidget(output_browse_btn)
        al.addLayout(output_folder_box)

        load_btn = QPushButton("Load")
        load_btn.setObjectName("primaryBtn")
        load_btn.clicked.connect(self._load_folder)
        al.addWidget(load_btn)

        # -- Annotation list --
        al.addWidget(_section_label("Annotations"))
        self.label_list = QListWidget()
        self.label_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.label_list.setMaximumHeight(300)
        self.label_list.itemSelectionChanged.connect(self._on_list_selection)
        al.addWidget(self.label_list)

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
        al.addLayout(op1)

        op2 = QHBoxLayout()
        op2.setSpacing(4)
        del_btn = QPushButton("Delete")
        del_btn.clicked.connect(self._delete_selected)
        clr_btn = QPushButton("Clear All")
        clr_btn.clicked.connect(self._clear_all)
        op2.addWidget(del_btn)
        op2.addWidget(clr_btn)
        al.addLayout(op2)

        # Save
        save_btn = QPushButton("Save")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save_labels)
        al.addWidget(save_btn)

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
        self.setStyleSheet("""
            * { font-family: "Segoe UI", "Helvetica Neue", sans-serif; }
            QMainWindow, QWidget { background: #0f0f1a; color: #ccd6f6; }
            #navBar { background: #161625; border: 1px solid #1e2d4a; border-radius: 8px; }
            #segmentationPanel { background: #161625; border: 1px solid #1e2d4a; border-radius: 8px; }
            QPushButton {
                background: #1e2d4a; border: 1px solid #2a3f6f; border-radius: 5px;
                padding: 5px 10px; color: #ccd6f6; font-size: 12px;
            }
            QPushButton:hover { background: #2a3f6f; border-color: #4a6fa5; }
            QPushButton:pressed { background: #3a5f9f; }
            QPushButton#primaryBtn {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5b6abf,stop:1 #7c5cbf);
                border: none; color: #fff; font-weight: 600;
            }
            QPushButton#primaryBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6b7acf,stop:1 #8c6ccf);
            }
            QLineEdit, QComboBox {
                background: #1a1a2e; border: 1px solid #2a3f6f; border-radius: 4px;
                padding: 4px 8px; color: #ccd6f6; selection-background-color: #5b6abf;
            }
            QLineEdit:focus, QComboBox:focus { border-color: #5b6abf; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #1a1a2e; border: 1px solid #2a3f6f; color: #ccd6f6;
                selection-background-color: #5b6abf;
            }
            QListWidget {
                background: #12121f; border: 1px solid #1e2d4a; border-radius: 4px; color: #ccd6f6;
            }
            QListWidget::item { padding: 3px 6px; border-radius: 3px; }
            QListWidget::item:selected { background: #2a3f6f; }
            QListWidget::item:hover { background: #1e2d4a; }
            QRadioButton, QCheckBox { color: #8892b0; spacing: 8px; }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 18px; height: 18px;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #1e2d4a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5b6abf; width: 14px; height: 14px; margin: -5px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #5b6abf; border-radius: 2px; }
            QProgressBar {
                background: #1a1a2e; border: 1px solid #2a3f6f; border-radius: 4px;
                text-align: center; color: #ccd6f6; font-size: 10px;
            }
            QProgressBar::chunk { background: #5b6abf; border-radius: 3px; }
            QStatusBar { background: #0f0f1a; color: #8892b0; border-top: 1px solid #1e2d4a; }
            QScrollArea { border: none; }
            QScrollBar:vertical {
                background: #12121f; width: 8px; border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2a3f6f; min-height: 30px; border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover { background: #5b6abf; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QSplitter::handle { background: #1e2d4a; width: 2px; }
        """)

    # ==================================================================
    # Folder / image management
    # ==================================================================

    def _browse_image_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder")
        if d:
            self.folder_input.setText(d)

    def _browse_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.output_input.setText(d)

    def _load_folder(self):
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
        self._nav(-1)

    def _next_image(self):
        self._nav(1)

    def _jump_to(self):
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
        c = self.class_combo.currentText()
        return self.state.classes.index(c) if c in self.state.classes else 0

    def _start_busy(self):
        self.canvas.set_busy(True)
        self.progress_bar.show()

    def _end_busy(self):
        self.canvas.set_busy(False)
        self.progress_bar.hide()

    def _segment_text(self):
        if self.state.current_image is None:
            return
        prompts = [p.strip() for p in self.text_prompt.text().split(",") if p.strip()]
        if not prompts:
            return
        self._start_busy()
        self._set_status("Running text segmentation...")

        def _do():
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
        if self.state.current_image is None:
            return
        cid = self._get_class_id()
        self._start_busy()
        self._set_status(f"Segmenting ({x},{y})...")

        def _do():
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
        if self.state.current_image is None:
            return
        cid = self._get_class_id()
        self._start_busy()
        self._set_status("Box segmenting...")

        def _do():
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
        self.status_label.setText(msg)

    def _refresh_labels_ui(self):
        self.canvas.set_labels(
            self.state.current_labels, self.state.classes, self.state.selected_labels
        )
        self.label_list.blockSignals(True)
        self.label_list.clear()
        for idx, lb in enumerate(self.state.current_labels):
            cid = lb[0]
            cn = self.state.classes[cid] if cid < len(self.state.classes) else f"c{cid}"
            item = QListWidgetItem(f"  {idx + 1}. {cn}")
            item.setIcon(color_icon(LABEL_COLORS[cid % len(LABEL_COLORS)]))
            self.label_list.addItem(item)
        self.label_list.blockSignals(False)

    def _refresh_class_combos(self):
        for cb in (self.class_combo, self.change_combo):
            cur = cb.currentText()
            cb.clear()
            cb.addItems(self.state.classes)
            if cur in self.state.classes:
                cb.setCurrentText(cur)

    def _on_label_selected(self, idx):
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
        sel = set()
        for it in self.label_list.selectedItems():
            try:
                sel.add(int(it.text().strip().split(".")[0]) - 1)
            except ValueError:
                pass
        self.state.selected_labels = sel
        self.canvas.set_selected(sel)

    def _mode_changed(self, btn):
        self.canvas.set_mode(btn.property("mode"))

    def _set_mode(self, idx):
        btn = self.mode_group.button(idx)
        if btn:
            btn.setChecked(True)
            self.canvas.set_mode(btn.property("mode"))

    def _display_mode_changed(self, btn):
        dm = btn.property("dm")
        self.state.display_mode = dm
        self.canvas.set_display_mode(dm)

    def _epsilon_changed(self, val):
        v = val / 1000.0
        self.state.polygon_epsilon = v
        self.epsilon_label.setText(f"{v:.3f}")

    def _overlap_changed(self, val):
        v = val / 100.0
        self.state.overlap_threshold = v
        self.overlap_label.setText("Off" if val == 0 else f"{val}%")

    def _fmt_changed(self):
        self.state.output_formats["obb"] = self.fmt_obb.isChecked()
        self.state.output_formats["seg"] = self.fmt_seg.isChecked()
        self.state.output_formats["mask"] = self.fmt_mask.isChecked()
        self.state.output_formats["coco"] = self.fmt_coco.isChecked()

    # ==================================================================
    # Annotation operations
    # ==================================================================

    def _delete_selected(self):
        if not self.state.selected_labels:
            return
        for i in sorted(self.state.selected_labels, reverse=True):
            if i < len(self.state.current_labels):
                del self.state.current_labels[i]
        self.state.selected_labels.clear()
        self._refresh_labels_ui()
        self._set_status("Deleted selected annotations")

    def _clear_all(self):
        self.state.current_labels.clear()
        self.state.selected_labels.clear()
        self._refresh_labels_ui()
        self._set_status("Cleared all annotations")

    def _select_all(self):
        self.state.selected_labels = set(range(len(self.state.current_labels)))
        self.canvas.set_selected(self.state.selected_labels)
        self.label_list.selectAll()

    def _deselect_all(self):
        self.state.selected_labels.clear()
        self.canvas.set_selected(set())
        self.label_list.clearSelection()

    def _change_selected_class(self):
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
        n = self.new_class_input.text().strip()
        if not n or n in self.state.classes:
            return
        self.state.classes.append(n)
        persist_classes(self.state.classes)
        self._refresh_class_combos()
        self.new_class_input.clear()
        self._set_status(f"Added class: {n}")

    def _delete_class(self):
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
        self.state.output_folder = Path(self.output_input.text().strip() or "output")
        msg = auto_save_labels(self.state)
        self._set_status(msg or "Saved successfully")
