"""
AnnotationCanvas v2 — QPainter vector rendering canvas

Features: middle-click pan, coordinate display, text background, busy overlay, double-click fit
"""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QIcon,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import QWidget

LABEL_COLORS = [
    QColor(0, 245, 255),
    QColor(255, 95, 86),
    QColor(255, 210, 63),
    QColor(125, 255, 87),
    QColor(255, 77, 166),
    QColor(122, 92, 255),
    QColor(255, 149, 5),
    QColor(64, 224, 208),
    QColor(255, 61, 113),
    QColor(173, 255, 47),
    QColor(0, 191, 255),
    QColor(255, 0, 255),
    QColor(255, 255, 102),
    QColor(0, 255, 170),
    QColor(255, 128, 0),
    QColor(205, 127, 255),
]
SELECTED_COLOR = QColor(0, 255, 255)
BG_COLOR = QColor(15, 15, 26)
CANVAS_ICON_EXCLUSIONS = {"person", "buoy"}

_icon_cache = {}


def icon_asset_exists(class_name: str) -> bool:
    """Return whether a class has a dedicated icon asset on disk."""
    norm_name = class_name.lower().strip().replace(" ", "_")
    return (Path("themes/icons") / f"{norm_name}.png").exists()


def has_canvas_label_icon(class_name: str) -> bool:
    """Return whether the canvas should render an icon for a class label."""
    norm_name = class_name.lower().strip().replace(" ", "_")
    if norm_name in CANVAS_ICON_EXCLUSIONS:
        return False
    return icon_asset_exists(class_name)


def get_class_pixmap(class_name: str) -> QPixmap | None:
    """Return a class icon pixmap when an asset exists, otherwise no pixmap."""
    norm_name = class_name.lower().strip().replace(" ", "_")
    if norm_name in _icon_cache:
        return _icon_cache[norm_name]

    icon_path = Path("themes/icons") / f"{norm_name}.png"
    if icon_path.exists():
        pm = QPixmap(str(icon_path))
        if not pm.isNull():
            _icon_cache[norm_name] = pm
            return pm

    return None


def get_class_icon(class_name: str, fallback_color: QColor | None = None) -> QIcon:
    """Wrap the class pixmap in a QIcon, or a color swatch if requested."""
    pm = get_class_pixmap(class_name)
    if pm is not None:
        return QIcon(pm)
    if fallback_color:
        pm2 = QPixmap(12, 12)
        pm2.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm2)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(fallback_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, 12, 12, 3, 3)
        p.end()
        return QIcon(pm2)
    return QIcon()


def numpy_to_qimage(a):
    """Convert an RGB numpy array into a detached QImage for painting."""
    h, w, ch = a.shape
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return QImage(a.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888).copy()


class AnnotationCanvas(QWidget):
    """Canvas widget that renders images, masks, boxes, and annotation labels."""

    point_clicked = pyqtSignal(int, int)
    box_drawn = pyqtSignal(int, int, int, int)
    label_selected = pyqtSignal(int)
    cursor_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        """Initialize canvas state used for panning, zooming, and label rendering."""
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.display_aabb = True
        self._pixmap = None
        self._img_w = self._img_h = 0
        self._labels = []
        self._classes = []
        self._selected = set()
        self.class_thresholds = {}
        self._display_mode = "both"
        self._mask_overlay = None
        self._mode = "point"
        self._dragging = False
        self._panning = False
        self._drag_start = QPoint()
        self._drag_current = QPoint()
        self._pan_start = QPoint()
        self._pan_off0 = QPointF()
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._hover_label = -1
        self._cursor_img = (-1, -1)
        self._busy = False
        self._space_held = False
        self.mask_opacity = 0.62
        self._positive_prompt_points = []
        self._negative_prompt_points = []

    # --- public ---
    def set_image(self, img_rgb):
        """Docstring for set_image."""
        self._pixmap = QPixmap.fromImage(numpy_to_qimage(img_rgb))
        self._img_w, self._img_h = img_rgb.shape[1], img_rgb.shape[0]
        self._mask_overlay = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._fit_image()
        self.update()

    def set_labels(self, labels, classes, selected=None):
        """Docstring for set_labels."""
        self._labels = labels
        self._classes = classes
        self._selected = selected or set()
        self._mask_overlay = None
        self.update()

    def set_selected(self, s):
        """Docstring for set_selected."""
        self._selected = s
        self.update()

    def set_mode(self, m):
        """Docstring for set_mode."""
        self._mode = m
        self._upd_cursor()

    def set_display_mode(self, m):
        """Docstring for set_display_mode."""
        self._display_mode = m
        self._mask_overlay = None
        self.update()

    def set_prompt_points(self, positive_points, negative_points):
        """Update the temporary positive and negative point prompts shown on canvas."""
        self._positive_prompt_points = [tuple(point) for point in positive_points]
        self._negative_prompt_points = [tuple(point) for point in negative_points]
        self.update()

    def set_mask_opacity(self, opacity):
        """Update the mask overlay opacity and rebuild cached mask rendering."""
        self.mask_opacity = max(0.05, min(1.0, float(opacity)))
        self._mask_overlay = None
        self.update()

    def set_busy(self, b):
        """Docstring for set_busy."""
        self._busy = b
        self._upd_cursor()
        self.update()

    def fit_view(self):
        """Docstring for fit_view."""
        self._fit_image()
        self.update()

    # --- coord ---
    def _fit_image(self):
        """Docstring for _fit_image."""
        if not self._img_w:
            return
        z = min(self.width() / self._img_w, self.height() / self._img_h) * 0.95
        self._zoom = z
        self._offset = QPointF(
            (self.width() - self._img_w * z) / 2, (self.height() - self._img_h * z) / 2
        )

    def _w2i(self, pos):
        """Docstring for _w2i."""
        x = (pos.x() - self._offset.x()) / self._zoom
        y = (pos.y() - self._offset.y()) / self._zoom
        return int(max(0, min(self._img_w, x))), int(max(0, min(self._img_h, y)))

    def _upd_cursor(self):
        """Docstring for _upd_cursor."""
        if self._busy:
            self.setCursor(Qt.CursorShape.WaitCursor)
        elif self._mode in ("box", "point", "click"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # --- mask cache ---
    def _build_mask_overlay(self):
        """Docstring for _build_mask_overlay."""
        if not self._img_w:
            return None
        ov = QImage(self._img_w, self._img_h, QImage.Format.Format_ARGB32_Premultiplied)
        ov.fill(QColor(0, 0, 0, 0))
        p = QPainter(ov)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for idx, lb in enumerate(self._labels):
            cid = lb[0]
            cn = self._classes[cid] if cid < len(self._classes) else ""
            score = lb[4] if len(lb) > 4 else None

            if score is not None and score < getattr(self, "class_thresholds", {}).get(cn, 0.25):
                continue

            co = LABEL_COLORS[cid % len(LABEL_COLORS)]
            pc = lb[2] if len(lb) > 2 and lb[2] else lb[1]
            if not pc:
                continue
            poly = QPolygonF()
            for i in range(0, len(pc), 2):
                poly.append(QPointF(pc[i] * self._img_w, pc[i + 1] * self._img_h))
            fc = QColor(SELECTED_COLOR if idx in self._selected else co)
            base_alpha = int(round(255 * self.mask_opacity))
            fc.setAlpha(min(255, base_alpha + 48) if idx in self._selected else base_alpha)
            p.setBrush(QBrush(fc))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(poly)
        p.end()
        return QPixmap.fromImage(ov)

    def _class_name_for_label(self, label):
        """Return the display class name for a label tuple."""
        class_id = label[0]
        return self._classes[class_id] if class_id < len(self._classes) else f"c{class_id}"

    def _label_score(self, label):
        """Return the confidence score stored on a label, if any."""
        return label[4] if len(label) > 4 else None

    def _is_label_visible(self, label):
        """Return whether a label should be shown under the active thresholds."""
        class_name = self._class_name_for_label(label)
        score = self._label_score(label)
        if score is None:
            return True
        return score >= getattr(self, "class_thresholds", {}).get(class_name, 0.25)

    def _start_panning(self, pos):
        """Enter panning mode from the provided mouse position."""
        self._panning = True
        self._pan_start = pos
        self._pan_off0 = QPointF(self._offset)
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _begin_drag(self, pos):
        """Start a box or selection drag from the provided position."""
        self._dragging = True
        self._drag_start = pos
        self._drag_current = pos

    def _find_label_at(self, ix, iy):
        """Return the topmost visible label index under an image-space point."""
        from core.utils import point_in_aabb

        for index, label in reversed(list(enumerate(self._labels))):
            if not self._is_label_visible(label):
                continue
            if point_in_aabb(ix, iy, label[1], self._img_w, self._img_h):
                return index
        return None

    def _handle_select_press(self, event):
        """Update selection state for a click in select mode."""
        image_x, image_y = self._w2i(event.pos())
        label_index = self._find_label_at(image_x, image_y)
        is_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if label_index is not None:
            if is_shift:
                if label_index in self._selected:
                    self._selected.discard(label_index)
                else:
                    self._selected.add(label_index)
            else:
                self._selected = {label_index}
            self.label_selected.emit(label_index)
            self.update()
            return

        if not is_shift:
            self._selected.clear()
            self.label_selected.emit(-1)
        self._begin_drag(event.pos())

    def _update_cursor_position(self, pos):
        """Refresh the image-space cursor coordinates from a widget position."""
        if not self._pixmap:
            return
        image_x, image_y = self._w2i(pos)
        inside = 0 <= image_x <= self._img_w and 0 <= image_y <= self._img_h
        self._cursor_img = (image_x, image_y) if inside else (-1, -1)
        self.cursor_moved.emit(image_x, image_y)

    def _update_hover_label(self, pos):
        """Update the hovered label based on the current pointer position."""
        if not self._pixmap:
            return
        image_x, image_y = self._w2i(pos)
        hovered_index = self._find_label_at(image_x, image_y)
        self._hover_label = hovered_index if hovered_index is not None else -1

    def _paint_empty_state(self, painter):
        """Render the placeholder message when no image is loaded."""
        painter.setPen(QPen(QColor(100, 100, 140)))
        painter.setFont(QFont("Segoe UI", 16))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Load an image folder to start annotating",
        )

    def _paint_outlines(self, painter, inverse_zoom):
        """Render visible annotation outlines and box fills."""
        for index, label in enumerate(self._labels):
            if not self._is_label_visible(label):
                continue

            class_id, obb = label[0], label[1]
            is_selected = index in self._selected
            is_hovered = index == self._hover_label
            color = SELECTED_COLOR if is_selected else LABEL_COLORS[class_id % len(LABEL_COLORS)]
            pen_width = (3.0 if is_selected else 2.0 if is_hovered else 1.2) * inverse_zoom
            pen = QPen(color, pen_width)
            if is_hovered and not is_selected:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)

            fill_color = QColor(color)
            fill_color.setAlpha(64 if is_selected else 32)
            painter.setBrush(QBrush(fill_color))

            if getattr(self, "display_aabb", False):
                xs = [obb[i] * self._img_w for i in range(0, 8, 2)]
                ys = [obb[i + 1] * self._img_h for i in range(0, 8, 2)]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                painter.drawRect(QRectF(min_x, min_y, max_x - min_x, max_y - min_y))
                continue

            polygon = QPolygonF()
            for coord_index in range(0, 8, 2):
                polygon.append(
                    QPointF(
                        obb[coord_index] * self._img_w,
                        obb[coord_index + 1] * self._img_h,
                    )
                )
            painter.drawPolygon(polygon)

    def _paint_label_badges(self, painter, inverse_zoom):
        """Render label chips with optional icons above each visible annotation."""
        font_size = max(9, int(11 * inverse_zoom))
        font = QFont("Segoe UI", font_size)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetricsF(font)

        for index, label in enumerate(self._labels):
            if not self._is_label_visible(label):
                continue

            class_id, obb = label[0], label[1]
            class_name = self._class_name_for_label(label)
            score = self._label_score(label)
            track_id = label[5] if len(label) > 5 else None
            is_selected = index in self._selected
            color = SELECTED_COLOR if is_selected else LABEL_COLORS[class_id % len(LABEL_COLORS)]

            score_text = f" - {score:.2f}" if score is not None else ""
            track_text = f" T{track_id} |" if track_id is not None else ""
            text = f"{track_text} ID {index + 1} - {class_name}{score_text} "
            pixmap = get_class_pixmap(class_name) if has_canvas_label_icon(class_name) else None

            if getattr(self, "display_aabb", False):
                xs = [obb[i] * self._img_w for i in range(0, 8, 2)]
                ys = [obb[i + 1] * self._img_h for i in range(0, 8, 2)]
                text_x, text_y = min(xs), min(ys) - 3 * inverse_zoom
            else:
                text_x = obb[0] * self._img_w
                text_y = obb[1] * self._img_h - 3 * inverse_zoom

            text_rect = metrics.boundingRect(text)
            icon_size = text_rect.height() if pixmap else 0
            total_width = text_rect.width() + (icon_size + 4 * inverse_zoom if pixmap else 0)
            background_rect = QRectF(
                text_x,
                text_y - text_rect.height(),
                total_width,
                text_rect.height() + 2 * inverse_zoom,
            )

            painter.setPen(Qt.PenStyle.NoPen)
            background_color = QColor(color)
            background_color.setAlpha(180)
            painter.setBrush(QBrush(background_color))
            painter.drawRoundedRect(background_rect, 2 * inverse_zoom, 2 * inverse_zoom)

            if pixmap:
                scaled_pixmap = pixmap.scaled(
                    int(icon_size),
                    int(icon_size),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                painter.drawPixmap(
                    QPointF(text_x + 2 * inverse_zoom, text_y - icon_size + 1 * inverse_zoom),
                    scaled_pixmap,
                )

            painter.setPen(QPen(QColor(15, 15, 26)))
            painter.drawText(
                QPointF(text_x + (icon_size + 4 * inverse_zoom if pixmap else 0), text_y),
                text,
            )

    def _paint_prompt_points(self, painter, inverse_zoom):
        """Render queued positive and negative point prompts."""
        for points, fill_color in (
            (self._positive_prompt_points, QColor(88, 255, 128)),
            (self._negative_prompt_points, QColor(255, 96, 96)),
        ):
            for point_x, point_y in points:
                radius = 5.5 * inverse_zoom
                painter.setPen(QPen(QColor(15, 15, 26), 1.6 * inverse_zoom))
                painter.setBrush(QBrush(fill_color))
                painter.drawEllipse(QPointF(point_x, point_y), radius, radius)

    def _paint_drag_rect(self, painter, inverse_zoom):
        """Render the active selection or box drag rectangle."""
        if not self._dragging or self._panning:
            return
        start_x, start_y = self._w2i(self._drag_start)
        current_x, current_y = self._w2i(self._drag_current)
        rect = QRectF(
            min(start_x, current_x),
            min(start_y, current_y),
            abs(current_x - start_x),
            abs(current_y - start_y),
        )
        painter.setPen(QPen(QColor(255, 165, 0), 2 * inverse_zoom, Qt.PenStyle.DashDotLine))
        painter.setBrush(QBrush(QColor(255, 165, 0, 40)))
        painter.drawRect(rect)

    def _paint_coordinate_overlay(self):
        """Render the image coordinate chip in widget space."""
        if self._cursor_img[0] < 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        text = f"  ({self._cursor_img[0]}, {self._cursor_img[1]})  "
        if 0 <= self._hover_label < len(self._labels):
            class_id = self._labels[self._hover_label][0]
            class_name = (
                self._classes[class_id] if class_id < len(self._classes) else f"c{class_id}"
            )
            text += f"[{self._hover_label + 1}. {class_name}]  "
        font = QFont("Consolas", 10)
        painter.setFont(font)
        metrics = QFontMetricsF(font)
        text_rect = metrics.boundingRect(text)
        box_x, box_y = 8, self.height() - 8
        background_rect = QRectF(
            box_x,
            box_y - text_rect.height() - 2,
            text_rect.width() + 4,
            text_rect.height() + 6,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
        painter.drawRoundedRect(background_rect, 4, 4)
        painter.setPen(QPen(QColor(200, 200, 220)))
        painter.drawText(QPointF(box_x + 2, box_y), text)
        painter.end()

    def _paint_busy_overlay(self):
        """Render the blocking busy overlay while SAM work is running."""
        if not self._busy:
            return
        painter = QPainter(self)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 120)))
        painter.drawRect(self.rect())
        painter.setPen(QPen(QColor(255, 200, 50)))
        painter.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "SAM inference running...")
        painter.end()

    # --- paint ---
    def paintEvent(self, a0):
        """Docstring for paintEvent."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), BG_COLOR)
        if self._pixmap is None:
            self._paint_empty_state(painter)
            painter.end()
            return

        painter.translate(self._offset)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._pixmap)

        if self._display_mode in ("mask", "both"):
            if self._mask_overlay is None:
                self._mask_overlay = self._build_mask_overlay()
            if self._mask_overlay:
                painter.drawPixmap(0, 0, self._mask_overlay)

        inverse_zoom = 1.0 / max(self._zoom, 0.01)

        if self._display_mode in ("outline", "both"):
            self._paint_outlines(painter, inverse_zoom)
        self._paint_label_badges(painter, inverse_zoom)
        self._paint_prompt_points(painter, inverse_zoom)
        self._paint_drag_rect(painter, inverse_zoom)
        painter.end()
        self._paint_coordinate_overlay()
        self._paint_busy_overlay()

    # --- mouse ---
    def keyPressEvent(self, a0):
        """Docstring for keyPressEvent."""
        if a0.key() == Qt.Key.Key_Space and not a0.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(a0)

    def keyReleaseEvent(self, a0):
        """Docstring for keyReleaseEvent."""
        if a0.key() == Qt.Key.Key_Space and not a0.isAutoRepeat():
            self._space_held = False
            self._upd_cursor()
        else:
            super().keyReleaseEvent(a0)

    def mousePressEvent(self, a0):
        """Docstring for mousePressEvent."""
        if a0.button() == Qt.MouseButton.MiddleButton or a0.button() == Qt.MouseButton.RightButton:
            self._start_panning(a0.pos())
            return
        if a0.button() == Qt.MouseButton.LeftButton and self._space_held:
            self._start_panning(a0.pos())
            return
        if a0.button() != Qt.MouseButton.LeftButton or self._busy:
            return
        if self._mode == "box":
            self._begin_drag(a0.pos())
            return
        if self._mode == "select":
            self._handle_select_press(a0)
            return
        if self._mode in ("point", "click"):
            image_x, image_y = self._w2i(a0.pos())
            self.point_clicked.emit(image_x, image_y)

    def mouseMoveEvent(self, a0):
        """Docstring for mouseMoveEvent."""
        self._update_cursor_position(a0.pos())
        if self._panning:
            delta = a0.pos() - self._pan_start
            self._offset = self._pan_off0 + QPointF(delta)
            self.update()
            return
        if self._dragging:
            self._drag_current = a0.pos()
            self.update()
            return
        previous_hover = self._hover_label
        self._update_hover_label(a0.pos())
        if self._hover_label != previous_hover:
            self.update()
            return
        self.update()

    def mouseReleaseEvent(self, a0):
        """Docstring for mouseReleaseEvent."""
        if self._panning and a0.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self._upd_cursor()
            return
        if a0.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        sx, sy = self._w2i(self._drag_start)
        ex, ey = self._w2i(a0.pos())
        if abs(ex - sx) < 4 and abs(ey - sy) < 4:
            self.update()
            return
        if self._mode == "box":
            self.box_drawn.emit(sx, sy, ex, ey)
        elif self._mode == "select":
            from core.utils import find_labels_in_box

            modifiers = a0.modifiers()
            is_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            if not is_shift:
                self._selected.clear()

            for i in find_labels_in_box(sx, sy, ex, ey, self._labels, self._img_w, self._img_h):
                self._selected.add(i)
            self.label_selected.emit(-1)
        self.update()

    def mouseDoubleClickEvent(self, a0):
        """Docstring for mouseDoubleClickEvent."""
        if a0.button() == Qt.MouseButton.LeftButton:
            self.fit_view()

    def wheelEvent(self, a0):
        """Docstring for wheelEvent."""
        if a0 is None:
            return
        f = 1.12 if a0.angleDelta().y() > 0 else 1 / 1.12
        mp = a0.position()
        op = QPointF(
            (mp.x() - self._offset.x()) / self._zoom,
            (mp.y() - self._offset.y()) / self._zoom,
        )
        self._zoom = max(0.05, min(30.0, self._zoom * f))
        nw = QPointF(
            op.x() * self._zoom + self._offset.x(),
            op.y() * self._zoom + self._offset.y(),
        )
        self._offset += mp - nw
        self.update()

    def resizeEvent(self, a0):
        """Docstring for resizeEvent."""
        if self._pixmap:
            self._fit_image()
        super().resizeEvent(a0)
