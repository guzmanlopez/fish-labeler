"""
AnnotationCanvas v2 — QPainter vector rendering canvas

Features: middle-click pan, coordinate display, text background, busy overlay, double-click fit
"""

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
)
from PyQt6.QtWidgets import QWidget

LABEL_COLORS = [
    QColor(46, 204, 113),
    QColor(231, 76, 60),
    QColor(52, 152, 219),
    QColor(241, 196, 15),
    QColor(155, 89, 182),
    QColor(26, 188, 156),
    QColor(230, 126, 34),
    QColor(236, 112, 160),
]
SELECTED_COLOR = QColor(0, 255, 255)
BG_COLOR = QColor(15, 15, 26)


def numpy_to_qimage(a):
    h, w, ch = a.shape
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return QImage(a.tobytes(), w, h, ch * w, QImage.Format.Format_RGB888).copy()


class AnnotationCanvas(QWidget):
    point_clicked = pyqtSignal(int, int)
    box_drawn = pyqtSignal(int, int, int, int)
    label_selected = pyqtSignal(int)
    cursor_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._pixmap = None
        self._img_w = self._img_h = 0
        self._labels = []
        self._classes = []
        self._selected = set()
        self._display_mode = "outline"
        self._mask_overlay = None
        self._mode = "click"
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

    # --- public ---
    def set_image(self, img_rgb):
        self._pixmap = QPixmap.fromImage(numpy_to_qimage(img_rgb))
        self._img_w, self._img_h = img_rgb.shape[1], img_rgb.shape[0]
        self._mask_overlay = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._fit_image()
        self.update()

    def set_labels(self, labels, classes, selected=None):
        self._labels = labels
        self._classes = classes
        self._selected = selected or set()
        self._mask_overlay = None
        self.update()

    def set_selected(self, s):
        self._selected = s
        self.update()

    def set_mode(self, m):
        self._mode = m
        self._upd_cursor()

    def set_display_mode(self, m):
        self._display_mode = m
        self._mask_overlay = None
        self.update()

    def set_busy(self, b):
        self._busy = b
        self._upd_cursor()
        self.update()

    def fit_view(self):
        self._fit_image()
        self.update()

    # --- coord ---
    def _fit_image(self):
        if not self._img_w:
            return
        z = min(self.width() / self._img_w, self.height() / self._img_h) * 0.95
        self._zoom = z
        self._offset = QPointF(
            (self.width() - self._img_w * z) / 2, (self.height() - self._img_h * z) / 2
        )

    def _w2i(self, pos):
        x = (pos.x() - self._offset.x()) / self._zoom
        y = (pos.y() - self._offset.y()) / self._zoom
        return int(max(0, min(self._img_w, x))), int(max(0, min(self._img_h, y)))

    def _upd_cursor(self):
        if self._busy:
            self.setCursor(Qt.CursorShape.WaitCursor)
        elif self._mode in ("box", "click"):
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # --- mask cache ---
    def _build_mask_overlay(self):
        if not self._img_w:
            return None
        ov = QImage(self._img_w, self._img_h, QImage.Format.Format_ARGB32_Premultiplied)
        ov.fill(QColor(0, 0, 0, 0))
        p = QPainter(ov)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for idx, lb in enumerate(self._labels):
            cid = lb[0]
            co = LABEL_COLORS[cid % len(LABEL_COLORS)]
            pc = lb[2] if len(lb) > 2 and lb[2] else lb[1]
            if not pc:
                continue
            poly = QPolygonF()
            for i in range(0, len(pc), 2):
                poly.append(QPointF(pc[i] * self._img_w, pc[i + 1] * self._img_h))
            fc = QColor(SELECTED_COLOR if idx in self._selected else co)
            fc.setAlpha(128 if idx in self._selected else 80)
            p.setBrush(QBrush(fc))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(poly)
        p.end()
        return QPixmap.fromImage(ov)

    # --- paint ---
    def paintEvent(self, a0):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), BG_COLOR)
        if self._pixmap is None:
            p.setPen(QPen(QColor(100, 100, 140)))
            p.setFont(QFont("Segoe UI", 16))
            p.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Load an image folder to start annotating",
            )
            p.end()
            return

        p.translate(self._offset)
        p.scale(self._zoom, self._zoom)
        p.drawPixmap(0, 0, self._pixmap)

        if self._display_mode in ("mask", "both"):
            if self._mask_overlay is None:
                self._mask_overlay = self._build_mask_overlay()
            if self._mask_overlay:
                p.drawPixmap(0, 0, self._mask_overlay)

        iz = 1.0 / max(self._zoom, 0.01)

        if self._display_mode in ("outline", "both"):
            for idx, lb in enumerate(self._labels):
                cid, obb = lb[0], lb[1]
                is_s = idx in self._selected
                is_h = idx == self._hover_label
                co = SELECTED_COLOR if is_s else LABEL_COLORS[cid % len(LABEL_COLORS)]
                pw = (3.0 if is_s else 2.0 if is_h else 1.2) * iz
                pen = QPen(co, pw)
                if is_h and not is_s:
                    pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.setBrush(Qt.BrushStyle.NoBrush)
                poly = QPolygonF()
                for i in range(0, 8, 2):
                    poly.append(QPointF(obb[i] * self._img_w, obb[i + 1] * self._img_h))
                p.drawPolygon(poly)

        # text labels with background
        fs = max(9, int(11 * iz))
        fnt = QFont("Segoe UI", fs)
        fnt.setBold(True)
        p.setFont(fnt)
        fm = QFontMetricsF(fnt)
        for idx, lb in enumerate(self._labels):
            cid, obb = lb[0], lb[1]
            is_s = idx in self._selected
            co = SELECTED_COLOR if is_s else LABEL_COLORS[cid % len(LABEL_COLORS)]
            cn = self._classes[cid] if cid < len(self._classes) else f"c{cid}"
            txt = f" {idx + 1}. {cn} "
            tx, ty = obb[0] * self._img_w, obb[1] * self._img_h - 3 * iz
            tr = fm.boundingRect(txt)
            bgr = QRectF(tx, ty - tr.height(), tr.width(), tr.height() + 2 * iz)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(0, 0, 0, 180)))
            p.drawRoundedRect(bgr, 2 * iz, 2 * iz)
            p.setPen(QPen(co))
            p.drawText(QPointF(tx, ty), txt)

        # drag rect
        if self._dragging and not self._panning:
            sx, sy = self._w2i(self._drag_start)
            cx, cy = self._w2i(self._drag_current)
            r = QRectF(min(sx, cx), min(sy, cy), abs(cx - sx), abs(cy - sy))
            p.setPen(QPen(QColor(255, 165, 0), 2 * iz, Qt.PenStyle.DashDotLine))
            p.setBrush(QBrush(QColor(255, 165, 0, 40)))
            p.drawRect(r)
        p.end()

        # coord overlay (widget coords)
        if self._cursor_img[0] >= 0:
            p2 = QPainter(self)
            p2.setRenderHint(QPainter.RenderHint.Antialiasing)
            ct = f"  ({self._cursor_img[0]}, {self._cursor_img[1]})  "
            if 0 <= self._hover_label < len(self._labels):
                ci = self._labels[self._hover_label][0]
                cn = self._classes[ci] if ci < len(self._classes) else f"c{ci}"
                ct += f"[{self._hover_label + 1}. {cn}]  "
            cf = QFont("Consolas", 10)
            p2.setFont(cf)
            cfm = QFontMetricsF(cf)
            cr = cfm.boundingRect(ct)
            bx, by = 8, self.height() - 8
            bgr = QRectF(bx, by - cr.height() - 2, cr.width() + 4, cr.height() + 6)
            p2.setPen(Qt.PenStyle.NoPen)
            p2.setBrush(QBrush(QColor(0, 0, 0, 200)))
            p2.drawRoundedRect(bgr, 4, 4)
            p2.setPen(QPen(QColor(200, 200, 220)))
            p2.drawText(QPointF(bx + 2, by), ct)
            p2.end()

        # busy overlay
        if self._busy:
            p3 = QPainter(self)
            p3.setPen(Qt.PenStyle.NoPen)
            p3.setBrush(QBrush(QColor(0, 0, 0, 120)))
            p3.drawRect(self.rect())
            p3.setPen(QPen(QColor(255, 200, 50)))
            p3.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
            p3.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "SAM inference running...")
            p3.end()

    # --- mouse ---
    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key.Key_Space and not a0.isAutoRepeat():
            self._space_held = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(a0)

    def keyReleaseEvent(self, a0):
        if a0.key() == Qt.Key.Key_Space and not a0.isAutoRepeat():
            self._space_held = False
            self._upd_cursor()
        else:
            super().keyReleaseEvent(a0)

    def mousePressEvent(self, a0):
        if a0.button() == Qt.MouseButton.MiddleButton or a0.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_start = a0.pos()
            self._pan_off0 = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if a0.button() == Qt.MouseButton.LeftButton and self._space_held:
            self._panning = True
            self._pan_start = a0.pos()
            self._pan_off0 = QPointF(self._offset)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if a0.button() != Qt.MouseButton.LeftButton or self._busy:
            return
        if self._mode == "box":
            self._dragging = True
            self._drag_start = a0.pos()
            self._drag_current = a0.pos()
        elif self._mode == "select":
            ix, iy = self._w2i(a0.pos())
            from core.utils import find_clicked_label

            idx = find_clicked_label(ix, iy, self._labels, self._img_w, self._img_h)
            if idx is not None:
                if idx in self._selected:
                    self._selected.discard(idx)
                else:
                    self._selected.add(idx)
                self.label_selected.emit(idx)
                self.update()
            else:
                self._dragging = True
                self._drag_start = a0.pos()
                self._drag_current = a0.pos()
        elif self._mode == "click":
            ix, iy = self._w2i(a0.pos())
            self.point_clicked.emit(ix, iy)

    def mouseMoveEvent(self, a0):
        if self._pixmap:
            ix, iy = self._w2i(a0.pos())
            self._cursor_img = (
                (ix, iy) if 0 <= ix <= self._img_w and 0 <= iy <= self._img_h else (-1, -1)
            )
            self.cursor_moved.emit(ix, iy)
        if self._panning:
            d = a0.pos() - self._pan_start
            self._offset = self._pan_off0 + QPointF(d)
            self.update()
            return
        if self._dragging:
            self._drag_current = a0.pos()
            self.update()
            return
        if self._pixmap:
            from core.utils import find_clicked_label

            ix, iy = self._w2i(a0.pos())
            nh = find_clicked_label(ix, iy, self._labels, self._img_w, self._img_h)
            nh = nh if nh is not None else -1
            if nh != self._hover_label:
                self._hover_label = nh
        self.update()

    def mouseReleaseEvent(self, a0):
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

            for i in find_labels_in_box(sx, sy, ex, ey, self._labels, self._img_w, self._img_h):
                self._selected.add(i)
            self.label_selected.emit(-1)
        self.update()

    def mouseDoubleClickEvent(self, a0):
        if a0.button() == Qt.MouseButton.LeftButton:
            self.fit_view()

    def wheelEvent(self, a0):
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
        if self._pixmap:
            self._fit_image()
        super().resizeEvent(a0)
