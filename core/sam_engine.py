"""
SAM 3 model loading and inference
Preserves original load_sam3_model / load_sam_model logic
"""

import os

import cv2

from core.logger import logger
from core.utils import (
    box_to_obb,
    check_mask_overlap,
    mask_to_binary_image,
    mask_to_obb,
    mask_to_polygon,
    polygon_to_mask,
)

DEFAULT_SAM_CONF = 0.25


class SAMEngine:
    """SAMEngine class."""

    """Wrapper for SAM 3 model operations"""

    def __init__(self, model_path="sam3.pt", device="cuda:0"):
        """Docstring for __init__."""
        self.model_path = model_path
        self.device = device
        self._predictor = None  # SAM3SemanticPredictor (for text prompts)
        self._sam_model = None  # SAM model (for click/box)

    def _ensure_predictor(self):
        """Docstring for _ensure_predictor."""
        if self._predictor is None:
            from ultralytics.models.sam import SAM3SemanticPredictor

            self._predictor = SAM3SemanticPredictor(
                overrides=dict(
                    conf=DEFAULT_SAM_CONF,
                    model=self.model_path,
                    device=self.device,
                    half=True,
                    verbose=False,
                )
            )
            logger.info(f"[OK] SAM 3 semantic model loaded ({self.device})")
        return self._predictor

    def _ensure_sam(self):
        """Docstring for _ensure_sam."""
        if self._sam_model is None:
            from ultralytics import SAM

            self._sam_model = SAM(self.model_path)
            logger.info(f"[OK] SAM click/box model loaded ({self.device})")
        return self._sam_model

    def _predict_with_temp_image(self, image_rgb, **predict_kwargs):
        """Run SAM prediction by writing the current image to a temporary file."""
        sam = self._ensure_sam()
        temp_path = "_temp_sam_img.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        try:
            return sam.predict(source=temp_path, device=self.device, **predict_kwargs)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _build_mask_label(
        self,
        mask,
        class_id,
        existing_labels,
        img_w,
        img_h,
        polygon_epsilon,
        overlap_threshold,
        success_message,
        missing_message,
    ):
        """Convert a SAM mask into a stored label tuple after overlap checks."""
        obb = mask_to_obb(mask, img_w, img_h)
        if obb is None:
            return None, missing_message

        poly = mask_to_polygon(mask, img_w, img_h, polygon_epsilon)
        mask_binary = mask_to_binary_image(mask)
        is_over, overlap_index, overlap_ratio = check_mask_overlap(
            mask_binary, existing_labels, img_w, img_h, overlap_threshold
        )
        if is_over:
            return (
                None,
                f"Overlaps with annotation {overlap_index + 1} ({overlap_ratio * 100:.0f}%)",
            )
        return (class_id, obb, poly, mask_binary, 1.0), success_message

    def segment_text(
        self,
        image_rgb,
        prompts,
        classes,
        existing_labels,
        polygon_epsilon=0.005,
        overlap_threshold=0.1,
    ):
        """Text prompt segmentation, returns (new_labels, added, skipped, new_classes)"""
        predictor = self._ensure_predictor()
        temp_path = "_temp_sam_img.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        try:
            predictor.set_image(temp_path)
            results = predictor(text=prompts)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not results or len(results) == 0 or results[0].masks is None:
            return [], 0, 0, []

        masks = results[0].masks.data
        boxes = results[0].boxes
        img_h, img_w = image_rgb.shape[:2]

        new_classes = [p for p in prompts if p not in classes]
        all_classes = list(classes) + new_classes

        new_labels = []
        added = skipped = 0
        for i, mask in enumerate(masks):
            cls_idx = int(boxes.cls[i].item()) if boxes is not None and boxes.cls is not None else 0
            if cls_idx >= len(prompts):
                cls_idx = 0
            prompt_class = prompts[min(cls_idx, len(prompts) - 1)]
            class_id = all_classes.index(prompt_class) if prompt_class in all_classes else 0

            conf = (
                float(boxes.conf[i].item()) if boxes is not None and boxes.conf is not None else 1.0
            )

            obb = mask_to_obb(mask, img_w, img_h)
            if obb is None:
                continue
            poly = mask_to_polygon(mask, img_w, img_h, polygon_epsilon)
            mb = mask_to_binary_image(mask)

            is_over, _, _ = check_mask_overlap(
                mb, existing_labels + new_labels, img_w, img_h, overlap_threshold
            )
            if is_over:
                skipped += 1
                continue
            new_labels.append((class_id, obb, poly, mb, conf))
            added += 1

        return new_labels, added, skipped, new_classes

    def segment_points(
        self,
        image_rgb,
        points,
        point_labels,
        class_id,
        existing_labels,
        polygon_epsilon=0.005,
        overlap_threshold=0.1,
    ):
        """Multi-point segmentation using positive and negative click prompts."""
        if not points:
            return None, "Add at least one positive point before running segmentation"

        results = self._predict_with_temp_image(
            image_rgb,
            points=[[int(x), int(y)] for x, y in points],
            labels=[int(label) for label in point_labels],
        )

        if not results or len(results) == 0 or results[0].masks is None:
            return None, "No object detected for the selected point prompts"
        masks_data = results[0].masks.data
        if len(masks_data) == 0:
            return None, "No object detected for the selected point prompts"

        mask = masks_data[0]
        img_h, img_w = image_rgb.shape[:2]
        return self._build_mask_label(
            mask,
            class_id,
            existing_labels,
            img_w,
            img_h,
            polygon_epsilon,
            overlap_threshold,
            f"Object detected from {len(points)} point prompt(s)",
            "No object detected for the selected point prompts",
        )

    def segment_point(
        self,
        image_rgb,
        x,
        y,
        class_id,
        existing_labels,
        polygon_epsilon=0.005,
        overlap_threshold=0.1,
    ):
        """Backwards-compatible single-point segmentation wrapper."""
        return self.segment_points(
            image_rgb,
            [(x, y)],
            [1],
            class_id,
            existing_labels,
            polygon_epsilon,
            overlap_threshold,
        )

    def segment_box(
        self,
        image_rgb,
        x1,
        y1,
        x2,
        y2,
        class_id,
        existing_labels,
        polygon_epsilon=0.005,
        overlap_threshold=0.1,
        fallback_to_box=True,
    ):
        """Box segmentation, returns (label_tuple_or_None, message)"""
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)

        img_h, img_w = image_rgb.shape[:2]
        results = self._predict_with_temp_image(
            image_rgb,
            bboxes=[[bx1, by1, bx2, by2]],
        )
        if results and len(results) > 0 and results[0].masks is not None:
            masks_data = results[0].masks.data
            if len(masks_data) > 0:
                return self._build_mask_label(
                    masks_data[0],
                    class_id,
                    existing_labels,
                    img_w,
                    img_h,
                    polygon_epsilon,
                    overlap_threshold,
                    "SAM detected object",
                    "SAM did not detect any object",
                )

        if fallback_to_box:
            if abs(bx2 - bx1) < 4 or abs(by2 - by1) < 4:
                return None, "Selection box too small"
            obb = box_to_obb(bx1, by1, bx2, by2, img_w, img_h)
            if obb is None:
                return None, "Selection box too small"
            poly = obb.copy()
            mb = polygon_to_mask(poly, img_w, img_h)
            is_over, oidx, oratio = check_mask_overlap(
                mb, existing_labels, img_w, img_h, overlap_threshold
            )
            if is_over:
                return (
                    None,
                    f"Overlaps with annotation {oidx + 1} ({oratio * 100:.0f}%)",
                )
            return (class_id, obb, poly, mb, 1.0), "Box annotation created (fallback)"

        return None, "SAM did not detect any object"
