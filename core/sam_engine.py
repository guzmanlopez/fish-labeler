"""
SAM 3 model loading and inference
Preserves original load_sam3_model / load_sam_model logic
"""

import os

import cv2

from core.utils import (
    box_to_obb,
    check_mask_overlap,
    mask_to_binary_image,
    mask_to_obb,
    mask_to_polygon,
    polygon_to_mask,
)


class SAMEngine:
    """Wrapper for SAM 3 model operations"""

    def __init__(self, model_path="sam3.pt", device="cuda:0"):
        self.model_path = model_path
        self.device = device
        self._predictor = None  # SAM3SemanticPredictor (for text prompts)
        self._sam_model = None  # SAM model (for click/box)

    def _ensure_predictor(self):
        if self._predictor is None:
            from ultralytics.models.sam import SAM3SemanticPredictor

            self._predictor = SAM3SemanticPredictor(
                overrides=dict(
                    conf=0.5,
                    model=self.model_path,
                    device=self.device,
                    half=True,
                    verbose=False,
                )
            )
            print(f"[OK] SAM 3 semantic model loaded ({self.device})")
        return self._predictor

    def _ensure_sam(self):
        if self._sam_model is None:
            from ultralytics import SAM

            self._sam_model = SAM(self.model_path)
            print(f"[OK] SAM click/box model loaded ({self.device})")
        return self._sam_model

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
            cls_idx = (
                int(boxes.cls[i].item())
                if boxes is not None and boxes.cls is not None
                else 0
            )
            if cls_idx >= len(prompts):
                cls_idx = 0
            prompt_class = prompts[min(cls_idx, len(prompts) - 1)]
            class_id = (
                all_classes.index(prompt_class) if prompt_class in all_classes else 0
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
            new_labels.append((class_id, obb, poly, mb))
            added += 1

        return new_labels, added, skipped, new_classes

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
        """Click segmentation, returns (label_tuple_or_None, message)"""
        sam = self._ensure_sam()
        temp_path = "_temp_sam_img.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        try:
            results = sam.predict(
                source=temp_path, points=[[x, y]], labels=[1], device=self.device
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not results or len(results) == 0 or results[0].masks is None:
            return None, f"No object detected at ({x}, {y})"
        masks_data = results[0].masks.data
        if len(masks_data) == 0:
            return None, f"No object detected at ({x}, {y})"

        mask = masks_data[0]
        img_h, img_w = image_rgb.shape[:2]
        obb = mask_to_obb(mask, img_w, img_h)
        if obb is None:
            return None, f"No object detected at ({x}, {y})"

        poly = mask_to_polygon(mask, img_w, img_h, polygon_epsilon)
        mb = mask_to_binary_image(mask)
        is_over, oidx, oratio = check_mask_overlap(
            mb, existing_labels, img_w, img_h, overlap_threshold
        )
        if is_over:
            return None, f"Overlaps with annotation {oidx + 1} ({oratio * 100:.0f}%)"

        return (class_id, obb, poly, mb), f"Object detected at ({x}, {y})"

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

        sam = self._ensure_sam()
        temp_path = "_temp_sam_img.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        try:
            results = sam.predict(
                source=temp_path, bboxes=[[bx1, by1, bx2, by2]], device=self.device
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        img_h, img_w = image_rgb.shape[:2]

        if results and len(results) > 0 and results[0].masks is not None:
            masks_data = results[0].masks.data
            if len(masks_data) > 0:
                mask = masks_data[0]
                obb = mask_to_obb(mask, img_w, img_h)
                if obb is not None:
                    poly = mask_to_polygon(mask, img_w, img_h, polygon_epsilon)
                    mb = mask_to_binary_image(mask)
                    is_over, oidx, oratio = check_mask_overlap(
                        mb, existing_labels, img_w, img_h, overlap_threshold
                    )
                    if is_over:
                        return (
                            None,
                            f"Overlaps with annotation {oidx + 1} ({oratio * 100:.0f}%)",
                        )
                    return (class_id, obb, poly, mb), "SAM detected object"

        # fallback to box
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
            return (class_id, obb, poly, mb), "Box annotation created (fallback)"

        return None, "SAM did not detect any object"
