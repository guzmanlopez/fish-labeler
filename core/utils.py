"""
Utility functions — preserves all original processing logic
mask_to_obb, mask_to_polygon, mask_to_binary_image, polygon_to_mask,
check_mask_overlap, box_to_obb, point_in_obb, find_clicked_label,
obb_intersects_box, find_labels_in_box, create_coco_annotation, create_coco_dataset
"""

import json

import cv2
import numpy as np

# ------------------------------------------------------------------
# Mask to coordinate conversion
# ------------------------------------------------------------------


def mask_to_obb(mask, img_width, img_height):
    """Convert binary mask to OBB coordinates"""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return None
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    normalized_box = []
    for point in box:
        x_norm = max(0, min(1, point[0] / img_width))
        y_norm = max(0, min(1, point[1] / img_height))
        normalized_box.extend([x_norm, y_norm])
    return normalized_box


def mask_to_polygon(mask, img_width, img_height, epsilon_ratio=0.005):
    """Convert binary mask to polygon coordinates (YOLO-Seg format)"""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[0]
    mask = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:
        return None
    epsilon = epsilon_ratio * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    if len(approx) < 3:
        return None
    normalized_polygon = []
    for point in approx:
        x_norm = max(0, min(1, point[0][0] / img_width))
        y_norm = max(0, min(1, point[0][1] / img_height))
        normalized_polygon.extend([x_norm, y_norm])
    return normalized_polygon


def mask_to_binary_image(mask):
    """Convert mask to binary image (0 or 255)"""
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    mask = mask.astype(np.uint8)
    if len(mask.shape) == 3:
        mask = mask[0]
    return (mask > 0).astype(np.uint8) * 255


def polygon_to_mask(polygon_coords, img_width, img_height):
    """Convert polygon coordinates back to mask image"""
    points = []
    for i in range(0, len(polygon_coords), 2):
        x = int(polygon_coords[i] * img_width)
        y = int(polygon_coords[i + 1] * img_height)
        points.append([x, y])
    points = np.array(points, dtype=np.int32)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


# ------------------------------------------------------------------
# Overlap detection
# ------------------------------------------------------------------


def check_mask_overlap(
    new_mask, existing_labels, img_width, img_height, threshold=None
):
    """Check if new mask overlaps with existing annotations"""
    if new_mask is None:
        return False, None, 0
    if threshold is not None and threshold <= 0:
        return False, None, 0
    if hasattr(new_mask, "cpu"):
        new_mask = new_mask.cpu().numpy()
    new_mask = new_mask.astype(np.uint8)
    if len(new_mask.shape) == 3:
        new_mask = new_mask[0]
    if new_mask.shape != (img_height, img_width):
        new_mask = cv2.resize(
            new_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
        )
    new_mask_binary = new_mask > 0
    new_mask_area = new_mask_binary.sum()
    if new_mask_area == 0:
        return False, None, 0

    for idx, label in enumerate(existing_labels):
        existing_mask = label[3] if len(label) > 3 and label[3] is not None else None
        if existing_mask is None:
            polygon_coords = label[2] if len(label) > 2 and label[2] else label[1]
            if polygon_coords:
                existing_mask = polygon_to_mask(polygon_coords, img_width, img_height)
        if existing_mask is None:
            continue
        if existing_mask.shape != (img_height, img_width):
            existing_mask = cv2.resize(
                existing_mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST
            )
        existing_mask_binary = existing_mask > 0
        intersection = np.logical_and(new_mask_binary, existing_mask_binary)
        intersection_area = intersection.sum()
        if intersection_area > 0:
            overlap_ratio = intersection_area / new_mask_area
            if threshold is not None and overlap_ratio > threshold:
                return True, idx, overlap_ratio
    return False, None, 0


# ------------------------------------------------------------------
# OBB geometry
# ------------------------------------------------------------------


def box_to_obb(x1, y1, x2, y2, img_w, img_h):
    """Convert bounding box to axis-aligned OBB coordinates"""
    if img_w <= 0 or img_h <= 0:
        return None
    box_x1 = max(0, min(img_w, min(x1, x2)))
    box_x2 = max(0, min(img_w, max(x1, x2)))
    box_y1 = max(0, min(img_h, min(y1, y2)))
    box_y2 = max(0, min(img_h, max(y1, y2)))
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        return None
    points = [
        (box_x1, box_y1),
        (box_x2, box_y1),
        (box_x2, box_y2),
        (box_x1, box_y2),
    ]
    normalized_box = []
    for px, py in points:
        normalized_box.extend(
            [
                max(0, min(1, px / img_w)),
                max(0, min(1, py / img_h)),
            ]
        )
    return normalized_box


def point_in_obb(x, y, obb_coords, img_w, img_h):
    """Check if point (x, y) is inside OBB"""
    points = []
    for i in range(0, 8, 2):
        px = int(obb_coords[i] * img_w)
        py = int(obb_coords[i + 1] * img_h)
        points.append([px, py])
    polygon = np.array(points, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
    return result >= 0


def find_clicked_label(x, y, labels, img_w, img_h):
    """Find annotation index at click position"""
    clicked_indices = []
    for idx, label in enumerate(labels):
        coords = label[1]
        if point_in_obb(x, y, coords, img_w, img_h):
            clicked_indices.append(idx)
    if clicked_indices:
        return clicked_indices[0]
    return None


def obb_intersects_box(obb_coords, box_x1, box_y1, box_x2, box_y2, img_w, img_h):
    """Check if OBB intersects or is contained within bounding box"""
    obb_points = []
    for i in range(0, 8, 2):
        px = obb_coords[i] * img_w
        py = obb_coords[i + 1] * img_h
        obb_points.append((px, py))
    center_x = sum(p[0] for p in obb_points) / 4
    center_y = sum(p[1] for p in obb_points) / 4
    if box_x1 <= center_x <= box_x2 and box_y1 <= center_y <= box_y2:
        return True
    for px, py in obb_points:
        if box_x1 <= px <= box_x2 and box_y1 <= py <= box_y2:
            return True
    return False


def find_labels_in_box(x1, y1, x2, y2, labels, img_w, img_h):
    """Find all annotation indices within box selection"""
    box_x1, box_x2 = min(x1, x2), max(x1, x2)
    box_y1, box_y2 = min(y1, y2), max(y1, y2)
    found_indices = []
    for idx, label in enumerate(labels):
        coords = label[1]
        if obb_intersects_box(coords, box_x1, box_y1, box_x2, box_y2, img_w, img_h):
            found_indices.append(idx)
    return found_indices


# ------------------------------------------------------------------
# COCO format
# ------------------------------------------------------------------


def create_coco_annotation(
    ann_id, image_id, category_id, polygon_coords, img_width, img_height
):
    """Create a single COCO annotation object"""
    segmentation = []
    for i in range(0, len(polygon_coords), 2):
        x = polygon_coords[i] * img_width
        y = polygon_coords[i + 1] * img_height
        segmentation.extend([x, y])
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    width = x_max - x_min
    height = y_max - y_min
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    area = abs(area) / 2.0
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [segmentation],
        "area": area,
        "bbox": [x_min, y_min, width, height],
        "iscrowd": 0,
    }


def create_coco_dataset(image_list, labels_dict, classes, output_path):
    """Create complete COCO dataset JSON"""
    coco_data = {
        "info": {"description": "SAM3 Labeler Dataset", "version": "1.0", "year": 2026},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    for idx, class_name in enumerate(classes):
        coco_data["categories"].append(
            {"id": idx, "name": class_name, "supercategory": "object"}
        )

    ann_id = 1
    for img_id, img_path in enumerate(image_list):
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        coco_data["images"].append(
            {"id": img_id, "file_name": img_name, "width": img_w, "height": img_h}
        )
        if img_name in labels_dict:
            for class_id, polygon_coords in labels_dict[img_name]:
                if polygon_coords and len(polygon_coords) >= 6:
                    ann = create_coco_annotation(
                        ann_id, img_id, class_id, polygon_coords, img_w, img_h
                    )
                    coco_data["annotations"].append(ann)
                    ann_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
