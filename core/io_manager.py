"""
File I/O management — config, progress, label persistence
Preserves all original logic
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .utils import mask_to_obb, polygon_to_mask

CLASSES_STORE = Path(__file__).resolve().parent.parent / "sam3_classes.txt"
PROGRESS_FILE = Path(__file__).resolve().parent.parent / "sam3_progress.json"
CONFIG_FILE = Path(__file__).resolve().parent.parent / "sam3_config.json"

DEFAULT_CONFIG = {
    "images_folder": "./sample_images",
    "output_folder": "./output",
}


def load_config():
    """Docstring for load_config."""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        return DEFAULT_CONFIG.copy()
    except Exception as e:
        print(f"Failed to load config: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(images_folder=None, output_folder=None):
    """Docstring for save_config."""
    try:
        config = load_config()
        if images_folder:
            config["images_folder"] = images_folder
        if output_folder:
            config["output_folder"] = output_folder
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save config: {e}")


def save_progress(folder_path, index, image_list):
    """Docstring for save_progress."""
    try:
        progress = {}
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                progress = json.load(f)
        folder_key = str(Path(folder_path).resolve())
        progress[folder_key] = {
            "last_index": index,
            "last_image": image_list[index].name if image_list else "",
        }
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save progress: {e}")


def load_progress(folder_path):
    """Docstring for load_progress."""
    try:
        if not PROGRESS_FILE.exists():
            return 0
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        folder_key = str(Path(folder_path).resolve())
        if folder_key in progress:
            return progress[folder_key].get("last_index", 0)
        return 0
    except Exception as e:
        print(f"Failed to load progress: {e}")
        return 0


def persist_classes(classes):
    """Docstring for persist_classes."""
    try:
        with open(CLASSES_STORE, "w", encoding="utf-8") as f:
            for c in classes:
                f.write(f"{c}\n")
    except Exception as e:
        print(f"Failed to write classes file: {e}")


def load_persisted_classes():
    """Docstring for load_persisted_classes."""
    try:
        with open(CLASSES_STORE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            return lines if lines else ["fish"]
    except FileNotFoundError:
        return ["fish"]
    except Exception as e:
        print(f"Failed to read classes file: {e}")
        return ["fish"]


def load_existing_labels(label_path, seg_label_path, current_image):
    """Docstring for load_existing_labels."""
    """Load existing annotations, returns labels list"""
    labels = []
    img_h, img_w = current_image.shape[:2]

    # Prefer YOLO-Seg format
    if seg_label_path and seg_label_path.exists():
        with open(seg_label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    class_id = int(parts[0])
                    polygon_coords = [float(x) for x in parts[1:]]
                    mask = polygon_to_mask(polygon_coords, img_w, img_h)
                    obb_coords = mask_to_obb(mask, img_w, img_h)
                    if obb_coords:
                        labels.append(
                            (class_id, obb_coords, polygon_coords, mask, None)
                        )
        return labels

    # OBB format (backward compatible)
    if label_path and label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 9:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    mask = polygon_to_mask(coords, img_w, img_h)
                    labels.append((class_id, coords, coords, mask, None))
    return labels


def auto_save_labels(state):
    """Docstring for auto_save_labels."""
    """Auto-save annotations with multi-format output. Preserves original logic."""
    if state.current_image is None or state.current_image_path is None:
        return None

    output_path = state.output_folder
    img_stem = state.current_image_path.stem

    # No annotations -> delete files
    if not state.current_labels:
        deleted = []
        for sub, ext, name in [
            ("labels", ".txt", "OBB"),
            ("labels_seg", ".txt", "Seg"),
            ("masks", ".png", "Mask"),
        ]:
            p = output_path / sub / f"{img_stem}{ext}"
            if p.exists():
                p.unlink()
                deleted.append(name)
        return f"Deleted ({', '.join(deleted)})" if deleted else None

    # Has annotations -> save
    images_folder = output_path / "images"
    images_folder.mkdir(parents=True, exist_ok=True)
    dst_image_path = images_folder / state.current_image_path.name
    if (
        not dst_image_path.exists()
        or state.current_image_path.resolve() != dst_image_path.resolve()
    ):
        try:
            shutil.copy2(state.current_image_path, dst_image_path)
        except shutil.SameFileError:
            pass

    saved = []

    # OBB
    if state.output_formats.get("obb", False):
        d = output_path / "labels"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{img_stem}.txt", "w") as f:
            for label in state.current_labels:
                coords_str = " ".join(f"{c:.6f}" for c in label[1])
                f.write(f"{label[0]} {coords_str}\n")
        saved.append("OBB")

    # Seg
    if state.output_formats.get("seg", True):
        d = output_path / "labels_seg"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{img_stem}.txt", "w") as f:
            for label in state.current_labels:
                poly = label[2] if len(label) > 2 and label[2] else label[1]
                if poly:
                    coords_str = " ".join(f"{c:.6f}" for c in poly)
                    f.write(f"{label[0]} {coords_str}\n")
        saved.append("Seg")

    # Mask
    if state.output_formats.get("mask", False):
        d = output_path / "masks"
        d.mkdir(parents=True, exist_ok=True)
        img_h, img_w = state.current_image.shape[:2]
        combined = np.zeros((img_h, img_w), dtype=np.uint8)
        for idx, label in enumerate(state.current_labels):
            mb = label[3] if len(label) > 3 and label[3] is not None else None
            if mb is not None:
                if mb.shape != (img_h, img_w):
                    mb = cv2.resize(mb, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                combined[mb > 0] = idx + 1
        cv2.imwrite(str(d / f"{img_stem}.png"), combined)
        saved.append("Mask")

    # classes.txt
    classes_file = output_path / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for c in state.classes:
            f.write(f"{c}\n")
    persist_classes(state.classes)

    return f"Saved {len(state.current_labels)} annotations ({'+'.join(saved)})"
