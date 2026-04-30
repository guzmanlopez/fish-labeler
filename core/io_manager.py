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
TRACKS_FILE_NAME = "tracks.json"

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


def _load_segmentation_labels(seg_label_path, img_w, img_h):
    """Load labels from a YOLO segmentation file."""
    labels = []
    with open(seg_label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            polygon_coords = [float(x) for x in parts[1:]]
            mask = polygon_to_mask(polygon_coords, img_w, img_h)
            obb_coords = mask_to_obb(mask, img_w, img_h)
            if obb_coords:
                labels.append((class_id, obb_coords, polygon_coords, mask, None))
    return labels


def _load_obb_labels(label_path, img_w, img_h):
    """Load labels from the legacy OBB text format."""
    labels = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            mask = polygon_to_mask(coords, img_w, img_h)
            labels.append((class_id, coords, coords, mask, None))
    return labels


def load_existing_labels(label_path, seg_label_path, current_image):
    """Docstring for load_existing_labels."""
    """Load existing annotations, returns labels list"""
    img_h, img_w = current_image.shape[:2]

    if seg_label_path and seg_label_path.exists():
        return _load_segmentation_labels(seg_label_path, img_w, img_h)

    if label_path and label_path.exists():
        return _load_obb_labels(label_path, img_w, img_h)
    return []


def label_is_visible(label, classes, class_thresholds, default_threshold=0.25):
    """Return whether a label passes the current class-threshold filters."""
    if not label:
        return False
    class_id = int(label[0])
    class_name = classes[class_id] if 0 <= class_id < len(classes) else f"c{class_id}"
    score = label[4] if len(label) > 4 else None
    if score is None:
        return True
    return score >= class_thresholds.get(class_name, default_threshold)


def get_visible_labels(labels, classes, class_thresholds, default_threshold=0.25):
    """Return only the labels currently shown by the UI threshold filters."""
    return [
        label
        for label in labels
        if label_is_visible(label, classes, class_thresholds, default_threshold)
    ]


def _delete_annotation_files(output_path, img_stem):
    """Delete persisted annotation artifacts for a frame when no labels remain."""
    deleted = []
    for subfolder, extension, name in [
        ("labels", ".txt", "OBB"),
        ("labels_seg", ".txt", "Seg"),
        ("masks", ".png", "Mask"),
    ]:
        path = output_path / subfolder / f"{img_stem}{extension}"
        if path.exists():
            path.unlink()
            deleted.append(name)
    return deleted


def _copy_frame_image(output_path, current_image_path):
    """Copy the current frame image into the export folder when needed."""
    images_folder = output_path / "images"
    images_folder.mkdir(parents=True, exist_ok=True)
    dst_image_path = images_folder / current_image_path.name
    if not dst_image_path.exists() or current_image_path.resolve() != dst_image_path.resolve():
        try:
            shutil.copy2(current_image_path, dst_image_path)
        except shutil.SameFileError:
            pass


def _save_obb_labels(output_path, img_stem, labels):
    """Write OBB label text output."""
    folder = output_path / "labels"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{img_stem}.txt", "w") as f:
        for label in labels:
            coords_str = " ".join(f"{coord:.6f}" for coord in label[1])
            f.write(f"{label[0]} {coords_str}\n")


def _save_segmentation_labels(output_path, img_stem, labels):
    """Write polygon segmentation label text output."""
    folder = output_path / "labels_seg"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{img_stem}.txt", "w") as f:
        for label in labels:
            polygon = label[2] if len(label) > 2 and label[2] else label[1]
            if polygon:
                coords_str = " ".join(f"{coord:.6f}" for coord in polygon)
                f.write(f"{label[0]} {coords_str}\n")


def _save_mask_image(output_path, img_stem, image_shape, labels):
    """Write the indexed mask export for the visible labels."""
    folder = output_path / "masks"
    folder.mkdir(parents=True, exist_ok=True)
    img_h, img_w = image_shape[:2]
    combined = np.zeros((img_h, img_w), dtype=np.uint8)
    for index, label in enumerate(labels):
        mask_binary = label[3] if len(label) > 3 and label[3] is not None else None
        if mask_binary is None:
            continue
        if mask_binary.shape != (img_h, img_w):
            mask_binary = cv2.resize(mask_binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        combined[mask_binary > 0] = index + 1
    cv2.imwrite(str(folder / f"{img_stem}.png"), combined)


def _write_classes_file(output_path, classes):
    """Persist the current class list alongside saved labels."""
    classes_file = output_path / "classes.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    persist_classes(classes)


def auto_save_labels(state):
    """Docstring for auto_save_labels."""
    """Auto-save annotations with multi-format output. Preserves original logic."""
    if state.current_image is None or state.current_image_path is None:
        return None

    output_path = state.output_folder
    img_stem = state.current_image_path.stem
    visible_labels = get_visible_labels(
        state.current_labels,
        state.classes,
        state.class_thresholds,
    )

    if not visible_labels:
        deleted = _delete_annotation_files(output_path, img_stem)
        return f"Deleted ({', '.join(deleted)})" if deleted else None

    _copy_frame_image(output_path, state.current_image_path)

    saved = []

    if state.output_formats.get("obb", False):
        _save_obb_labels(output_path, img_stem, visible_labels)
        saved.append("OBB")

    if state.output_formats.get("seg", True):
        _save_segmentation_labels(output_path, img_stem, visible_labels)
        saved.append("Seg")

    if state.output_formats.get("mask", False):
        _save_mask_image(output_path, img_stem, state.current_image.shape, visible_labels)
        saved.append("Mask")

    _write_classes_file(output_path, state.classes)

    return f"Saved {len(visible_labels)} annotations ({'+'.join(saved)})"


def load_tracking_data(output_folder):
    """Load persisted tracking assignments and tracker configuration."""
    tracks_file = Path(output_folder) / TRACKS_FILE_NAME
    if not tracks_file.exists():
        return {"frame_track_ids": {}, "tracks": {}, "config": {}}
    try:
        with open(tracks_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tracks = {
            int(track_id): track_data for track_id, track_data in payload.get("tracks", {}).items()
        }
        return {
            "frame_track_ids": payload.get("frame_track_ids", {}),
            "tracks": tracks,
            "config": payload.get("config", {}),
        }
    except Exception as e:
        print(f"Failed to load tracking data: {e}")
        return {"frame_track_ids": {}, "tracks": {}, "config": {}}


def save_tracking_data(output_folder, frame_track_ids, tracks, config):
    """Persist tracking assignments and summary metadata next to the labels."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    tracks_file = output_path / TRACKS_FILE_NAME
    serializable_tracks = {str(track_id): track for track_id, track in tracks.items()}
    payload = {
        "frame_track_ids": frame_track_ids,
        "tracks": serializable_tracks,
        "config": config,
    }
    try:
        with open(tracks_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save tracking data: {e}")
