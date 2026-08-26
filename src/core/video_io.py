"""Video frame extraction and YOLO segmentation export helpers."""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata for an opened video."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float


class VideoHandler:
    """Read metadata and individual frames with OpenCV."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info = VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            total_frames=total_frames,
            duration_seconds=total_frames / fps if fps > 0 else 0.0,
        )

    def save_frame(self, frame_index: int, destination: str):
        """Save one frame and return its destination, or None on failure."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = self._capture.read()
        if not success or not cv2.imwrite(destination, frame):
            return None
        return destination

    def release(self):
        """Release the underlying video capture."""
        self._capture.release()


class YOLOExporter:
    """Write sampled annotations as a YOLO segmentation dataset."""

    def __init__(self, output_dir: str, labels: dict[int, str]):
        self.output_dir = Path(output_dir)
        self.labels = labels

    def export(
        self,
        video_handler,
        annotations,
        include_detection=False,
        include_segmentation=True,
    ):
        """Write labels, JSON annotations, metadata, and dataset configuration."""
        labels_dir = self.output_dir / "labels"
        annotations_dir = self.output_dir / "annotations_json"
        labels_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        for frame_index, annotation in annotations.items():
            stem = f"frame_{frame_index:06d}"
            lines = []
            if include_segmentation:
                for detection in annotation.get("detections", []):
                    polygon = detection.get("mask_polygon")
                    if not polygon:
                        continue
                    coordinates = " ".join(f"{value:.6f}" for point in polygon for value in point)
                    lines.append(f"{detection['class_id']} {coordinates}")
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            (annotations_dir / f"{stem}.json").write_text(json.dumps(annotation, indent=2))

        names = "\n".join(f"  {index}: {json.dumps(name)}" for index, name in self.labels.items())
        (self.output_dir / "data.yaml").write_text(
            f"path: .\ntrain: images\nval: images\nnames:\n{names}\n"
        )
        (self.output_dir / "classes.txt").write_text("\n".join(self.labels.values()) + "\n")
        metadata = {
            "video_path": video_handler.video_path,
            "total_annotated_frames": len(annotations),
            "labels": self.labels,
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        video_handler.release()
