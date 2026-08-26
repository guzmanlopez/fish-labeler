"""Sample video frames with SAM3.1 and export YOLO segmentation labels.

This script is intended to be run from the terminal. It samples frames from a
video, runs SAM3.1 semantic segmentation from text prompts on each sampled
frame, and writes a YOLO-seg style dataset layout:

    <output_dir>/
        images/frame_000000.jpg
        labels/frame_000000.txt
        annotations_json/frame_000000.json
        data.yaml
        metadata.json
        run_config.json

Example:
    fish-labeler video \
        --video /path/to/video.mp4 \
        --output-dir vessel-trip-01 \
        --classes fish tuna shark \
        --conf 0.5 \
        --imgsz 1024 \
        --frame-step 12
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from tqdm.auto import tqdm

from core.io_manager import resolve_output_folder
from core.sam_engine import DEFAULT_MODEL_PATH
from core.video_io import VideoHandler, YOLOExporter

LOG = logging.getLogger(__name__)
CONSOLE = Console()


def configure_logging(verbose: bool) -> None:
    """Docstring for configure_logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[RichHandler(console=CONSOLE, show_path=False, rich_tracebacks=True)],
        force=True,
    )


def parse_prompt_classes(values: Sequence[str]) -> list[str]:
    """Docstring for parse_prompt_classes."""
    prompts: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        prompts.extend(part for part in parts if part)
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        if prompt not in seen:
            deduped.append(prompt)
            seen.add(prompt)
    return deduped


def str2bool(value: str) -> bool:
    """Docstring for str2bool."""
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Docstring for parse_args."""
    parser = argparse.ArgumentParser(
        description="Run SAM3.1 text-prompt segmentation on sampled video frames and export YOLO-seg labels."
    )
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run name below the repository output directory.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=False,
        default=[
            "fish",
            "swordfish",
            "tuna",
            "shark",
            "stingray",
            "sea turtle",
            "person",
        ],
        help="Text prompts to segment. You can pass repeated values or comma-separated values.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the SAM3.1 model weights.",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=728,
        help="Inference image size passed to Ultralytics.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=12,
        help="Process every N frames. For a 12 FPS video, use 12 to process one frame per second.",
    )
    parser.add_argument(
        "--sample-every-seconds",
        type=float,
        default=None,
        help="Alternative to --frame-step. If set, computes frame_step as round(fps * seconds).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device string, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--half",
        type=str2bool,
        default=True,
        help="Enable FP16 if supported. Defaults to automatic selection.",
    )
    parser.add_argument(
        "--overwrite-frames",
        action="store_true",
        help="Rewrite already exported frame images if they exist.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of sampled frames to process.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args(argv)
    args.classes = parse_prompt_classes(args.classes)
    if not args.classes:
        parser.error("At least one class prompt is required via --classes.")
    if args.frame_step < 1:
        parser.error("--frame-step must be at least 1.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1 when provided.")
    if args.sample_every_seconds is not None and args.sample_every_seconds <= 0:
        parser.error("--sample-every-seconds must be > 0 when provided.")
    return args


def resolve_half_precision(device: str | None, requested_half: bool | None) -> bool:
    """Docstring for resolve_half_precision."""
    if requested_half is not None:
        if requested_half and device == "cpu":
            LOG.warning("FP16 requested on CPU; disabling half precision.")
            return False
        return requested_half

    if device == "cpu":
        return False

    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def build_predictor(model_path: Path, args: argparse.Namespace):
    """Build the SAM predictor configured for the requested video export."""
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = {
        "conf": args.conf,
        "iou": args.iou,
        "task": "segment",
        "mode": "predict",
        "model": str(model_path),
        "imgsz": args.imgsz,
        "half": resolve_half_precision(args.device, args.half),
        "save": False,
        "verbose": False,
        "batch": 1,
    }
    if args.device:
        overrides["device"] = args.device
    return SAM3SemanticPredictor(overrides=overrides)


def compute_frame_step(fps: float, frame_step: int, sample_every_seconds: float | None) -> int:
    """Return the frame interval from an explicit interval or elapsed seconds."""
    if sample_every_seconds is None:
        return frame_step
    computed = max(1, round(fps * sample_every_seconds))
    LOG.info(
        "Using time-based sampling: fps=%.3f, sample_every_seconds=%.3f, frame_step=%d",
        fps,
        sample_every_seconds,
        computed,
    )
    return computed


def frame_indices_for_video(
    total_frames: int, frame_step: int, max_frames: int | None
) -> list[int]:
    """Return sampled frame indices, optionally capped to a maximum count."""
    indices = list(range(0, total_frames, frame_step))
    if max_frames is not None:
        return indices[:max_frames]
    return indices


def ensure_frame_image(
    video_handler: VideoHandler,
    frame_idx: int,
    images_dir: Path,
    overwrite: bool,
) -> Path:
    """Extract and return a sampled frame image, reusing it when allowed."""
    images_dir.mkdir(parents=True, exist_ok=True)
    frame_path = images_dir / f"frame_{frame_idx:06d}.jpg"
    if overwrite or not frame_path.exists():
        saved = video_handler.save_frame(frame_idx, str(frame_path))
        if saved is None:
            raise RuntimeError(
                f"Failed to extract frame {frame_idx} from {video_handler.video_path}"
            )
    return frame_path


def polygon_from_mask(
    mask: np.ndarray, image_width: int, image_height: int
) -> list[list[float]] | None:
    """Convert the largest mask contour to normalized polygon coordinates."""
    mask_uint8 = (mask.astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = contours[0]
    contour_area = cv2.contourArea(contour)
    for candidate in contours[1:]:
        candidate_area = cv2.contourArea(candidate)
        if candidate_area > contour_area:
            contour = candidate
            contour_area = candidate_area
    if len(contour) < 3:
        return None
    points = contour.reshape(-1, 2)
    return [[float(x / image_width), float(y / image_height)] for x, y in points]


# complexipy: ignore
def result_to_annotation(result, frame_idx: int, class_prompts: Sequence[str]) -> dict:
    """Docstring for result_to_annotation."""
    image_height, image_width = result.orig_shape
    annotation = {
        "frame_idx": frame_idx,
        "image_width": image_width,
        "image_height": image_height,
        "detections": [],
    }

    if result.boxes is None or len(result.boxes) == 0:
        return annotation

    boxes = result.boxes
    masks = result.masks

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item()) if boxes.cls is not None else 0
        class_name = (
            class_prompts[class_id] if class_id < len(class_prompts) else f"class_{class_id}"
        )
        xyxy = boxes.xyxy[index].cpu().numpy().tolist()
        confidence = float(boxes.conf[index].item()) if boxes.conf is not None else 1.0
        cx = ((xyxy[0] + xyxy[2]) / 2) / image_width
        cy = ((xyxy[1] + xyxy[3]) / 2) / image_height
        bw = (xyxy[2] - xyxy[0]) / image_width
        bh = (xyxy[3] - xyxy[1]) / image_height

        mask_polygon = None
        if masks is not None and getattr(masks, "xy", None) is not None and index < len(masks.xy):
            polygon = masks.xy[index]
            if len(polygon) >= 3:
                mask_polygon = [
                    [float(x / image_width), float(y / image_height)] for x, y in polygon
                ]
        if mask_polygon is None and masks is not None and getattr(masks, "data", None) is not None:
            mask_array = masks.data[index].detach().cpu().numpy() > 0.5
            mask_polygon = polygon_from_mask(mask_array, image_width, image_height)

        annotation["detections"].append({
            "class_id": class_id,
            "class_name": class_name,
            "bbox_xyxy": [float(value) for value in xyxy],
            "bbox_xywhn": [float(cx), float(cy), float(bw), float(bh)],
            "confidence": confidence,
            "mask_polygon": mask_polygon,
            "track_id": None,
        })

    return annotation


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    fps: float,
    total_frames: int,
    sampled_frames: int,
    frame_step: int,
    total_detections: int,
) -> None:
    """Write the inference settings and export totals for a completed run."""
    run_config = {
        "video": str(args.video),
        "model": str(args.model),
        "classes": args.classes,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "device": args.device,
        "half": resolve_half_precision(args.device, args.half),
        "fps": fps,
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "frame_step": frame_step,
        "sample_every_seconds": args.sample_every_seconds,
        "total_detections": total_detections,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))


# complexipy: ignore
def enrich_metadata_file(
    output_dir: Path,
    video_path: Path,
    video_info,
    frame_step: int,
    sampled_indices: Sequence[int],
    sample_every_seconds: float | None,
    frames_with_detections: int,
    total_detections: int,
    annotations: dict[int, dict],
    class_prompts: Sequence[str],
) -> dict:
    """Update and persist video, sampling, and detection summary metadata."""
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {}

    file_size_bytes = video_path.stat().st_size
    sampled_count = len(sampled_indices)
    duration_seconds = float(video_info.duration_seconds)
    effective_sample_interval_seconds = frame_step / video_info.fps if video_info.fps > 0 else None
    aspect_ratio = (video_info.width / video_info.height) if video_info.height > 0 else None
    class_counts = {prompt: 0 for prompt in class_prompts}
    class_frame_counts = {prompt: 0 for prompt in class_prompts}
    confidence_values: list[float] = []

    for annotation in annotations.values():
        frame_classes: set[str] = set()
        for det in annotation.get("detections", []):
            class_name = str(det.get("class_name", "unknown"))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            frame_classes.add(class_name)
            confidence = det.get("confidence")
            if isinstance(confidence, int | float):
                confidence_values.append(float(confidence))
        for class_name in frame_classes:
            class_frame_counts[class_name] = class_frame_counts.get(class_name, 0) + 1

    detected_class_names = sorted([name for name, count in class_counts.items() if count > 0])

    metadata["video"] = {
        "path": str(video_path),
        "filename": video_path.name,
        "stem": video_path.stem,
        "extension": video_path.suffix,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
        "total_frames": video_info.total_frames,
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "resolution": f"{video_info.width}x{video_info.height}",
        "aspect_ratio": round(aspect_ratio, 6) if aspect_ratio is not None else None,
        "duration_seconds": round(duration_seconds, 3),
        "duration_minutes": round(duration_seconds / 60, 3),
    }
    metadata["sampling"] = {
        "frame_step": frame_step,
        "requested_sample_every_seconds": sample_every_seconds,
        "effective_sample_interval_seconds": (
            round(effective_sample_interval_seconds, 6)
            if effective_sample_interval_seconds is not None
            else None
        ),
        "sampled_frame_count": sampled_count,
        "first_sampled_frame": sampled_indices[0] if sampled_indices else None,
        "last_sampled_frame": sampled_indices[-1] if sampled_indices else None,
    }
    metadata["export_summary"] = {
        "frames_with_detections": frames_with_detections,
        "frames_without_detections": sampled_count - frames_with_detections,
        "total_detections": total_detections,
        "avg_detections_per_sampled_frame": round(total_detections / sampled_count, 6)
        if sampled_count > 0
        else 0.0,
        "avg_detections_per_detected_frame": round(total_detections / frames_with_detections, 6)
        if frames_with_detections > 0
        else 0.0,
    }
    metadata["detection_summary"] = {
        "num_requested_classes": len(class_prompts),
        "num_detected_classes": len(detected_class_names),
        "requested_classes": list(class_prompts),
        "detected_classes": detected_class_names,
        "undetected_classes": [name for name in class_prompts if class_counts.get(name, 0) == 0],
        "detections_per_class": class_counts,
        "frames_with_class": class_frame_counts,
        "class_detection_share": {
            name: round(count / total_detections, 6) if total_detections > 0 else 0.0
            for name, count in class_counts.items()
        },
        "class_frame_coverage": {
            name: round(count / sampled_count, 6) if sampled_count > 0 else 0.0
            for name, count in class_frame_counts.items()
        },
        "confidence": {
            "min": round(min(confidence_values), 6) if confidence_values else None,
            "max": round(max(confidence_values), 6) if confidence_values else None,
            "mean": round(sum(confidence_values) / len(confidence_values), 6)
            if confidence_values
            else None,
        },
        "detections_per_detected_class": round(total_detections / len(detected_class_names), 6)
        if detected_class_names
        else 0.0,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def log_export_metadata(metadata: dict) -> None:
    """Docstring for log_export_metadata."""
    video = metadata.get("video", {})
    sampling = metadata.get("sampling", {})
    export_summary = metadata.get("export_summary", {})
    detection_summary = metadata.get("detection_summary", {})

    LOG.info(
        "Video details: %s | %s | %.3f FPS | %d frames | %.3f s | %.3f MB",
        video.get("filename", "unknown"),
        video.get("resolution", "unknown"),
        float(video.get("fps", 0.0) or 0.0),
        int(video.get("total_frames", 0) or 0),
        float(video.get("duration_seconds", 0.0) or 0.0),
        float(video.get("file_size_mb", 0.0) or 0.0),
    )
    LOG.info(
        "Sampling details: step=%s | requested_seconds=%s | effective_seconds=%s | sampled_frames=%s | first=%s | last=%s",
        sampling.get("frame_step"),
        sampling.get("requested_sample_every_seconds"),
        sampling.get("effective_sample_interval_seconds"),
        sampling.get("sampled_frame_count"),
        sampling.get("first_sampled_frame"),
        sampling.get("last_sampled_frame"),
    )
    LOG.info(
        "Detection summary: %s detections across %s/%s classes | %s frames with detections | avg/sample=%.3f | avg/detected-frame=%.3f",
        export_summary.get("total_detections", 0),
        detection_summary.get("num_detected_classes", 0),
        detection_summary.get("num_requested_classes", 0),
        export_summary.get("frames_with_detections", 0),
        float(export_summary.get("avg_detections_per_sampled_frame", 0.0) or 0.0),
        float(export_summary.get("avg_detections_per_detected_frame", 0.0) or 0.0),
    )

    confidence = detection_summary.get("confidence", {})
    if confidence:
        LOG.info(
            "Confidence stats: min=%s | mean=%s | max=%s",
            confidence.get("min"),
            confidence.get("mean"),
            confidence.get("max"),
        )

    detected_classes = detection_summary.get("detected_classes", [])
    undetected_classes = detection_summary.get("undetected_classes", [])
    LOG.info(
        "Detected classes: %s",
        ", ".join(detected_classes) if detected_classes else "none",
    )
    if undetected_classes:
        LOG.info("Undetected classes: %s", ", ".join(undetected_classes))

    detections_per_class = detection_summary.get("detections_per_class", {})
    frames_with_class = detection_summary.get("frames_with_class", {})
    class_frame_coverage = detection_summary.get("class_frame_coverage", {})
    for class_name, count in sorted(
        detections_per_class.items(), key=lambda item: (-item[1], item[0])
    ):
        LOG.info(
            "Class '%s': detections=%s | frames=%s | frame_coverage=%s",
            class_name,
            count,
            frames_with_class.get(class_name, 0),
            class_frame_coverage.get(class_name, 0.0),
        )


def log_initial_video_metadata(video_path: Path, video_info) -> None:
    """Docstring for log_initial_video_metadata."""
    file_size_bytes = video_path.stat().st_size
    duration_seconds = float(video_info.duration_seconds)
    aspect_ratio = (video_info.width / video_info.height) if video_info.height > 0 else None

    LOG.info("Starting SAM3.1 video export")
    LOG.info("Video path: %s", video_path)
    LOG.info(
        "Video metadata: filename=%s | resolution=%dx%d | fps=%.3f | total_frames=%d",
        video_path.name,
        video_info.width,
        video_info.height,
        video_info.fps,
        video_info.total_frames,
    )
    LOG.info(
        "Video metadata: duration_seconds=%.3f | duration_minutes=%.3f | file_size_mb=%.3f | aspect_ratio=%s",
        duration_seconds,
        duration_seconds / 60,
        file_size_bytes / (1024 * 1024),
        round(aspect_ratio, 6) if aspect_ratio is not None else None,
    )


def run(args: argparse.Namespace) -> Path:
    """Docstring for run."""
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    output_dir = resolve_output_folder(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_handler = VideoHandler(str(args.video.resolve()))
    video_info = video_handler.info
    log_initial_video_metadata(args.video.resolve(), video_info)
    frame_step = compute_frame_step(
        fps=video_info.fps,
        frame_step=args.frame_step,
        sample_every_seconds=args.sample_every_seconds,
    )
    sampled_indices = frame_indices_for_video(
        total_frames=video_info.total_frames,
        frame_step=frame_step,
        max_frames=args.max_frames,
    )
    if not sampled_indices:
        raise RuntimeError("No frames selected for processing. Check sampling arguments.")

    labels = {index: prompt for index, prompt in enumerate(args.classes)}
    exporter = YOLOExporter(output_dir=str(output_dir), labels=labels)
    predictor = build_predictor(args.model.resolve(), args)
    images_dir = output_dir / "images"

    LOG.info("Video: %s", args.video)
    LOG.info("Output directory: %s", output_dir)
    LOG.info(
        "Video metadata: %d frames at %.3f FPS (%dx%d)",
        video_info.total_frames,
        video_info.fps,
        video_info.width,
        video_info.height,
    )
    LOG.info("Class prompts: %s", ", ".join(args.classes))
    LOG.info(
        "Processing %d sampled frames with frame_step=%d",
        len(sampled_indices),
        frame_step,
    )

    annotations: dict[int, dict] = {}
    total_detections = 0
    frames_with_detections = 0

    progress = tqdm(total=len(sampled_indices), desc="Segmenting video frames")
    try:
        for frame_idx in sampled_indices:
            frame_path = ensure_frame_image(
                video_handler=video_handler,
                frame_idx=frame_idx,
                images_dir=images_dir,
                overwrite=args.overwrite_frames,
            )
            predictor.set_image(str(frame_path))
            results = predictor(text=args.classes)
            annotation = result_to_annotation(results[0], frame_idx, args.classes)
            annotations[frame_idx] = annotation
            detection_count = len(annotation["detections"])
            total_detections += detection_count
            if detection_count > 0:
                frames_with_detections += 1
            progress.update(1)
    finally:
        progress.close()
        video_handler.release()

    exporter.export(
        video_handler=VideoHandler(str(args.video.resolve())),
        annotations=annotations,
        include_detection=False,
        include_segmentation=True,
    )
    metadata = enrich_metadata_file(
        output_dir=output_dir,
        video_path=args.video.resolve(),
        video_info=video_info,
        frame_step=frame_step,
        sampled_indices=sampled_indices,
        sample_every_seconds=args.sample_every_seconds,
        frames_with_detections=frames_with_detections,
        total_detections=total_detections,
        annotations=annotations,
        class_prompts=args.classes,
    )
    log_export_metadata(metadata)

    write_run_config(
        output_dir=output_dir,
        args=args,
        fps=video_info.fps,
        total_frames=video_info.total_frames,
        sampled_frames=len(sampled_indices),
        frame_step=frame_step,
        total_detections=total_detections,
    )

    empty_label_files = len(sampled_indices) - frames_with_detections
    LOG.info("Export complete: %s", output_dir)
    LOG.info("Sampled frames: %d", len(sampled_indices))
    LOG.info("Frames with detections: %d", frames_with_detections)
    LOG.info("Frames without detections: %d", empty_label_files)
    LOG.info("Total detections exported: %d", total_detections)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Docstring for main."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    try:
        run(args)
    except KeyboardInterrupt:
        LOG.warning("Interrupted by user.")
        return 130
    except Exception as exc:
        LOG.error("Export failed: %s", exc, exc_info=args.verbose)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
