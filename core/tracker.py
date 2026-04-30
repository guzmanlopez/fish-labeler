"""Offline multi-object tracking based on geometric consistency across frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


@dataclass(slots=True)
class TrackingConfig:
    """Parameters controlling conservative frame association and offline stitching."""

    confidence_threshold: float = 0.55
    iou_gate: float = 0.35
    max_center_distance: float = 0.16
    max_missed_frames: int = 3
    max_size_change: float = 0.45
    max_aspect_change: float = 0.35
    velocity_weight: float = 0.25
    max_stitch_gap: int = 12
    stitch_center_distance: float = 0.20
    stitch_size_change: float = 0.30
    stitch_aspect_change: float = 0.22
    gap_penalty: float = 0.15


@dataclass(slots=True)
class DetectionRecord:
    """Normalized detection geometry used by the tracker."""

    frame_index: int
    detection_index: int
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    width: float
    height: float
    aspect_ratio: float
    area: float


@dataclass(slots=True)
class Tracklet:
    """Sequence of detection records linked across consecutive or nearby frames."""

    detections: list[DetectionRecord] = field(default_factory=list)
    missed_frames: int = 0

    @property
    def start_frame(self) -> int:
        """Return the first frame index in the tracklet."""
        return self.detections[0].frame_index

    @property
    def end_frame(self) -> int:
        """Return the last frame index in the tracklet."""
        return self.detections[-1].frame_index

    @property
    def class_id(self) -> int:
        """Return the tracked class id."""
        return self.detections[-1].class_id

    def last_detection(self) -> DetectionRecord:
        """Return the most recent detection assigned to this tracklet."""
        return self.detections[-1]

    def velocity(self) -> tuple[float, float]:
        """Estimate a simple constant velocity from the last two observations."""
        if len(self.detections) < 2:
            return (0.0, 0.0)
        last = self.detections[-1]
        prev = self.detections[-2]
        gap = max(1, last.frame_index - prev.frame_index)
        return (
            (last.center[0] - prev.center[0]) / gap,
            (last.center[1] - prev.center[1]) / gap,
        )

    def predicted_center(self, frame_index: int) -> tuple[float, float]:
        """Predict the detection center for a future frame using constant velocity."""
        last = self.last_detection()
        vx, vy = self.velocity()
        gap = max(0, frame_index - last.frame_index)
        return (last.center[0] + vx * gap, last.center[1] + vy * gap)

    def predicted_bbox(self, frame_index: int) -> tuple[float, float, float, float]:
        """Predict a future box by shifting the latest box with the track velocity."""
        last = self.last_detection()
        cx, cy = self.predicted_center(frame_index)
        half_w = last.width / 2
        half_h = last.height / 2
        return (
            max(0.0, cx - half_w),
            max(0.0, cy - half_h),
            min(1.0, cx + half_w),
            min(1.0, cy + half_h),
        )


@dataclass(slots=True)
class TrackingResult:
    """Per-frame track assignments and per-track summary metadata."""

    frame_track_ids: dict[int, list[int | None]]
    tracks: dict[int, dict]


def _log(message: str) -> None:
    """Print tracker progress and debug information to the console."""
    print(f"[tracking] {message}", flush=True)


def label_to_detection(
    frame_index: int,
    detection_index: int,
    label: tuple | list,
) -> DetectionRecord | None:
    """Convert a stored label tuple into a normalized detection record."""
    if len(label) < 2 or not label[1]:
        return None

    coords = list(label[1])
    if len(coords) < 8 or len(coords) % 2 != 0:
        _log(
            "skipping malformed detection "
            f"at frame={frame_index} index={detection_index}: expected 8 OBB values, got {len(coords)}"
        )
        return None

    coords = coords[:8]
    try:
        coords = [float(value) for value in coords]
    except (TypeError, ValueError):
        _log(f"skipping non-numeric detection at frame={frame_index} index={detection_index}")
        return None

    xs = [coords[i] for i in range(0, 8, 2)]
    ys = [coords[i + 1] for i in range(0, 8, 2)]
    if not xs or not ys:
        return None

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(1.0, max(xs))
    y2 = min(1.0, max(ys))
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    score = label[4] if len(label) > 4 and label[4] is not None else 1.0
    return DetectionRecord(
        frame_index=frame_index,
        detection_index=detection_index,
        class_id=int(label[0]),
        confidence=float(score),
        bbox=(x1, y1, x2, y2),
        center=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        width=width,
        height=height,
        aspect_ratio=width / height,
        area=width * height,
    )


def run_offline_tracker(
    frames: list[list[tuple | list]],
    config: TrackingConfig | None = None,
) -> TrackingResult:
    """Run conservative tracklet building followed by offline stitching."""
    tracker_config = config or TrackingConfig()
    _log(f"starting offline tracking for {len(frames)} frames")
    filtered_frames = _build_frame_detections(frames, tracker_config)
    _log(
        "frame parsing complete: "
        f"{sum(len(frame) for frame in filtered_frames)} detections kept after confidence filtering"
    )
    tracklets = _build_tracklets(filtered_frames, tracker_config)
    _log(f"pass 1 complete: built {len(tracklets)} tracklets")
    stitched = _stitch_tracklets(tracklets, tracker_config)
    _log(f"pass 2 complete: stitched down to {len(stitched)} tracks")
    return _build_tracking_result(frames, stitched)


def _build_frame_detections(
    frames: list[list[tuple | list]],
    config: TrackingConfig,
) -> list[list[DetectionRecord]]:
    """Convert raw frame labels into detections filtered by confidence."""
    detections_by_frame: list[list[DetectionRecord]] = []
    malformed_count = 0
    for frame_index, labels in enumerate(frames):
        frame_detections = []
        for detection_index, label in enumerate(labels):
            detection = label_to_detection(frame_index, detection_index, label)
            if detection is None:
                malformed_count += 1
                continue
            if detection.confidence < config.confidence_threshold:
                continue
            frame_detections.append(detection)
        detections_by_frame.append(frame_detections)
        if frame_index == len(frames) - 1 or (frame_index + 1) % 25 == 0:
            _log(
                f"parsed {frame_index + 1}/{len(frames)} frames "
                f"({sum(len(frame) for frame in detections_by_frame)} detections kept)"
            )
    if malformed_count:
        _log(f"skipped {malformed_count} malformed detections while building tracking inputs")
    return detections_by_frame


def _build_tracklets(
    detections_by_frame: list[list[DetectionRecord]],
    config: TrackingConfig,
) -> list[Tracklet]:
    """Associate detections across nearby frames using strict gating rules."""
    active: list[Tracklet] = []
    finished: list[Tracklet] = []

    for frame_index, detections in enumerate(detections_by_frame):
        assignments = _associate_tracklets(active, detections, frame_index, config)
        matched_tracks = set()
        matched_detections = set()

        for track_idx, det_idx in assignments:
            if track_idx < 0 or det_idx < 0:
                continue
            active[track_idx].detections.append(detections[det_idx])
            active[track_idx].missed_frames = 0
            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

        next_active: list[Tracklet] = []
        for idx, tracklet in enumerate(active):
            if idx in matched_tracks:
                next_active.append(tracklet)
                continue
            tracklet.missed_frames += 1
            if tracklet.missed_frames > config.max_missed_frames:
                finished.append(tracklet)
            else:
                next_active.append(tracklet)
        active = next_active

        for det_idx, detection in enumerate(detections):
            if det_idx not in matched_detections:
                active.append(Tracklet(detections=[detection]))

        if frame_index == len(detections_by_frame) - 1 or (frame_index + 1) % 25 == 0:
            _log(
                f"pass 1 progress {frame_index + 1}/{len(detections_by_frame)} frames: "
                f"active={len(active)} finished={len(finished)}"
            )

    finished.extend(active)
    return finished


def _associate_tracklets(
    active: list[Tracklet],
    detections: list[DetectionRecord],
    frame_index: int,
    config: TrackingConfig,
) -> list[tuple[int, int]]:
    """Assign active tracklets to current detections with gated costs."""
    if not active or not detections:
        return []

    cost_matrix: list[list[float]] = []
    valid_matrix: list[list[bool]] = []
    invalid_cost = 1_000_000.0

    for tracklet in active:
        row_costs = []
        row_valid = []
        for detection in detections:
            is_valid, cost = _track_detection_cost(tracklet, detection, frame_index, config)
            row_valid.append(is_valid)
            row_costs.append(cost if is_valid else invalid_cost)
        cost_matrix.append(row_costs)
        valid_matrix.append(row_valid)

    matches = _hungarian(cost_matrix)
    accepted: list[tuple[int, int]] = []
    for track_idx, det_idx in matches:
        if det_idx == -1:
            continue
        if track_idx < 0 or track_idx >= len(valid_matrix):
            _log(f"ignoring invalid assignment with track index {track_idx}")
            continue
        if det_idx < 0 or det_idx >= len(valid_matrix[track_idx]):
            _log(
                "ignoring invalid assignment "
                f"track={track_idx} detection={det_idx} outside current frame bounds"
            )
            continue
        if valid_matrix[track_idx][det_idx]:
            accepted.append((track_idx, det_idx))
    return accepted


def _track_detection_cost(
    tracklet: Tracklet,
    detection: DetectionRecord,
    frame_index: int,
    config: TrackingConfig,
) -> tuple[bool, float]:
    """Compute gated association cost for a detection candidate."""
    if detection.class_id != tracklet.class_id:
        return False, 0.0

    last = tracklet.last_detection()
    frame_gap = max(1, frame_index - last.frame_index)
    predicted_bbox = tracklet.predicted_bbox(frame_index)
    predicted_center = tracklet.predicted_center(frame_index)
    iou = _bbox_iou(predicted_bbox, detection.bbox)
    center_distance = _center_distance(predicted_center, detection.center)
    center_gate = config.max_center_distance * frame_gap
    size_change = _size_change(last, detection)
    aspect_change = _aspect_change(last, detection)

    if iou < config.iou_gate:
        return False, 0.0
    if center_distance > center_gate:
        return False, 0.0
    if size_change > config.max_size_change:
        return False, 0.0
    if aspect_change > config.max_aspect_change:
        return False, 0.0

    velocity_error = _center_distance(predicted_center, detection.center)
    cost = (
        (1.0 - iou)
        + (center_distance / max(center_gate, 1e-6))
        + size_change
        + aspect_change
        + config.velocity_weight * velocity_error
    )
    return True, cost


def _stitch_tracklets(tracklets: list[Tracklet], config: TrackingConfig) -> list[Tracklet]:
    """Merge clean tracklets offline when their motion and geometry remain compatible."""
    stitched = [tracklet for tracklet in tracklets if tracklet.detections]
    if len(stitched) < 2:
        return stitched

    changed = True
    while changed:
        changed = False
        active_indices = list(range(len(stitched)))
        cost_matrix: list[list[float]] = []
        valid_matrix: list[list[bool]] = []
        invalid_cost = 1_000_000.0

        for left_idx in active_indices:
            row_costs = []
            row_valid = []
            for right_idx in active_indices:
                if left_idx == right_idx:
                    row_valid.append(False)
                    row_costs.append(invalid_cost)
                    continue
                is_valid, cost = _tracklet_stitch_cost(
                    stitched[left_idx], stitched[right_idx], config
                )
                row_valid.append(is_valid)
                row_costs.append(cost if is_valid else invalid_cost)
            cost_matrix.append(row_costs)
            valid_matrix.append(row_valid)

        matches = _hungarian(cost_matrix)
        merges: list[tuple[int, int]] = []
        used = set()
        for left_idx, right_idx in matches:
            if right_idx == -1:
                continue
            if left_idx in used or right_idx in used:
                continue
            if valid_matrix[left_idx][right_idx]:
                merges.append((left_idx, right_idx))
                used.add(left_idx)
                used.add(right_idx)

        if not merges:
            break

        _log(f"stitching iteration merged {len(merges)} tracklet pairs")

        for left_idx, right_idx in sorted(merges, reverse=True):
            left = stitched[left_idx]
            right = stitched[right_idx]
            merged = Tracklet(
                detections=sorted(
                    left.detections + right.detections,
                    key=lambda det: (det.frame_index, det.detection_index),
                )
            )
            del stitched[max(left_idx, right_idx)]
            del stitched[min(left_idx, right_idx)]
            stitched.append(merged)
            changed = True
    stitched.sort(key=lambda tracklet: (tracklet.start_frame, tracklet.class_id))
    return stitched


def _tracklet_stitch_cost(
    left: Tracklet,
    right: Tracklet,
    config: TrackingConfig,
) -> tuple[bool, float]:
    """Compute a conservative merge cost between two disjoint tracklets."""
    if left.class_id != right.class_id:
        return False, 0.0
    if right.start_frame <= left.end_frame:
        return False, 0.0

    gap = right.start_frame - left.end_frame - 1
    if gap > config.max_stitch_gap:
        return False, 0.0

    predicted_bbox = left.predicted_bbox(right.start_frame)
    predicted_center = left.predicted_center(right.start_frame)
    start_detection = right.detections[0]
    end_detection = left.detections[-1]
    center_distance = _center_distance(predicted_center, start_detection.center)
    center_gate = config.stitch_center_distance * max(1, gap + 1)
    size_change = _size_change(end_detection, start_detection)
    aspect_change = _aspect_change(end_detection, start_detection)
    if center_distance > center_gate:
        return False, 0.0
    if size_change > config.stitch_size_change:
        return False, 0.0
    if aspect_change > config.stitch_aspect_change:
        return False, 0.0

    predicted_iou = _bbox_iou(predicted_bbox, start_detection.bbox)
    cost = (
        (center_distance / max(center_gate, 1e-6))
        + size_change
        + aspect_change
        + (1.0 - predicted_iou)
        + config.gap_penalty * (gap / max(1, config.max_stitch_gap))
    )
    return True, cost


def _build_tracking_result(
    frames: list[list[tuple | list]],
    tracklets: list[Tracklet],
) -> TrackingResult:
    """Convert tracklets into per-frame assignments and compact summaries."""
    frame_track_ids = {
        frame_index: [None] * len(labels) for frame_index, labels in enumerate(frames)
    }
    tracks: dict[int, dict] = {}
    for track_id, tracklet in enumerate(tracklets, start=1):
        frame_ids = [detection.frame_index for detection in tracklet.detections]
        confidences = [detection.confidence for detection in tracklet.detections]
        for detection in tracklet.detections:
            frame_track_ids[detection.frame_index][detection.detection_index] = track_id
        tracks[track_id] = {
            "track_id": track_id,
            "class_id": tracklet.class_id,
            "start_frame": min(frame_ids),
            "end_frame": max(frame_ids),
            "frame_count": len(set(frame_ids)),
            "detection_count": len(tracklet.detections),
            "mean_confidence": sum(confidences) / len(confidences),
            "max_gap": _max_gap(frame_ids),
        }
    return TrackingResult(frame_track_ids=frame_track_ids, tracks=tracks)


def _hungarian(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Solve a rectangular assignment problem using the Hungarian algorithm."""
    if not cost_matrix or not cost_matrix[0]:
        return []

    transposed = False
    matrix = cost_matrix
    row_count = len(matrix)
    col_count = len(matrix[0])
    if row_count > col_count:
        transposed = True
        matrix = [list(row) for row in zip(*matrix)]
        row_count, col_count = col_count, row_count

    u = [0.0] * (row_count + 1)
    v = [0.0] * (col_count + 1)
    p = [0] * (col_count + 1)
    way = [0] * (col_count + 1)

    for i in range(1, row_count + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (col_count + 1)
        used = [False] * (col_count + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, col_count + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(col_count + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * row_count
    for j in range(1, col_count + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    if transposed:
        return [
            (col_index, row_index)
            for row_index, col_index in enumerate(assignment)
            if col_index != -1
        ]
    return [
        (row_index, col_index) for row_index, col_index in enumerate(assignment) if col_index != -1
    ]


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two normalized boxes."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0
    left_area = max(1e-6, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1e-6, (right[2] - right[0]) * (right[3] - right[1]))
    return intersection / max(1e-6, left_area + right_area - intersection)


def _center_distance(
    left: tuple[float, float],
    right: tuple[float, float],
) -> float:
    """Compute Euclidean distance in normalized image coordinates."""
    return hypot(left[0] - right[0], left[1] - right[1])


def _size_change(left: DetectionRecord, right: DetectionRecord) -> float:
    """Measure relative box-size drift between two detections."""
    width_change = abs(left.width - right.width) / max(left.width, right.width, 1e-6)
    height_change = abs(left.height - right.height) / max(left.height, right.height, 1e-6)
    return max(width_change, height_change)


def _aspect_change(left: DetectionRecord, right: DetectionRecord) -> float:
    """Measure normalized aspect-ratio drift between two detections."""
    return abs(left.aspect_ratio - right.aspect_ratio) / max(
        left.aspect_ratio, right.aspect_ratio, 1e-6
    )


def _max_gap(frame_ids: list[int]) -> int:
    """Return the largest missing-frame gap observed inside a track."""
    if len(frame_ids) < 2:
        return 0
    return max(frame_ids[i + 1] - frame_ids[i] - 1 for i in range(len(frame_ids) - 1))
