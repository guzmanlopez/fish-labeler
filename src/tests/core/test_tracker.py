"""Tests for the offline tracklet tracker."""

from core.tracker import TrackingConfig, run_offline_tracker


def make_box(class_id, x1, y1, x2, y2, score=0.9):
    """Create a label tuple using axis-aligned OBB coordinates for tracker tests."""
    obb = [x1, y1, x2, y1, x2, y2, x1, y2]
    return (class_id, obb, obb, None, score)


def test_offline_tracker_stitches_tracklets_across_small_gaps():
    """A short gap in detections should be stitched into a single stable track."""
    frames = [
        [make_box(0, 0.10, 0.10, 0.20, 0.20)],
        [make_box(0, 0.12, 0.10, 0.22, 0.20)],
        [],
        [make_box(0, 0.16, 0.10, 0.26, 0.20)],
    ]
    config = TrackingConfig(
        confidence_threshold=0.1,
        iou_gate=0.2,
        max_center_distance=0.20,
        max_missed_frames=0,
        max_stitch_gap=3,
        stitch_center_distance=0.25,
        stitch_size_change=0.5,
        stitch_aspect_change=0.5,
    )

    result = run_offline_tracker(frames, config)

    assert len(result.tracks) == 1
    assert result.frame_track_ids[0][0] == 1
    assert result.frame_track_ids[1][0] == 1
    assert result.frame_track_ids[3][0] == 1


def test_offline_tracker_keeps_distant_objects_separate():
    """Large jumps should produce separate track ids instead of identity switches."""
    frames = [
        [make_box(0, 0.10, 0.10, 0.20, 0.20)],
        [make_box(0, 0.70, 0.70, 0.80, 0.80)],
    ]
    config = TrackingConfig(
        confidence_threshold=0.1,
        iou_gate=0.2,
        max_center_distance=0.08,
        max_missed_frames=0,
        max_stitch_gap=1,
        stitch_center_distance=0.10,
    )

    result = run_offline_tracker(frames, config)

    assert len(result.tracks) == 2
    assert result.frame_track_ids[0][0] != result.frame_track_ids[1][0]


def test_offline_tracker_skips_malformed_boxes_without_crashing():
    """Malformed OBB arrays should be ignored instead of raising index errors."""
    frames = [
        [(0, [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1], [0.1, 0.1], None, 0.9)],
        [make_box(0, 0.12, 0.10, 0.22, 0.20)],
    ]

    result = run_offline_tracker(frames, TrackingConfig(confidence_threshold=0.1))

    assert result.frame_track_ids[0] == [None]
    assert result.frame_track_ids[1][0] == 1
