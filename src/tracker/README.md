# Offline Tracking

`offline_tracker.py` links detections with the same class across a loaded image sequence. It first creates conservative tracklets from nearby frames, then stitches compatible tracklets across short gaps. Coordinates are normalized to the range `0.0` to `1.0`.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `confidence_threshold` | `0.55` | Discards detections below this confidence before tracking begins. |
| `iou_gate` | `0.35` | Minimum intersection-over-union between predicted and detected boxes for frame-to-frame association. |
| `max_center_distance` | `0.16` | Largest normalized center displacement allowed when matching a detection to an active tracklet. |
| `max_missed_frames` | `3` | Number of consecutive unmatched frames an active tracklet may survive. |
| `max_size_change` | `0.45` | Largest relative box-area change allowed between matched detections. |
| `max_aspect_change` | `0.35` | Largest relative width-to-height ratio change allowed between matched detections. |
| `velocity_weight` | `0.25` | Weight of constant-velocity prediction in the frame-to-frame match cost. |
| `max_stitch_gap` | `12` | Largest number of frames between tracklets that may be joined during offline stitching. |
| `stitch_center_distance` | `0.20` | Largest normalized predicted-center mismatch allowed when stitching tracklets. |
| `stitch_size_change` | `0.30` | Largest relative area change allowed when stitching tracklets. |
| `stitch_aspect_change` | `0.22` | Largest relative aspect-ratio change allowed when stitching tracklets. |
| `gap_penalty` | `0.15` | Additional match-cost penalty for each longer stitching gap. |

Higher gate values make matching less strict. Increase `max_missed_frames` or `max_stitch_gap` for longer occlusions; lower the geometry gates when preventing incorrect joins matters more than maintaining continuous tracks.
