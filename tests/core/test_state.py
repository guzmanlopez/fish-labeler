from pathlib import Path

from core.sam_engine import DEFAULT_SAM_CONF
from core.state import LabelingState


def test_labeling_state_init():
    """Test state initialization."""
    state = LabelingState()
    assert state.current_labels == []
    assert state.current_image_path is None
    assert state.current_image is None
    assert state.classes == ["fish"]
    assert state.class_thresholds == {"fish": DEFAULT_SAM_CONF}
    assert state.mask_opacity == 0.62
    assert state.positive_prompt_points == []
    assert state.negative_prompt_points == []
    assert state.point_prompt_target == "positive"
    assert state.keep_positive_points_across_frames is False
    assert state.keep_negative_points_across_frames is False
    assert state.frame_track_ids == {}
    assert state.track_summaries == {}
    assert state.selected_track_ids == set()
    assert state.tracking_config["max_stitch_gap"] == 12


def test_labeling_state_reset():
    """Test state reset."""
    state = LabelingState()
    state.current_labels = [{"class_idx": 0, "points": [], "mask": None}]
    state.current_image_path = Path("test.jpg")
    state.current_image = "dummy"
    state.mask_opacity = 0.2
    state.positive_prompt_points = [(1, 2)]
    state.negative_prompt_points = [(3, 4)]
    state.keep_positive_points_across_frames = True
    state.keep_negative_points_across_frames = True
    state.frame_track_ids = {"frame_0001.jpg": [1, None]}
    state.unsaved_changes = True
    state.reset()
    assert state.current_labels == []
    assert state.current_image_path is None
    assert state.current_image is None
    assert state.mask_opacity == 0.62
    assert state.positive_prompt_points == []
    assert state.negative_prompt_points == []
    assert state.keep_positive_points_across_frames is False
    assert state.keep_negative_points_across_frames is False
    assert state.frame_track_ids == {}
