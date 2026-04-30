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


def test_labeling_state_reset():
    """Test state reset."""
    state = LabelingState()
    state.current_labels = [{"class_idx": 0, "points": [], "mask": None}]
    state.current_image_path = Path("test.jpg")
    state.current_image = "dummy"
    state.mask_opacity = 0.2
    state.unsaved_changes = True
    state.reset()
    assert state.current_labels == []
    assert state.current_image_path is None
    assert state.current_image is None
    assert state.mask_opacity == 0.62
