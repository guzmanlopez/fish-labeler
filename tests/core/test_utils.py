import numpy as np
from core.utils import mask_to_binary_image, point_in_obb


def test_mask_to_binary_image():
    """Test generating binary image from mask."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    binary = mask_to_binary_image(mask)
    assert binary.shape == (10, 10)
    assert binary[3, 3] == 255
    assert binary[0, 0] == 0


def test_point_in_obb():
    """Test if point is in OBB."""
    obb = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]  # normalized!
    assert point_in_obb(5, 5, obb, 20, 20)
    assert not point_in_obb(15, 15, obb, 20, 20)
