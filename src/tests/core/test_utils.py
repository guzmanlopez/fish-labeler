"""Tests for core geometric and mask utilities."""

import numpy as np

from core.utils import mask_to_binary_image, point_in_quad, remove_labels_in_box


def test_mask_to_binary_image():
    """Test generating binary image from mask."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    binary = mask_to_binary_image(mask)
    assert binary.shape == (10, 10)
    assert binary[3, 3] == 255
    assert binary[0, 0] == 0


def test_point_in_quad():
    """Test whether a point is in a quadrilateral."""
    quad = [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
    assert point_in_quad(5, 5, quad, 20, 20)
    assert not point_in_quad(15, 15, quad, 20, 20)


def test_remove_labels_in_box_keeps_only_non_intersecting_labels():
    """Rectangle removal should preserve labels completely outside the selection."""
    labels = [
        (0, [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]),
        (1, [0.7, 0.7, 0.9, 0.7, 0.9, 0.9, 0.7, 0.9]),
    ]

    remaining = remove_labels_in_box(5, 5, 35, 35, labels, 100, 100)

    assert remaining == [labels[1]]
