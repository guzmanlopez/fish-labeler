"""
Annotation state management
Preserves all original LabelingState fields and logic
"""

from pathlib import Path


class LabelingState:
    """Annotation state management (preserves all original fields)"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.current_image = None  # numpy RGB image
        self.current_image_path = None  # Path object
        self.current_masks = []
        # Label data: [(class_id, obb_coords, polygon_coords, mask_binary), ...]
        self.current_labels = []
        self.image_list = []  # List[Path]
        self.current_index = 0
        self.classes = ["fish"]
        self.sam_predictor = None  # SAM3SemanticPredictor
        self.sam_model = None  # SAM model (click/box)
        self.output_folder = Path("labeled_dataset")
        # Box selection state
        self.box_first_point = None
        # Multi-select state
        self.selected_labels = set()
        # Output formats
        self.output_formats = {
            "obb": False,
            "seg": True,
            "mask": False,
            "coco": False,
        }
        # Polygon simplification
        self.polygon_epsilon = 0.005
        # Overlap threshold
        self.overlap_threshold = 0.1
        # Display mode
        self.display_mode = "outline"  # outline / mask / both
