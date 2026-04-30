from pathlib import Path

from core.sam_engine import DEFAULT_SAM_CONF

"""Annotation state model for the labeling workflow."""


class LabelingState:
    """Store mutable UI state for the active dataset, image, and annotations."""

    def __init__(self):
        """Docstring for __init__."""
        self.reset()

    def reset(self):
        """Docstring for reset."""
        self.current_image = None  # numpy RGB image
        self.current_image_path = None  # Path object
        self.current_masks = []
        # Label data: [(class_id, obb_coords, polygon_coords, mask_binary, score), ...]
        self.current_labels = []
        self.image_list = []  # List[Path]
        self.current_index = 0
        self.classes = ["fish"]
        self.sam_predictor = None  # SAM3SemanticPredictor
        self.sam_model = None  # SAM model (click/box)
        self.output_folder = Path("labeled_dataset")
        # Box selection state
        self.box_first_point = None
        # Queued point prompts for interactive SAM point segmentation
        self.positive_prompt_points = []
        self.negative_prompt_points = []
        self.point_prompt_target = "positive"
        self.keep_positive_points_across_frames = False
        self.keep_negative_points_across_frames = False
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
        # Class score thresholds
        self.class_thresholds = {"fish": DEFAULT_SAM_CONF}
        # Mask overlay opacity in the canvas
        self.mask_opacity = 0.62
        # Persisted track ids by frame filename
        self.frame_track_ids = {}
        # Track summary metadata keyed by track id
        self.track_summaries = {}
        # Selected track ids in the tracking sidebar
        self.selected_track_ids = set()
        # Offline tracker configuration
        self.tracking_config = {
            "confidence_threshold": 0.55,
            "iou_gate": 0.35,
            "max_center_distance": 0.16,
            "max_missed_frames": 3,
            "max_size_change": 0.45,
            "max_aspect_change": 0.35,
            "velocity_weight": 0.25,
            "max_stitch_gap": 12,
            "stitch_center_distance": 0.20,
            "stitch_size_change": 0.30,
            "stitch_aspect_change": 0.22,
            "gap_penalty": 0.15,
        }
        # Axis-aligned bounding box display mode
        self.display_aabb = True
        # Display mode
        self.display_mode = "both"  # outline / mask / both
