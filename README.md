# SAM3 Labeler - Interactive Annotation Tool

**SAM3 Labeler** is a web-based interactive annotation tool powered by [SAM 3 (Segment Anything Model)](https://github.com/ultralytics/ultralytics). It provides an intuitive interface for creating high-quality image annotations in multiple formats for object detection and segmentation tasks.

[中文使用手冊](docs/USER_MANUAL_zh-TW.md) | [English User Manual](docs/USER_MANUAL_en.md)

```
┌──────────────────────────────────────────────────────────────────┐
│  SAM3    [Image Folder]      [Output Folder]     [Load] [Jump]  │
├────────┬────────────────────────────────┬────────────────────────┤
│        │                                │                        │
│  Tool  │                                │    Control Panel       │
│  Panel │      Image Display Area        │                        │
│        │                                │    - Text Segment      │
│ ○ Select│   ←  [   Image   ]  →        │    - Class Selection   │
│ ○ Click │                               │    - Output Settings   │
│ ○ Box   │                               │    - Overlap Control   │
│        │                                │                        │
├────────┴────────────────────────────────┴────────────────────────┤
│  Label List                              [Delete] [Delete All]   │
└──────────────────────────────────────────────────────────────────┘
```

## Features

- **3 Segmentation Methods** - Point click, box selection, and text prompt
- **AI-Powered** - SAM 3 automatically generates precise segmentation masks
- **Multi-Format Output** - YOLO OBB, YOLO-Seg, PNG Mask, COCO JSON
- **Overlap Prevention** - Configurable annotation overlap detection
- **Batch Navigation** - Browse and annotate large image datasets efficiently
- **Auto-Save** - Annotations are saved automatically when navigating between images
- **Dynamic Classes** - Add/remove annotation classes on the fly
- **Dark Theme** - Eye-friendly web interface built with Gradio

## Quick Start

### 1. Install Dependencies

**Prerequisites**: Python 3.8+, NVIDIA GPU with CUDA (recommended)

```bash
# Install PyTorch first (choose your CUDA version)
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 2. Download SAM 3 Model

The SAM 3 model file (`sam3.pt`, ~3.4 GB) is **not included** in this repository. You must **manually download** it from Hugging Face.

**Download Steps**:

1. Visit the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3)
2. **Request access** (approval required by Meta)
3. Once approved, download `sam3.pt`:
   - Direct link: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true
4. Place `sam3.pt` in the **project root directory** (same folder as `sam3_labeler.py`)

| Model | Size | Source |
|-------|------|--------|
| `sam3.pt` | ~3.4 GB | [Hugging Face - facebook/sam3](https://huggingface.co/facebook/sam3) |

> **Alternative**: You can also use SAM 2 models which do **not** require access approval:
>
> | Model | Size | Source |
> |-------|------|--------|
> | `sam2_t.pt` | ~150 MB | Auto-download via Ultralytics |
> | `sam2_s.pt` | ~180 MB | Auto-download via Ultralytics |
> | `sam2_b.pt` | ~350 MB | Auto-download via Ultralytics |
> | `sam2_l.pt` | ~900 MB | Auto-download via Ultralytics |
>
> SAM 2 models are downloaded automatically on first use. See [Ultralytics SAM 2 Docs](https://docs.ultralytics.com/models/sam-2/).

**Reference**:
- [Ultralytics SAM 3 Docs](https://docs.ultralytics.com/models/sam-3/)
- [Ultralytics SAM 2 Docs](https://docs.ultralytics.com/models/sam-2/)
- [Ultralytics SAM (Original) Docs](https://docs.ultralytics.com/models/sam/)

### 3. Run

```bash
python sam3_labeler.py
```

The web interface will open in your browser at `http://localhost:7860`.

**With command-line arguments:**
```bash
# Specify image folder
python sam3_labeler.py --images /path/to/images

# Specify output folder
python sam3_labeler.py --output /path/to/output
```

## Project Structure

```
sam3-labeler/
├── sam3_labeler.py          # Main application
├── sam3_classes.txt          # Class definitions (editable)
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
│
├── sample_images/            # Sample images for testing
│   └── frame_000000.jpg ...
│
├── sample_output/            # Example annotation output
│   ├── labels/               # YOLO OBB format labels
│   └── classes.txt
│
├── scripts/                  # Utility scripts
│   ├── extract_frames.py     # Extract frames from video
│   ├── auto_label_obb.py     # Auto-label with SAM 3 text prompts
│   ├── prepare_dataset.py    # Split dataset (train/val)
│   ├── train_yolo_obb.py     # Train YOLO OBB model
│   └── predict_obb.py        # Run inference with trained model
│
└── docs/                     # Documentation
    ├── USER_MANUAL_zh-TW.md  # Detailed manual (Traditional Chinese)
    └── USER_MANUAL_en.md     # Detailed manual (English)
```

## Output Formats

SAM3 Labeler supports 4 output formats simultaneously:

### YOLO OBB (Oriented Bounding Box)
```
output/labels/image_name.txt
# class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)
0 0.512 0.234 0.612 0.234 0.612 0.456 0.512 0.456
```

### YOLO-Seg (Polygon Segmentation)
```
output/labels_seg/image_name.txt
# class_id x1 y1 x2 y2 ... xn yn (normalized polygon coordinates)
0 0.512 0.234 0.534 0.245 0.556 0.267 ...
```

### PNG Mask
```
output/masks/image_name_label_0.png
# Binary mask image (0 = background, 255 = object)
```

### COCO JSON
```
output/coco_annotations.json
# Standard COCO format with polygon annotations
```

## Utility Scripts

### Extract Frames from Video
```bash
# Extract 1 frame per second from a video
python scripts/extract_frames.py video.mp4 -o frames -m seconds -i 1

# Process all videos in a folder recursively
python scripts/extract_frames.py /path/to/videos -o frames -m seconds -i 1 --recursive

# View video info only
python scripts/extract_frames.py /path/to/videos --info
```

### Auto-Label with Text Prompts
```bash
# Automatically label objects using SAM 3 text segmentation
python scripts/auto_label_obb.py -i images_folder -o output_folder -t "trash,boat,diver"
```

### Prepare Dataset for Training
```bash
# Split labeled data into train/val sets (80/20)
python scripts/prepare_dataset.py --input labeled_dataset --output yolo_dataset --split 0.8
```

### Train YOLO OBB Model
```bash
python scripts/train_yolo_obb.py --epochs 100 --batch 8 --model yolov8n-obb.pt
```

### Run Inference
```bash
# Detect objects in images
python scripts/predict_obb.py --source images_folder/

# Detect in video
python scripts/predict_obb.py --source video.mp4
```

## Complete Workflow

```
                    ┌─────────────────┐
                    │   Video Files   │
                    │  (MP4/AVI/MOV)  │
                    └────────┬────────┘
                             │
                    extract_frames.py
                             │
                    ┌────────▼────────┐
                    │  Image Frames   │
                    └────────┬────────┘
                             │
                    sam3_labeler.py        ← Interactive annotation
                             │
                    ┌────────▼────────┐
                    │  Labeled Data   │
                    │  (OBB/Seg/Mask) │
                    └────────┬────────┘
                             │
                    prepare_dataset.py
                             │
                    ┌────────▼────────┐
                    │  Train / Val    │
                    └────────┬────────┘
                             │
                    train_yolo_obb.py
                             │
                    ┌────────▼────────┐
                    │  Trained Model  │
                    │  (best.pt)      │
                    └────────┬────────┘
                             │
                    predict_obb.py
                             │
                    ┌────────▼────────┐
                    │   Predictions   │
                    └─────────────────┘
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 |
| Python | 3.8 | 3.10+ |
| GPU | NVIDIA GTX 1060 (6GB) | NVIDIA RTX 3060+ (8GB+) |
| CUDA | 11.7 | 12.1+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB (with model) | 10 GB+ |

> **Note**: CPU-only mode is supported but significantly slower for SAM 3 inference.

## Troubleshooting

### Model not found
```
Error: SAM model not found
```
**Solution**: Ensure `sam3.pt` is in the project root or Ultralytics cache directory. Or let the tool download it automatically on first run.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Try a smaller model variant (`sam3_s.pt`) or close other GPU-intensive applications.

### Gradio port conflict
```
OSError: [Errno 98] Address already in use
```
**Solution**: Another instance may be running. Close it or specify a different port:
```python
# In sam3_labeler.py, modify the launch() call:
demo.launch(server_port=7861)
```

### GoPro video read issues
The tool sets `OPENCV_FFMPEG_READ_ATTEMPTS=65536` automatically for multi-stream video compatibility (GoPro, DJI, etc).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO and SAM model framework
- [Meta AI SAM](https://segment-anything.com/) - Segment Anything Model
- [Gradio](https://www.gradio.app/) - Web UI framework
