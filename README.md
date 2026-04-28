# SAM3 Labeler - Interactive Annotation Tool

**SAM3 Labeler** is a desktop interactive annotation tool powered by [SAM 3 (Segment Anything Model)](https://github.com/ultralytics/ultralytics). It provides an intuitive interface for creating high-quality image annotations in multiple formats for object detection and segmentation tasks.

> **v1 (Gradio)** is available at tag [`v1.0`](../../tree/v1.0).
> **v2 (PyQt6)** is the current version — a full desktop rewrite with real-time rendering.

[中文使用手冊](docs/USER_MANUAL_zh-TW.md) | [English User Manual](docs/USER_MANUAL_en.md)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SAM3  [Image Folder] [📁] [Output Folder]  [Load] │ [N] [Go]  7/13   │
├────────┬──────────────────────────────────────┬──────────────────────────┤
│        │                                      │                          │
│ Tools  │                                      │  Text Prompt             │
│        │                                      │  [input] [▶ Run]         │
│ ○ Click│                                      │                          │
│ ○ Box  │          Canvas                      │  Class                   │
│ ○ Select│                                     │  [dropdown] [+] [−]      │
│        │     (scroll zoom / right-click pan)  │                          │
│        │     (coords at bottom-left)          │  Settings                │
│ [⊞ Fit]│                                      │  ☑ Fallback to box      │
│ [◀Prev]│                                      │  ○outline ○mask ○both   │
│ [Next▶]│                                      │  Simplify ─●── 0.005    │
│        │                                      │  Overlap  ──●─ 10%      │
│        │                                      │                          │
│        │                                      │  Output Formats          │
│        │                                      │  ☑OBB ☑Seg ☐Mask ☐COCO │
│        │                                      │                          │
│        │                                      │  Annotations             │
│        │                                      │  ■ 1. plastic_bottle    │
│        │                                      │  ■ 2. foam (selected)   │
│        │                                      │                          │
│        │                                      │  [Class▼] [Apply]        │
│        │                                      │  [🗑Delete] [Clear All]  │
│        │                                      │  [💾 Save]              │
├────────┴──────────────────────────────────────┴──────────────────────────┤
│ 📷 7/13  🏷️ 5 annotations  |  frame_006.jpg                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Features

- **3 Segmentation Methods** — Point click, box selection, and text prompt
- **AI-Powered** — SAM 3 automatically generates precise segmentation masks
- **Multi-Format Output** — YOLO OBB, YOLO-Seg, PNG Mask, COCO JSON
- **Real-Time Rendering** — QPainter vector canvas with millisecond-level interaction
- **Zoom & Pan** — Scroll wheel zoom, right-click / Space+click / middle-click pan
- **Hover Highlight** — Dashed outline on hover, cyan highlight on selection
- **Background Inference** — SAM runs in a separate thread, UI stays responsive
- **Overlap Prevention** — Configurable annotation overlap detection
- **Batch Navigation** — Browse and annotate large image datasets efficiently
- **Auto-Save** — Annotations are saved automatically when navigating between images
- **Dynamic Classes** — Add/remove annotation classes on the fly
- **Dark Theme** — Eye-friendly desktop interface built with PyQt6

## v2 Highlights (vs v1 Gradio)

| Action | Gradio (v1) | PyQt6 (v2) |
|--------|-------------|------------|
| Click → redraw | 300-500 ms | < 5 ms |
| Hover highlight | Not supported | < 3 ms |
| Box drag | No feedback | Real-time |
| SAM inference | UI freezes | Background thread |
| Zoom / Pan | Not supported | Scroll + drag |

## Quick Start

### 1. Install Dependencies

**Prerequisites**: Python 3.10+, GPU recommended (NVIDIA CUDA / Apple MPS)

```bash
# Install PyTorch first (choose your CUDA version)
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install PyQt6 opencv-python numpy ultralytics
```

### 2. Download SAM 3 Model

The SAM 3 model file (`sam3.pt`, ~3.4 GB) is **not included** in this repository. You must **manually download** it from Hugging Face.

**Download Steps**:

1. Visit the [SAM 3 model page on Hugging Face](https://huggingface.co/facebook/sam3)
2. **Request access** (approval required by Meta)
3. Once approved, download `sam3.pt`:
   - Direct link: https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true
4. Place `sam3.pt` in the **project root directory** (same folder as `main.py`)

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
python main.py --model /path/to/sam3.pt
python main.py --model sam3.pt --images ./images --output ./output
```

## Three Annotation Modes

| Mode | Shortcut | Description |
|------|----------|-------------|
| Click | `1` | Click object center, SAM auto-segments |
| Box Select | `2` | Drag a rectangle, SAM segments within |
| Text Prompt | — | Type object name, SAM finds all matches |

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
output/masks/image_name.png
# Binary mask image (0 = background, 255 = object)
```

### COCO JSON
```
output/coco_annotations.json
# Standard COCO format with polygon annotations
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` `→` | Previous / Next image (auto-save) |
| `1` `2` `3` | Switch mode: Click / Box / Select |
| `Delete` | Delete selected annotations |
| `Ctrl+S` | Save |
| `Ctrl+A` | Select all |
| `Esc` | Deselect |
| `F` | Fit to window |
| Scroll wheel | Zoom (centered on cursor) |
| Right-click drag | Pan |
| Space + left-click drag | Pan |
| Middle-click drag | Pan |
| Double-click | Fit to window |

## Project Structure

```
sam3-labeler/
├── main.py              # Entry point
├── requirements.txt
├── docs/
│   ├── USER_MANUAL_en.md
│   └── USER_MANUAL_zh-TW.md
├── core/
│   ├── state.py         # LabelingState
│   ├── utils.py         # Coordinate transforms, overlap detection
│   ├── io_manager.py    # Config, progress, label I/O
│   └── sam_engine.py    # SAM 3 model wrapper
└── ui/
    ├── canvas.py        # QPainter vector canvas
    └── main_window.py   # Main window + control panels
```

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 / macOS 12+ | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| Python | 3.10 | 3.12 |
| GPU | NVIDIA GTX 1060 (6GB) / Apple M1 | NVIDIA RTX 3060+ (8GB+) / Apple M2+ |
| CUDA | 11.7 (macOS uses MPS, no CUDA needed) | 12.1+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB (with model) | 10 GB+ |

> **Cross-platform:** The application automatically detects the best compute device (CUDA GPU → Apple MPS → CPU). No manual configuration needed.

## Troubleshooting

### Model not found
```
Error: SAM model not found
```
**Solution**: Ensure `sam3.pt` is in the project root or specify the path with `--model /path/to/sam3.pt`.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Try a smaller model variant (`sam2_s.pt`) or close other GPU-intensive applications.

### Click does nothing
Check the tool is set to Click mode (`1`). Verify cursor is within the image (coordinates shown at bottom-left). Try Box mode or enable "Fallback to box".

### SAM inference takes long
First inference loads the model (10-30s). Subsequent runs take 1-3s. UI remains responsive during inference.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is supported by the **Ocean Conservation Administration, Ocean Affairs Council** (海洋委員會海洋保育署)
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO and SAM model framework
- [Meta AI SAM](https://segment-anything.com/) — Segment Anything Model
- [Qt / PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — Desktop UI framework
