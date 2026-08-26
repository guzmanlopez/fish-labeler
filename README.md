# Fish Labeler

**Fish Labeler** prepares and annotates fish imagery collected aboard fishing vessels. Use the video workflow to sample frames and create an initial YOLO segmentation dataset, then use the Qt workflow to review and refine annotations.

[English User Manual](docs/USER_MANUAL_en.md)

## Features

- **3 Segmentation Methods** — Point click, box selection, and text prompt
- **AI-Powered** — SAM 3 automatically generates precise segmentation masks
- **Multi-Format Output** — YOLO OBB, YOLO-Seg, and PNG masks
- **Real-Time Rendering** — QPainter vector canvas with millisecond-level interaction
- **Zoom & Pan** — Scroll wheel zoom, right-click / Space+click / middle-click pan
- **Hover Highlight** — Dashed outline on hover, cyan highlight on selection
- **Background Inference** — SAM runs in a separate thread, UI stays responsive
- **Overlap Prevention** — Configurable annotation overlap detection
- **Batch Navigation** — Browse and annotate large image datasets efficiently
- **Auto-Save** — Annotations are saved automatically when navigating between images
- **Dynamic Classes** — Add/remove annotation classes on the fly
- **Dark Theme** — Eye-friendly desktop interface built with PyQt6

## Quick Start

### 1. Install Dependencies

**Prerequisites**: Python 3.12+, [uv](https://docs.astral.sh/uv/), and an NVIDIA GPU for practical SAM inference.

```bash
# Install PyTorch first (choose your CUDA version)
# Visit https://pytorch.org/get-started/locally/ for the correct command
# Example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install the locked project dependencies
uv sync --locked --all-extras --dev
```

### 2. Download SAM 3 From Hugging Face

The SAM 3 weights (`sam3.pt`, about 3.4 GB) are not included in this repository. Meta gates the [facebook/sam3 model repository](https://huggingface.co/facebook/sam3), so the Hugging Face account used to download the file must be approved first.

1. Sign in to Hugging Face and open the [SAM 3 model page](https://huggingface.co/facebook/sam3).
2. Select **Request access**, accept the model terms, and wait for Meta approval.
3. After approval, download `sam3.pt` from the repository's **Files and versions** tab.
4. Place the downloaded file at `src/models/sam3.pt`.

Alternatively, download directly to the expected directory with the Hugging Face CLI. Authenticate with a token for an account approved to access `facebook/sam3`:

```bash
uv tool install "huggingface_hub[cli]"
hf auth login
hf download facebook/sam3 sam3.pt --local-dir src/models
```

Verify the file before starting a workflow:

```bash
test -f src/models/sam3.pt && echo "SAM 3 model ready"
```

See the [Ultralytics SAM 3 documentation](https://docs.ultralytics.com/models/sam-3/) for model details and hardware guidance.

### 3. Run

```bash
uv run fish-labeler app
uv run fish-labeler app --images /data/vessel-trip-01/images --output vessel-trip-01
uv run fish-labeler app --model /path/to/another-model.pt
uv run fish-labeler video --video /data/vessel-trip-01.mp4 --output-dir vessel-trip-01
```

`fish-labeler video` writes sampled images and initial labels to `output/<run-name>/`. Open those images in the Qt application with `fish-labeler app --images output/<run-name>/images --output <run-name>` to review and refine the dataset. All generated files stay under the repository `output/` directory, even when source media is external or linked.

## Three Annotation Modes

| Mode | Shortcut | Description |
|------|----------|-------------|
| Click | `1` | Click object center, SAM auto-segments |
| Box Select | `2` | Drag a rectangle, SAM segments within |
| Text Prompt | — | Type object name, SAM finds all matches |

## Output Formats

Fish Labeler supports three output formats simultaneously:

### YOLO OBB (Oriented Bounding Box)
```
output/<run-name>/labels/image_name.txt
# class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates)
0 0.512 0.234 0.612 0.234 0.612 0.456 0.512 0.456
```

### YOLO-Seg (Polygon Segmentation)
```
output/<run-name>/labels_seg/image_name.txt
# class_id x1 y1 x2 y2 ... xn yn (normalized polygon coordinates)
0 0.512 0.234 0.534 0.245 0.556 0.267 ...
```

### PNG Mask
```
output/<run-name>/masks/image_name.png
# Binary mask image (0 = background, 255 = object)
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
fish-labeler/
├── pyproject.toml        # Commands and dependency configuration
├── docs/
│   └── USER_MANUAL_en.md
├── output/               # All generated run directories
└── src/
    ├── main.py          # CLI workflows: app and video
    ├── config/          # Local classes, progress, and UI settings
    ├── models/           # SAM model weights
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
**Solution**: Ensure `sam3.pt` is at `src/models/sam3.pt` or specify `--model /path/to/sam3.pt`.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Try a smaller model variant (`sam2_s.pt`) or close other GPU-intensive applications.

### Click does nothing
Check the tool is set to Click mode (`1`). Verify cursor is within the image (coordinates shown at bottom-left). Try Box mode or enable "Fallback to box".

### SAM inference takes long
First inference loads the model (10-30s). Subsequent runs take 1-3s. UI remains responsive during inference.

## Acknowledgments

- This project is supported by the **Ocean Conservation Administration, Ocean Affairs Council** (海洋委員會海洋保育署)
- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO and SAM model framework
- [Meta AI SAM](https://segment-anything.com/) — Segment Anything Model
- [Qt / PyQt6](https://www.riverbankcomputing.com/software/pyqt/) — Desktop UI framework
