# SAM3 Labeler (PyQt6) — User Manual

## Introduction

Welcome to SAM3 Labeler PyQt6 edition! This desktop application helps you annotate objects in images. Compared to the original Gradio web version, the PyQt6 version offers millisecond-level interaction, native zoom/pan, real-time hover highlights, and non-blocking SAM inference.

> **Also available in:** [繁體中文](USER_MANUAL_zh-TW.md)

**What can this tool do?**

- Annotate objects using three methods: click, box-select, or text prompt
- Auto-detect object boundaries via Meta SAM 3
- Export in OBB / YOLO-Seg / PNG Mask / COCO JSON simultaneously
- Zoom with scroll wheel, pan with right-click drag — ideal for high-res images

---

## Chapter 1: Installation

### 1.1 System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 / macOS 12+ | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| Python | 3.10 | 3.12 |
| GPU | NVIDIA GTX 1060 (6GB) / Apple M1 | NVIDIA RTX 3060+ (8GB+) / Apple M2+ |
| CUDA | 11.7 (macOS uses MPS, no CUDA needed) | 12.1+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB (incl. model) | 10 GB+ |

> **Cross-platform:** The application automatically detects the best compute device (CUDA GPU → Apple MPS → CPU). No manual configuration needed.

### 1.2 Installation

```bash
# 1. Install PyTorch (choose your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Install dependencies
pip install PyQt6 opencv-python numpy ultralytics
```

### 1.3 Download SAM 3 Model

`sam3.pt` (~3.4 GB) is not included. Download from [Hugging Face](https://huggingface.co/facebook/sam3) (requires Meta approval).

### 1.4 Launch

```bash
python main.py
python main.py --model /path/to/sam3.pt
python main.py --model sam3.pt --images ./images --output ./output
```

---

## Chapter 2: Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  SAM3  [Image Folder] [📁] [Output Folder]  [Load] │ [N] [Go]  7/13   │ ← Navigation
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
│        │                                      │  Annotation List         │
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

### 2.1 Navigation Bar (Top)

| Element | Description |
|---------|-------------|
| **Image Folder** | Path to images, type or browse with 📁 |
| **Output Folder** | Where annotations are saved |
| **Load** | Load images and resume from last position |
| **N / Go** | Jump to image number |

### 2.2 Tool Panel (Left)

| Tool | Shortcut | Cursor | Description |
|------|----------|--------|-------------|
| 🖱️ Click | `1` | Crosshair | Click object center, SAM auto-segments |
| ⬜ Box | `2` | Crosshair | Drag rectangle, SAM segments inside |
| ✋ Select | `3` | Arrow | Click/drag to select existing annotations |

### 2.3 Canvas (Center)

| Action | Effect |
|--------|--------|
| Scroll up/down | Zoom in/out (centered on cursor) |
| Right-click drag | Pan |
| Space + left-click drag | Pan |
| Middle-click drag | Pan |
| Double-click | Fit to window |
| Press `F` | Fit to window |
| Click ⊞ Fit button | Fit to window |

**Visual elements:**
- Colored outlines per class
- Labels with dark background
- Coordinates at bottom-left
- Dashed outline on hover
- Cyan highlight on selection
- Semi-transparent overlay during SAM inference

### 2.4 Control Panel (Right, scrollable)

#### Text Prompt
Enter object names (comma-separated), press Enter. SAM 3 finds all matching objects. Names auto-add as classes.

#### Class Management
- **Dropdown**: Select active class for annotation
- **+ button**: Add new class
- **− button**: Delete unused class

#### Settings

| Setting | Description |
|---------|-------------|
| Fallback to box | If SAM fails, create manual box annotation |
| Display mode | Outline / Mask / Both |
| Polygon simplify | Slider 0.001~0.020, lower = more precise |
| Overlap threshold | Slider 0%~50%, 0% = allow overlap |

#### Output Formats

| Format | Default | Description |
|--------|---------|-------------|
| OBB | ☑ | `labels/*.txt` — YOLO OBB training |
| YOLO-Seg | ☑ | `labels_seg/*.txt` — instance segmentation |
| PNG Mask | ☐ | `masks/*.png` — semantic segmentation |
| COCO JSON | ☐ | `coco_annotations.json` — COCO format |

#### Annotation List
- Color square per annotation matching class color
- Multi-select with Ctrl+click or Shift+click
- Apply class, delete, or clear all

---

## Chapter 3: Quick Start

1. **Launch**: `python main.py --model sam3.pt`
2. **Load images**: Enter folder paths, click Load
3. **Annotate**: Select class, click on objects (mode `1`)
4. **Next image**: Press `→` (auto-saves)
5. **Edit**: Press `3` to select, `Delete` to remove, apply class to fix mistakes

---

## Chapter 4: Three Annotation Methods

### Method 1: Click Segmentation
Best for clear, distinct objects. Press `1`, click object center.

### Method 2: Box Selection
Best for overlapping objects. Press `2`, drag a rectangle around the object.

### Method 3: Text Prompt
Best for batch annotation. Type object names in the text field, press Enter. SAM 3 finds all matches automatically. UI stays responsive during inference.

---

## Chapter 5: Editing Annotations

### Select
- Switch to Select tool (`3`)
- Click annotation to toggle selection
- Ctrl+click for multi-select
- Drag in empty area to box-select
- `Ctrl+A` to select all, `Esc` to deselect

### Delete
Select annotations, press `Delete` or click 🗑

### Change Class
Select annotations, choose class from dropdown, click "Apply Class"

---

## Chapter 6: Zoom & Navigation

| Action | Effect |
|--------|--------|
| Scroll up | Zoom in 12% |
| Scroll down | Zoom out 12% |
| Right-click drag | Pan |
| Space + left-click drag | Pan |
| Middle-click drag | Pan |
| `F` / Double-click / ⊞ button | Fit to window |

Zoom range: 5% ~ 3000%

---

## Chapter 7: Save & Output

### Auto-save
Switching images (← → keys, buttons, jump) triggers auto-save.

### Manual save
`Ctrl+S` or click 💾

### Output Structure

```
output_folder/
├── images/              ← Annotated image copies
├── labels/              ← OBB format (class x1 y1 x2 y2 x3 y3 x4 y4)
├── labels_seg/          ← YOLO-Seg polygon format
├── masks/               ← PNG masks (if enabled)
├── coco_annotations.json← COCO JSON (if enabled)
└── classes.txt          ← Class list
```

---

## Chapter 8: Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `←` | Previous image (auto-save) |
| `→` | Next image (auto-save) |
| `1` | Click mode |
| `2` | Box mode |
| `3` | Select mode |
| `Delete` | Delete selected |
| `Ctrl+S` | Save |
| `Ctrl+A` | Select all |
| `Esc` | Deselect all |
| `F` | Fit to window |
| `Enter` (text field) | Run text segmentation |
| `Enter` (jump field) | Jump to image |
| Scroll wheel | Zoom |
| Right-click drag | Pan |
| Space + left-click drag | Pan |
| Middle-click drag | Pan |
| Double-click | Fit to window |

---

## Chapter 9: FAQ

**Q: Click does nothing?**
Check tool is set to Click (`1`). Verify cursor is within image (coords shown at bottom-left). Try Box mode or enable "Fallback to box".

**Q: SAM inference takes long?**
First inference loads the model (10-30s). Subsequent runs take 1-3s. UI remains responsive.

**Q: Wrong class assigned?**
Press `3`, click the annotation, select correct class, click "Apply Class".

**Q: Lost annotations after crash?**
If you switched images at least once, previous annotations are saved. Reload the same folder to continue.

---

## Chapter 10: Glossary

| Term | Definition |
|------|------------|
| **OBB** | Oriented Bounding Box — rotated rectangle with 4 vertices |
| **Seg** | Segmentation — polygon vertices defining precise boundary |
| **Mask** | Binary image (0=background, 255=object) |
| **COCO** | Common Objects in Context — standard JSON annotation format |
| **SAM** | Segment Anything Model by Meta |
| **QPainter** | Qt framework's rendering engine for vector graphics |
