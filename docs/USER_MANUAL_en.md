# Fish Labeler User Manual

## Introduction

Fish Labeler is intended for annotating fish, catch, crew, and vessel context in imagery collected aboard fishing vessels. Its PyQt6 interface supports responsive zoom/pan, hover feedback, and background SAM inference.

**What can this tool do?**

- Annotate objects using three methods: click, box-select, or text prompt
- Auto-detect object boundaries via Meta SAM 3
- Export in YOLO-Seg / PNG Mask formats
- Zoom with scroll wheel, pan with right-click drag — ideal for high-res images

---

## Chapter 1: Installation

### 1.1 System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Windows 10 / Ubuntu 20.04 / macOS 12+ | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| Python | 3.12 | 3.12 |
| GPU | NVIDIA GTX 1060 (6GB) / Apple M1 | NVIDIA RTX 3060+ (8GB+) / Apple M2+ |
| CUDA | 11.7 (macOS uses MPS, no CUDA needed) | 12.1+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 5 GB (incl. model) | 10 GB+ |

> **Cross-platform:** The application automatically detects the best compute device (CUDA GPU → Apple MPS → CPU). No manual configuration needed.

### 1.2 Installation

```bash
# 1. Install PyTorch (choose your CUDA version)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. Install locked dependencies
uv sync --locked --all-extras --dev
```

### 1.3 Download SAM 3 Model

Download `sam3.pt` from [Hugging Face](https://huggingface.co/facebook/sam3) (requires Meta approval) and place it at `src/models/sam3.pt`.

### 1.4 Launch

```bash
uv run fish-labeler app
uv run fish-labeler app --images /data/vessel-trip-01/images --output vessel-trip-01
uv run fish-labeler app --model /path/to/sam3.pt
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
│        │                                      │  ☑Seg ☐Mask           │
│        │                                      │                          │
│        │                                      │  Annotation List         │
│        │                                      │  ■ 1. tuna              │
│        │                                      │  ■ 2. swordfish         │
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
| **Output Folder** | Run name created below the repository `output/` directory |
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
| YOLO-Seg | ☑ | `labels/*.txt` — instance segmentation |
| PNG Mask | ☐ | `masks/*.png` — semantic segmentation |

#### Annotation List
- Color square per annotation matching class color
- Multi-select with Ctrl+click or Shift+click
- Apply class, delete, or clear all

---

## Chapter 3: Quick Start

1. **Launch**: `uv run fish-labeler app`
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
output/<run-name>/
├── images/              ← Annotated image copies
├── labels/              ← YOLO-Seg polygon format
├── masks/               ← PNG masks (if enabled)
└── classes.txt          ← Class list
```

Input images may be outside the repository or reached through a symbolic link. Output is always written to `fish-labeler/output/<run-name>/`; entering an absolute output path uses only its final directory name.

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
| **Seg** | Segmentation — polygon vertices defining precise boundary |
| **Mask** | Binary image (0=background, 255=object) |
| **SAM** | Segment Anything Model by Meta |
| **QPainter** | Qt framework's rendering engine for vector graphics |
