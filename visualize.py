# %%
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, patheffects

# Configuration
parser = argparse.ArgumentParser(description="Visualize a labeled fish image")
parser.add_argument("run_name", help="Run directory name below output/")
parser.add_argument("image_name", help="Image filename stem")
args = parser.parse_args()

project_root = Path(__file__).resolve().parent
run_dir = project_root / "output" / Path(args.run_name).name
image_name = args.image_name
image_dir = run_dir / "images"
labels_seg_dir = run_dir / "labels_seg"
classes_file = run_dir / "classes.txt"

# Read classes
classes = []
if classes_file.exists():
    with open(classes_file, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

# Find the image file
image_path = None
for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
    p = image_dir / f"{image_name}{ext}"
    if p.exists():
        image_path = p
        break

if not image_path:
    print(f"Error: Could not find image for {image_name} in {image_dir}")
else:
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Find the label file
    label_path = labels_seg_dir / f"{image_name}.txt"

    # Initialize plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(f"Image: {image_name}\nResolution: {w}x{h}")

    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()

        colors = plt.get_cmap("tab10", len(classes) if classes else 10)

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            cls_id = int(parts[0])
            coords = np.array([float(x) for x in parts[1:]])

            # Reshape coordinates to (N, 2)
            pts_norm = coords.reshape(-1, 2)
            pts_abs = pts_norm * np.array([w, h])

            cls_name = classes[cls_id] if cls_id < len(classes) else f"Class {cls_id}"
            color = colors(cls_id % 10)

            # Plot Polygon (Mask outline)
            polygon = patches.Polygon(
                pts_abs, linewidth=2, edgecolor=color, facecolor=(*color[:3], 0.3)
            )
            ax.add_patch(polygon)

            # Compute bounding box
            x_min, y_min = pts_abs.min(axis=0)
            x_max, y_max = pts_abs.max(axis=0)
            box_w = x_max - x_min
            box_h = y_max - y_min

            # Plot Bounding Box
            rect = patches.Rectangle(
                (x_min, y_min),
                box_w,
                box_h,
                linewidth=1,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)

            # Add text label
            txt = ax.text(
                x_min,
                y_min - 5,
                cls_name,
                color="white",
                fontsize=10,
                fontweight="bold",
                backgroundcolor=color,
            )
            txt.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])

        print(f"Loaded {len(lines)} labels for {image_name}.")
    else:
        print(f"Warning: No label file found at {label_path}")

    plt.tight_layout()
    plt.show()

# %%
