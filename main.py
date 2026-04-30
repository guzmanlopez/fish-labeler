"""
SAM3 Labeler — PyQt6 Edition
Uses QPainter vector rendering instead of Gradio server-side redraw,
achieving millisecond-level interactive feedback.

Usage:
    python main.py
    python main.py --model /path/to/sam3.pt
"""

import argparse
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="Fish Labeler (PyQt6)")
    parser.add_argument("--model", default="sam3.pt", help="SAM 3 model path")
    parser.add_argument("--images", default=None, help="Image folder path")
    parser.add_argument("--output", default=None, help="Output folder path")
    args = parser.parse_args()

    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Cross-platform consistent appearance

    window = MainWindow(sam_model_path=args.model)

    # If paths are specified via command line, fill in the UI
    if args.images:
        window.folder_input.setText(args.images)
    if args.output:
        window.output_input.setText(args.output)

    window.showMaximized()

    from core.logger import logger

    logger.info("=" * 90)
    logger.info("Fish Labeler — PyQt6 Edition")
    logger.info("Shortcuts: Left/Right switch images | Del delete | Ctrl+S save")
    logger.info("Scroll to zoom | Real-time hover highlight")
    logger.info("=" * 90)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
