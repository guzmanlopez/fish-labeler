"""Command-line entry point for video preparation and Qt fish labeling."""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel

from core.sam_engine import DEFAULT_MODEL_PATH

CONSOLE = Console()


def _run_app(args: argparse.Namespace) -> int:
    """Launch the Qt annotation workflow with an optional input sequence."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from ui.main_window import MainWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(sam_model_path=args.model)
    if args.images:
        window.folder_input.setText(args.images)
    if args.output:
        window.output_input.setText(args.output)
    window.showMaximized()
    CONSOLE.print(
        Panel("Qt annotation workflow started", title="Fish Labeler", border_style="cyan")
    )
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    """Dispatch the video preparation or Qt annotation workflow."""
    arguments = argv if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "video":
        from core.sam3_video_to_yolo import main as video_main

        CONSOLE.print(
            Panel(
                "Preparing a SAM 3 dataset below output/<run-name>.",
                title="Fish Labeler",
                border_style="cyan",
            )
        )
        return video_main(arguments[1:])

    parser = argparse.ArgumentParser(description="Prepare and label fishing-vessel imagery")
    workflows = parser.add_subparsers(dest="workflow", required=True)

    app_parser = workflows.add_parser("app", help="Launch the Qt annotation application")
    app_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="SAM 3 model path")
    app_parser.add_argument("--images", help="Image folder path")
    app_parser.add_argument("--output", help="Run name below the repository output directory")

    workflows.add_parser("video", help="Prepare a video dataset below output/<run-name>")
    args = parser.parse_args(arguments)
    return _run_app(args)


if __name__ == "__main__":
    raise SystemExit(main())
