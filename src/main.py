"""Command-line entry point for video preparation and Qt fish labeling."""

import argparse
import signal
import sys

from rich.console import Console
from rich.panel import Panel

from segmentation.sam_engine import DEFAULT_MODEL_PATH

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
    interrupted = False

    def _handle_sigint(signum, frame):
        """Leave the Qt event loop so annotations can be saved outside paint handlers."""
        nonlocal interrupted
        interrupted = True
        app.quit()

    previous_sigint_handler = signal.signal(signal.SIGINT, _handle_sigint)
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        interrupted = True
        exit_code = 0
    finally:
        signal.signal(signal.SIGINT, previous_sigint_handler)

    if interrupted:
        CONSOLE.print("[yellow]Interrupt received. Saving the current Qt annotations...[/yellow]")
        try:
            window._save_labels()
        except Exception as exc:  # noqa: BLE001
            CONSOLE.print(f"[red]Could not save current annotations: {exc}[/red]")
        else:
            CONSOLE.print("[green]Current Qt annotations saved. Exiting.[/green]")
        return 130
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Dispatch the video preparation or Qt annotation workflow."""
    arguments = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Prepare and label fishing-vessel imagery")
    workflows = parser.add_subparsers(dest="workflow", required=True)

    app_parser = workflows.add_parser("app", help="Launch the Qt annotation application")
    app_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="SAM 3 model path")
    app_parser.add_argument("--images", help="Image folder path")
    app_parser.add_argument("--output", help="Run name below the repository output directory")

    video_parser = workflows.add_parser(
        "sam3video", help="Prepare a SAM3 video dataset below output/<run-name>"
    )
    from segmentation.sam3_video_to_yolo import add_video_arguments, run_workflow

    add_video_arguments(video_parser)
    args = parser.parse_args(arguments)
    if args.workflow == "sam3video":
        CONSOLE.print(
            Panel(
                "Preparing a SAM 3 dataset below output/<run-name>.",
                title="Fish Labeler",
                border_style="cyan",
            )
        )
        return run_workflow(args)
    return _run_app(args)


if __name__ == "__main__":
    raise SystemExit(main())
