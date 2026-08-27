"""Tests for command-line workflow interruption handling."""

import argparse
import sys
from types import ModuleType
from typing import Any, cast

import main


def test_qt_workflow_saves_current_annotations_on_interrupt(monkeypatch):
    """KeyboardInterrupt from Qt should save the active frame and return 130."""
    created_windows = []

    class FakeApplication:
        @staticmethod
        def setHighDpiScaleFactorRoundingPolicy(policy):
            pass

        def __init__(self, argv):
            pass

        def setStyle(self, style):
            pass

        def exec(self):
            raise KeyboardInterrupt

    class FakeQt:
        class HighDpiScaleFactorRoundingPolicy:
            PassThrough = object()

    class FakeInput:
        def setText(self, value):
            self.value = value

    class FakeWindow:
        def __init__(self, sam_model_path):
            self.folder_input = FakeInput()
            self.output_input = FakeInput()
            self.save_calls = 0
            created_windows.append(self)

        def showMaximized(self):
            pass

        def _save_labels(self):
            self.save_calls += 1

    qt_core = ModuleType("PyQt6.QtCore")
    cast(Any, qt_core).Qt = FakeQt
    qt_widgets = ModuleType("PyQt6.QtWidgets")
    cast(Any, qt_widgets).QApplication = FakeApplication
    main_window = ModuleType("ui.main_window")
    cast(Any, main_window).MainWindow = FakeWindow
    monkeypatch.setitem(sys.modules, "PyQt6.QtCore", qt_core)
    monkeypatch.setitem(sys.modules, "PyQt6.QtWidgets", qt_widgets)
    monkeypatch.setitem(sys.modules, "ui.main_window", main_window)

    exit_code = main._run_app(argparse.Namespace(model="model.pt", images=None, output=None))

    assert exit_code == 130
    assert created_windows[0].save_calls == 1


def test_main_sam3video_subcommand_passes_registered_video_arguments(monkeypatch):
    """The main CLI should parse SAM3 video parameters before starting the workflow."""
    received = {}

    def fake_run_workflow(args):
        received.update(vars(args))
        return 0

    monkeypatch.setattr("segmentation.sam3_video_to_yolo.run_workflow", fake_run_workflow)

    exit_code = main.main([
        "sam3video",
        "--video",
        "source.mp4",
        "--output-dir",
        "run",
        "--frame-step",
        "3",
    ])

    assert exit_code == 0
    assert received["video"].name == "source.mp4"
    assert received["output_dir"].name == "run"
    assert received["frame_step"] == 3
