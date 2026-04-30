import json
from pathlib import Path
import core.io_manager as io_manager
from core.io_manager import load_config, save_config, persist_classes, load_persisted_classes

def test_load_save_config(tmp_path):
    """Test saving and loading config."""
    io_manager.CONFIG_FILE = tmp_path / "sam3_config.json"
    save_config("test_in", "test_out")
    assert io_manager.CONFIG_FILE.exists()
    config = load_config()
    assert config["images_folder"] == "test_in"
    assert config["output_folder"] == "test_out"

def test_persist_classes(tmp_path):
    """Test persisting classes."""
    io_manager.CLASSES_STORE = tmp_path / "sam3_classes.txt"
    persist_classes(["fish", "turtle"])
    classes = load_persisted_classes()
    assert classes == ["fish", "turtle"]
