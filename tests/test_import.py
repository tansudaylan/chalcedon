import importlib
import importlib.util
import sys
from pathlib import Path


def test_import_chalcedon_package():
    chalcedon = importlib.import_module('chalcedon')
    assert hasattr(chalcedon, '__file__')


def test_import_chalcedon_main_module():
    main = importlib.import_module('chalcedon.main')
    assert hasattr(main, 'retr_radieins_inft')


def test_examples_module_is_import_safe(monkeypatch):
    examples_path = Path(__file__).resolve().parents[1] / 'examples' / 'examples.py'
    monkeypatch.setattr(sys, 'argv', ['examples.py', 'cnfg_microlens'])

    spec = importlib.util.spec_from_file_location('chalcedon_examples_test', examples_path)
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert hasattr(module, 'cnfg_microlens')
