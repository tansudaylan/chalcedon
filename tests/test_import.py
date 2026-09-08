import importlib


def test_import_chalcedon_package():
    chalcedon = importlib.import_module('chalcedon')
    assert hasattr(chalcedon, '__file__')


def test_import_chalcedon_main_module():
    main = importlib.import_module('chalcedon.main')
    assert hasattr(main, 'retr_radieins_inft')
