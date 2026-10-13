import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "microlens_caustic.py"
SPEC = importlib.util.spec_from_file_location("microlens_caustic", EXAMPLE_PATH)
microlens_caustic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(microlens_caustic)


TRUNCATION_PATH = EXAMPLE_PATH.with_name("plot_truncation_mass.py")
TRUNCATION_SPEC = importlib.util.spec_from_file_location("plot_truncation_mass", TRUNCATION_PATH)
truncation_mass = importlib.util.module_from_spec(TRUNCATION_SPEC)
TRUNCATION_SPEC.loader.exec_module(truncation_mass)


def test_truncation_mass_example_writes_nonblank_monotonic_curve(tmp_path):
    output_path = tmp_path / "truncation_mass.png"

    radius_ratio, mass_ratio = truncation_mass.run_example(output_path)

    image = mpimg.imread(output_path)
    assert radius_ratio.shape == mass_ratio.shape == (500,)
    assert np.isfinite(mass_ratio).all()
    assert np.all(np.diff(mass_ratio) > 0)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8


def test_truncation_mass_example_writes_bokeh_html(tmp_path):
    pytest.importorskip("bokeh")
    output_path = truncation_mass.run_bokeh_example(tmp_path / "truncation_mass.html")

    assert output_path.is_file()
    assert "Bokeh" in output_path.read_text(encoding="utf-8")


def test_microlens_example_writes_finite_pipeline_maps(tmp_path):
    output_path = tmp_path / "microlens_caustic.png"

    output = microlens_caustic.run_example(output_path)

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert output["magn"].shape == (100, 100)
    assert output["defltotl"].shape == (10000, 2)
    assert np.isfinite(output["magn"]).all()
    assert len(output["contours"]) >= 1
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8