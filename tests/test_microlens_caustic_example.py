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


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_truncation_mass_example_writes_nonblank_monotonic_curve(tmp_path, typeplotback, inherit):
    from tdpy.plotting import plot_style

    output_path = tmp_path / "truncation_mass.png"

    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        radius_ratio, mass_ratio = truncation_mass.run_example(output_path, **options)

    image = mpimg.imread(output_path)
    assert radius_ratio.shape == mass_ratio.shape == (500,)
    assert np.isfinite(mass_ratio).all()
    assert np.all(np.diff(mass_ratio) > 0)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    np.testing.assert_array_equal(image[0, 0], (0, 0, 0, 1) if typeplotback == 'dark' else (1, 1, 1, 1))


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
def test_truncation_mass_example_writes_bokeh_html(tmp_path, monkeypatch, typeplotback):
    pytest.importorskip("bokeh")
    from bokeh.models import Axis, Plot
    import tdpy.plotting as plotting

    save = plotting.save_bokeh_html

    def capture_and_save(figure, *args, **kwargs):
        result = save(figure, *args, **kwargs)
        background, foreground = plotting.plot_background_colors(typeplotback)
        assert all(plot.background_fill_color == background for plot in figure.select({'type': Plot}))
        assert all(axis.major_label_text_color == foreground for axis in figure.select({'type': Axis}))
        return result

    monkeypatch.setattr(plotting, 'save_bokeh_html', capture_and_save)
    output_path = truncation_mass.run_bokeh_example(tmp_path / "truncation_mass.html", typeplotback=typeplotback)

    assert output_path.is_file()
    assert "Bokeh" in output_path.read_text(encoding="utf-8")


@pytest.mark.parametrize('typeplotback', ['white', 'dark'])
@pytest.mark.parametrize('inherit', [False, True])
def test_microlens_example_writes_finite_pipeline_maps(tmp_path, typeplotback, inherit):
    from tdpy.plotting import plot_style

    output_path = tmp_path / "microlens_caustic.png"

    options = {} if inherit else {'typeplotback': typeplotback}
    outer_theme = typeplotback if inherit else ('dark' if typeplotback == 'white' else 'white')
    with plot_style(outer_theme):
        original = dict(microlens_caustic.plt.rcParams)
        output = microlens_caustic.run_example(output_path, **options)
        assert dict(microlens_caustic.plt.rcParams) == original

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert output["magn"].shape == (100, 100)
    assert output["defltotl"].shape == (10000, 2)
    assert np.isfinite(output["magn"]).all()
    assert len(output["contours"]) >= 1
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8
    np.testing.assert_array_equal(image[0, 0], (0, 0, 0, 1) if typeplotback == 'dark' else (1, 1, 1, 1))