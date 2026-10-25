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


@pytest.mark.parametrize("phase", [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
def test_moving_lens_images_and_curves_obey_native_mapping(phase):
    from chalcedon.strong_lensing import LensModel

    result = microlens_caustic.evaluate_moving_lens(phase)
    lens = result["lens_position_arcsec"]
    parameters = [{"theta_E": 0.9, "center_x": lens[0], "center_y": lens[1]},
                  {"gamma1": 0.18, "gamma2": 0.04}]
    model = LensModel(["SIS", "SHEAR"])
    images = result["image_positions_arcsec"]
    mapped = np.column_stack(model.ray_shooting(images[:, 0], images[:, 1], parameters))
    np.testing.assert_allclose(mapped, np.broadcast_to(result["source_position_arcsec"], mapped.shape), atol=1e-7)
    assert len(images) in (2, 4)
    assert np.max(np.abs(images)) < 1.5
    assert np.max(np.abs(result["source_position_arcsec"])) < 0.55
    assert result["max_lens_equation_residual_arcsec"] < 1e-7
    assert len(result["curves"][0]) >= 1
    for critical_x, critical_y, caustic_x, caustic_y in zip(*result["curves"]):
        expected = model.ray_shooting(critical_x, critical_y, parameters)
        np.testing.assert_allclose(expected, (caustic_x, caustic_y), atol=1e-12)
        assert np.isfinite([critical_x, critical_y, caustic_x, caustic_y]).all()
        assert np.min(np.hypot(critical_x - lens[0], critical_y - lens[1])) > 0.5


@pytest.mark.parametrize("theme", ["white", "dark"])
def test_moving_lens_animation_has_loop_motion_and_exact_background(tmp_path, monkeypatch, capsys, theme):
    from PIL import Image
    import json

    monkeypatch.setenv("TDPY_VERBOSITY", "0")
    path = tmp_path / "moving.gif"
    metadata = microlens_caustic.run_moving_animation(path, frame_count=4, typeplotback=theme)
    assert capsys.readouterr().out == ""
    assert len(metadata["frames"]) == 4
    assert json.loads(path.with_suffix(".json").read_text())["source_sha256"] == metadata["source_sha256"]
    expected_background = (255, 255, 255) if theme == "white" else (0, 0, 0)
    with Image.open(path) as animation:
        assert animation.n_frames == 4 and animation.info["loop"] == 0
        frames = []
        for index in range(animation.n_frames):
            animation.seek(index)
            frame = np.asarray(animation.convert("RGB"))
            assert tuple(frame[0, 0]) == expected_background
            assert frame.std() > 10
            frames.append(frame.copy())
        assert all(np.any(first != second) for first, second in zip(frames, frames[1:]))
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        microlens_caustic.run_moving_animation(path, frame_count=4)
    assert path.read_bytes() == original


def test_moving_lens_animation_narrates_paths_when_enabled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TDPY_VERBOSITY", "1")
    path = tmp_path / "moving.gif"
    microlens_caustic.run_moving_animation(path, frame_count=2)
    output = capsys.readouterr().out
    assert f"Writing to {path}..." in output
    assert f"Writing to {path.with_suffix('.json')}..." in output
    assert f"Reading from {EXAMPLE_PATH}..." in output


def test_moving_lens_animation_legend_stays_outside_data(tmp_path, monkeypatch):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    draw = FigureCanvasAgg.draw
    checked = []

    def check(canvas):
        draw(canvas)
        if canvas.figure.legends:
            renderer = canvas.get_renderer()
            legend = canvas.figure.legends[0].get_window_extent(renderer)
            assert not any(legend.overlaps(axis.get_window_extent(renderer)) for axis in canvas.figure.axes)
            assert not any(legend.overlaps(axis.xaxis.label.get_window_extent(renderer)) for axis in canvas.figure.axes)
            assert canvas.figure.bbox.contains(legend.x0, legend.y0)
            assert canvas.figure.bbox.contains(legend.x1, legend.y1)
            checked.append(True)

    monkeypatch.setattr(FigureCanvasAgg, "draw", check)
    microlens_caustic.run_moving_animation(tmp_path / "moving.gif", frame_count=2)
    assert len(checked) >= 2


@pytest.mark.parametrize("count", [0, 1, 1.5, True])
def test_moving_lens_animation_rejects_invalid_frame_count(tmp_path, count):
    with pytest.raises(ValueError, match="at least two"):
        microlens_caustic.run_moving_animation(tmp_path / "invalid.gif", frame_count=count)


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