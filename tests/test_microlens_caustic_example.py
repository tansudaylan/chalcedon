import importlib.util
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np


EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "microlens_caustic.py"
SPEC = importlib.util.spec_from_file_location("microlens_caustic", EXAMPLE_PATH)
microlens_caustic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(microlens_caustic)


def test_microlens_example_writes_finite_pipeline_maps(tmp_path):
    output_path = tmp_path / "microlens_caustic.png"

    output = microlens_caustic.run_example(output_path)

    image = mpimg.imread(output_path)
    assert output_path.is_file()
    assert output["magn"].shape == (100, 100)
    assert output["defltotl"].shape == (10000, 2)
    assert np.isfinite(output["magn"]).all()
    assert len(output["contours"]) == 4
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert image[..., :3].min() < 0.8