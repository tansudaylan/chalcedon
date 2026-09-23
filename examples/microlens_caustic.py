#!/usr/bin/env python3
"""Visualize Chalcedon's analytic lens deflection and magnification maps."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import chalcedon


def evaluate_lens_model(
    grid_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Evaluate a host-plus-subhalo lens on a normalized angular grid."""

    field_half_width = 2.0
    coordinate = np.linspace(-field_half_width, field_half_width, grid_size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="ij")
    lens_parameters = {
        "sherextr": 0.3,
        "sangextr": 0.3,
        "xposhost": 0.3,
        "yposhost": 0.3,
        "beinhost": 0.3,
        "ellphost": 0.3,
        "xpossubh": np.array([1.0]),
        "ypossubh": np.array([1.0]),
        "defssubh": np.array([1.0]),
        "ascasubh": np.array([1.0]),
        "acutsubh": np.array([1.0]),
    }
    output = chalcedon.retr_caustics(
        xposgrid=x_grid.ravel(),
        yposgrid=y_grid.ravel(),
        indxpixlelem=np.arange(grid_size**2),
        dictchalinpt=lens_parameters,
    )
    return x_grid, y_grid, lens_parameters, output


def run_example(output_path: Path) -> dict:
    """Plot the intermediate deflection and resulting magnification."""

    x_grid, y_grid, lens_parameters, output = evaluate_lens_model()
    grid_shape = x_grid.shape
    deflection = np.linalg.norm(output["defltotl"], axis=1).reshape(grid_shape)
    log_magnification = np.log10(output["magn"])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.7),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    deflection_image = axes[0].pcolormesh(
        x_grid,
        y_grid,
        deflection,
        shading="auto",
        cmap="viridis",
    )
    figure.colorbar(deflection_image, ax=axes[0], label="Deflection amplitude")
    axes[0].set_title("Intermediate deflection field")

    magnification_image = axes[1].pcolormesh(
        x_grid,
        y_grid,
        log_magnification,
        shading="auto",
        cmap="magma",
    )
    figure.colorbar(
        magnification_image,
        ax=axes[1],
        label=r"$\log_{10}$ magnification",
    )
    axes[1].set_title("Derived magnification map")

    coordinate = x_grid[:, 0]
    for contour in output["contours"]:
        contour_x = np.interp(contour[:, 0], np.arange(coordinate.size), coordinate)
        contour_y = np.interp(contour[:, 1], np.arange(coordinate.size), coordinate)
        axes[1].plot(contour_x, contour_y, color="white", linewidth=1.4)

    for axis in axes:
        axis.scatter(
            lens_parameters["xposhost"],
            lens_parameters["yposhost"],
            marker="x",
            s=65,
            linewidth=2.0,
            color="white",
            label="Host lens",
        )
        axis.scatter(
            lens_parameters["xpossubh"],
            lens_parameters["ypossubh"],
            marker="+",
            s=80,
            linewidth=2.0,
            color="#00D4FF",
            label="Subhalo",
        )
        axis.set_xlabel("Normalized angular x coordinate")
        axis.grid(False)
    axes[0].set_ylabel("Normalized angular y coordinate")
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0, loc="lower left")
    figure.suptitle("Analytic lens geometry maps deflection into magnification")
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Chalcedon's analytic deflection and magnification maps."
    )
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"microlens_caustic.{arguments.typefileplot}"
    )
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
