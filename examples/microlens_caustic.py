#!/usr/bin/env python3
"""Visualize Chalcedon's analytic lens deflection and magnification maps."""

from tdpy.verbosity import print

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

import chalcedon
from chalcedon.strong_lensing import LensEquationSolver, LensModel, LensModelExtensions
from tdpy.cli import add_plot_arguments
from tdpy.plotting import plot_background_colors, save_plot, themed_plot


def evaluate_moving_lens(phase: float, grid_scale: float = 0.04) -> dict:
    """Evaluate a synthetic moving SIS lens with external shear in angular coordinates."""
    if not np.isfinite(phase) or not np.isfinite(grid_scale) or not 0 < grid_scale < 0.2:
        raise ValueError("phase must be finite and grid_scale must lie between zero and 0.2 arcsec")
    lens_position = np.array([0.15 * np.sin(phase), 0.08 * np.cos(phase)])  # [arcsec]
    source_position = np.array([0.30 * np.cos(phase), 0.13 * np.sin(phase)])  # [arcsec]
    model = LensModel(["SIS", "SHEAR"])
    parameters = [
        {"theta_E": 0.9, "center_x": lens_position[0], "center_y": lens_position[1]},  # [arcsec]
        {"gamma1": 0.18, "gamma2": 0.04},
    ]
    images = LensEquationSolver(model).image_position_from_source(
        *source_position, parameters, search_window=4.0, min_distance=0.002,
    )
    image_positions = np.column_stack(images)  # [arcsec]
    singular_radius = 1e-6  # [arcsec]
    image_positions = image_positions[
        np.linalg.norm(image_positions - lens_position, axis=1) > singular_radius
    ]
    if not len(image_positions):
        raise RuntimeError("The moving lens produced no resolved images")
    mapped_sources = np.column_stack(model.ray_shooting(
        image_positions[:, 0], image_positions[:, 1], parameters,
    ))
    residual = np.linalg.norm(mapped_sources - source_position, axis=1)  # [arcsec]
    if not np.isfinite(residual).all() or residual.max() > 1e-7:
        raise RuntimeError("A moving-lens image fails the lens equation")
    grid_side = max(20, 2 * int(np.ceil(2.0 / grid_scale)))
    window = (grid_side - 1) * grid_scale  # [arcsec]
    curves = LensModelExtensions(model).critical_curve_caustics(
        parameters, compute_window=window, grid_scale=grid_scale,
        center_x=lens_position[0], center_y=lens_position[1],
    )
    return {"lens_position_arcsec": lens_position, "source_position_arcsec": source_position,
            "image_positions_arcsec": image_positions, "curves": curves,
            "max_lens_equation_residual_arcsec": float(residual.max())}


@themed_plot
def run_moving_animation(output_path: Path, *, frame_count: int = 48,
                         duration_ms: int = 80, grid_scale: float = 0.04) -> dict:
    """Animate native image solutions and critical curves for prescribed angular tracks."""
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix(".json")
    if output_path.suffix != ".gif":
        raise ValueError("The moving-lens animation requires a .gif output")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 2:
        raise ValueError("frame_count must be an integer of at least two")
    if not isinstance(duration_ms, int) or duration_ms <= 0:
        raise ValueError("duration_ms must be a positive integer")
    if output_path.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite moving-lens output {output_path}")
    background, foreground = plot_background_colors()
    phases = np.linspace(0, 2 * np.pi, frame_count, endpoint=False)  # [rad]
    frames = []
    rows = []
    background_rgb = np.array(matplotlib.colors.to_rgb(background)) * 255
    background_rgb = np.rint(background_rgb).astype(np.uint8)
    with plt.rc_context({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10}):
        figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), dpi=150, facecolor=background)
        FigureCanvasAgg(figure)
        try:
            for frame_index, phase in enumerate(phases):
                result = evaluate_moving_lens(float(phase), grid_scale)
                lens = result["lens_position_arcsec"]
                source = result["source_position_arcsec"]
                images = result["image_positions_arcsec"]
                for axis in axes:
                    axis.clear()
                    axis.set_facecolor(background)
                    axis.set_aspect("equal")
                    axis.grid(False)
                    axis.set_xlabel("x [arcsec]")
                    axis.set_ylabel("y [arcsec]")
                for curve_index, (critical_x, critical_y, caustic_x, caustic_y) in enumerate(zip(*result["curves"])):
                    axes[0].plot(critical_x, critical_y, color="#4477AA", linewidth=1.5,
                                 label="Critical curve" if curve_index == 0 else None)
                    axes[1].plot(caustic_x, caustic_y, color="#CC6677", linewidth=1.5,
                                 label="Caustic" if curve_index == 0 else None)
                axes[0].scatter(*lens, marker="+", s=85, color=foreground, label="Lens")
                axes[0].scatter(images[:, 0], images[:, 1], s=38, color="#E69F00",
                                edgecolors=foreground, linewidths=0.5, label="Images")
                axes[1].scatter(*source, marker="*", s=100, color="#009E73",
                                edgecolors=foreground, linewidths=0.5, label="Source")
                axes[0].set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), title="Image plane")
                axes[1].set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), title="Source plane")
                handles, labels = [], []
                for axis in axes:
                    axis_handles, axis_labels = axis.get_legend_handles_labels()
                    handles.extend(axis_handles)
                    labels.extend(axis_labels)
                for legend in list(figure.legends):
                    legend.remove()
                figure.legend(handles, labels, loc="lower center", ncol=5, frameon=True,
                              fancybox=True, framealpha=1, facecolor=background, edgecolor=foreground)
                figure.suptitle(f"Synthetic moving lens    Frame {frame_index + 1:03d}/{frame_count:03d}", fontsize=10)
                figure.subplots_adjust(left=0.08, right=0.98, bottom=0.25, top=0.86, wspace=0.32)
                figure.canvas.draw()
                rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
                palette_frame = Image.fromarray(rgb).quantize(colors=255)
                palette = palette_frame.getpalette()
                palette.extend([0] * (768 - len(palette)))
                palette[765:768] = background_rgb.tolist()
                palette_pixels = np.asarray(palette_frame).copy()
                palette_pixels[np.all(rgb == background_rgb, axis=-1)] = 255
                palette_frame = Image.fromarray(palette_pixels)
                palette_frame.putpalette(palette)
                frames.append(palette_frame)
                rows.append({"phase_rad": float(phase), "lens_position_arcsec": lens.tolist(),
                             "source_position_arcsec": source.tolist(), "image_positions_arcsec": images.tolist(),
                             "max_lens_equation_residual_arcsec": result["max_lens_equation_residual_arcsec"]})
        finally:
            plt.close(figure)
    print(f"Reading from {Path(__file__)}...")
    metadata = {"profiles": ["SIS", "SHEAR"], "theta_E_arcsec": 0.9,
                "shear": [0.18, 0.04], "grid_scale_arcsec": grid_scale,
                "frame_duration_ms": duration_ms, "background": background,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "frames": rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=duration_ms, disposal=2, optimize=False, background=255)
    print(f"Writing to {metadata_path}...")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


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


@themed_plot
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
    save_plot(
        figure,
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output


def parse_arguments():
    parser = add_plot_arguments(argparse.ArgumentParser(
        description="Plot Chalcedon's analytic deflection and magnification maps."))
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--animation-output', type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    visuals_directory = Path(__file__).parent / "visuals"
    if arguments.animate:
        tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = arguments.animation_output or visuals_directory / f"moving_lens_{tag}.gif"
        run_moving_animation(output, frame_count=arguments.frames, typeplotback=arguments.typeplotback)
        return 0
    output_path = visuals_directory / f"microlens_caustic.{arguments.typefileplot}"
    run_example(output_path, typeplotback=arguments.typeplotback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
