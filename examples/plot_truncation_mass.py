#!/usr/bin/env python3
"""Plot the analytic truncated-lens mass kernel."""

from tdpy.verbosity import print

import argparse
from tdpy.cli import add_plot_arguments
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chalcedon import retr_mcutfrommscl
from tdpy.plotting import plot_background_colors, save_plot, themed_plot


@themed_plot
def run_example(output_path):
    """Generate the dimensionless truncation-mass figure."""

    radius_ratio = np.logspace(-2.0, 2.0, 500)
    mass_ratio = retr_mcutfrommscl(radius_ratio)

    figure, axis = plt.subplots(figsize=(6.5, 4.0), facecolor="white")
    axis.plot(radius_ratio, mass_ratio, color="#007A6F", linewidth=2.0)
    axis.scatter([1.0], [retr_mcutfrommscl(1.0)], color="#A51417", s=35, zorder=3)
    axis.annotate(
        r"$a_{cut}=a_s$",
        xy=(1.0, retr_mcutfrommscl(1.0)),
        xytext=(1.6, 0.18),
        arrowprops={"arrowstyle": "->", "color": plot_background_colors()[1]},
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"Cutoff-to-scale-radius ratio $a_{cut}/a_s$")
    axis.set_ylabel(r"Truncation-mass ratio $M_{cut}/M_s$")
    axis.grid(False)
    figure.tight_layout()

    print(f"Writing to {output_path}...")
    save_plot(figure, output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return radius_ratio, mass_ratio


@themed_plot
def run_bokeh_example(output_path):
    """Write an interactive version of the truncation-mass curve."""

    from bokeh.models import HoverTool
    from bokeh.plotting import figure

    from tdpy.plotting import save_bokeh_html

    radius_ratio = np.logspace(-2.0, 2.0, 500)
    mass_ratio = retr_mcutfrommscl(radius_ratio)
    plot = figure(
        title="Truncated-halo mass",
        x_axis_type="log",
        y_axis_type="log",
        x_axis_label="Cutoff-to-scale-radius ratio",
        y_axis_label="Truncation-mass ratio",
        width=700,
        height=440,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    plot.line(radius_ratio, mass_ratio, line_width=2.2, color="#007A6F")
    plot.scatter(
        [1.0],
        [retr_mcutfrommscl(1.0)],
        size=10,
        color="#A51417",
        legend_label="Equal cutoff and scale radii",
    )
    plot.add_tools(
        HoverTool(tooltips=[("Radius ratio", "$x"), ("Mass ratio", "$y")])
    )
    plot.grid.visible = False
    plot.legend.location = "top_left"
    return save_bokeh_html(plot, output_path, title="Truncated-halo mass")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_arguments(parser)
    parser.add_argument('--typeplotback', choices=('white', 'dark'), default='white')
    parser.add_argument(
        "--bokeh",
        action="store_true",
        help="also write an interactive HTML version",
    )
    arguments = parser.parse_args()
    output_path = Path(__file__).with_name(f"truncation_mass.{arguments.typefileplot}")
    run_example(output_path, typeplotback=arguments.typeplotback)
    if arguments.bokeh:
        run_bokeh_example(Path(__file__).with_name("truncation_mass.html"), typeplotback=arguments.typeplotback)


if __name__ == "__main__":
    main()