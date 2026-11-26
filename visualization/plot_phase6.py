"""Plot the Phase 6 paired height and coverage results from the summary CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        data = {(r["reference"], int(r["beams"]), r["distance_m"], r["method"]): r
                for r in csv.DictReader(stream)}

    def values(reference, method, field, distance="all", scale=1):
        return [float(data[reference, beams, distance, method][field]) * scale for beams in (64, 32, 16)]

    blue, yellow, teal = "#196697", "#b76f24", "#4e7d87"
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#142d3d", "axes.labelcolor": "#142d3d",
                         "xtick.color": "#325063", "ytick.color": "#325063",
                         "axes.edgecolor": "#9db0b7"})
    fig, axes = plt.subplots(1, 3, figsize=(14.7, 4.78))
    fig.subplots_adjust(left=.064, right=.988, bottom=.255, top=.77, wspace=.40)
    fig.text(.064, .925, "RoadSurf-3D  |  Phase 6 acceptance", fontsize=19, fontweight="bold")
    fig.text(.064, .863, "KITTI DC holdout and synthetic geometry gaps", fontsize=11, color="#456271")
    x = np.arange(3)
    for i, (method, label, color) in enumerate((
            ("B0", "Plane", "#a8b7bf"), ("B1", "LiDAR spline", teal),
            ("A0", "Multimodal spline", "#77bfa5"), ("A4", "SGSR", blue))):
        err = values("dc_non_input", method, "common_height_mae_m", scale=100)
        axes[0].bar(x + (i - 1.5) * .19, err, width=.17, label=label, color=color, zorder=3)
    axes[0].set_title("Native elevation error", fontweight="bold", pad=11)
    axes[0].set_ylabel("Common-coverage MAE (cm)")
    for i, (size, label, color) in enumerate((("2x4", "2 × 4 m", blue), ("3x6", "3 × 6 m", yellow))):
        coverage = values(f"hole_{size}_heldout_lidar_core", "A4", "coverage_pct")
        bars = axes[1].bar(x + (i - .5) * .32, coverage, width=.25, label=label, color=color, zorder=3)
        axes[1].bar_label(bars, labels=[f"{v:.0f}%" for v in coverage], padding=3, fontsize=9)
    axes[1].set_title("Synthetic hole-core coverage", fontweight="bold", pad=11)
    axes[1].set_ylim(0, 113)
    axes[1].set_ylabel("Coverage of held-out LiDAR")
    for distance, color, marker in (("5-20", blue, "o"), ("20-40", teal, "s"), ("40-70", yellow, "^")):
        coverage = values("dc_non_input", "A4", "coverage_pct", distance=distance)
        axes[2].plot(x, coverage, color=color, label=distance.replace("-", "–") + " m", marker=marker,
                     markersize=6, linewidth=2.3, zorder=3)
    axes[2].set_title("SGSR coverage by distance", fontweight="bold", pad=11)
    axes[2].set_ylim(0, 106)
    axes[2].set_ylabel("Coverage of DC reference")
    for ax in axes[1:]:
        ax.yaxis.set_major_formatter(PercentFormatter(100))
    for ax, columns in zip(axes, (2, 2, 3)):
        ax.set_xticks(x, ["64", "32", "16"])
        ax.set_xlabel("LiDAR beams")
        ax.grid(axis="y", color="#dce7ec", linewidth=.8, zorder=0)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.26), ncol=columns, frameon=False, fontsize=9)
    fig.savefig(args.out, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=.16)


if __name__ == "__main__":
    main()
