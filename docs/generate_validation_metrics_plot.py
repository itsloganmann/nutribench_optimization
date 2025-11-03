"""Create visualization of validation metrics for NutriBench prompt optimization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "validation_evaluation.json"
OUTPUT_PATH = ROOT / "slides" / "validation_metrics.png"


def load_metrics() -> dict:
    with RESULTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["metrics"]


def make_plot(metrics: dict) -> None:
    output_dir = OUTPUT_PATH.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=150, constrained_layout=True)
    fig.suptitle("Validation Performance (1,000 Meals)", fontsize=16, color="#333333", fontweight="bold")

    error_labels = ["MAE", "RMSE"]
    error_values = [metrics["mae"], metrics["rmse"]]
    error_colors = ["#009688", "#455A64"]

    ax_error = axes[0]
    bars = ax_error.bar(error_labels, error_values, color=error_colors)
    ax_error.set_ylabel("Error (grams)", color="#333333")
    ax_error.set_ylim(0, max(error_values) * 1.15)
    ax_error.set_title("Error Magnitude", fontsize=12, color="#333333", pad=12)
    ax_error.tick_params(colors="#333333")
    ax_error.spines["top"].set_visible(False)
    ax_error.spines["right"].set_visible(False)
    for bar, value in zip(bars, error_values):
        ax_error.text(bar.get_x() + bar.get_width() / 2, value + (max(error_values) * 0.03), f"{value:.2f}",
                      ha="center", va="bottom", fontsize=9, color="#333333")

    signal_labels = ["Accuracy within 7.5 g", "Correlation"]
    signal_values = [metrics["acc_within_7_5"] * 100, metrics["corr"] * 100]
    signal_colors = ["#80CBC4", "#B0BEC5"]

    ax_signal = axes[1]
    bars = ax_signal.bar(signal_labels, signal_values, color=signal_colors)
    ax_signal.set_ylabel("Percentage (%)", color="#333333")
    ax_signal.set_ylim(0, max(signal_values) * 1.2)
    ax_signal.set_title("Accuracy & Alignment", fontsize=12, color="#333333", pad=12)
    ax_signal.tick_params(axis="x", rotation=25, colors="#333333")
    ax_signal.tick_params(axis="y", colors="#333333")
    ax_signal.spines["top"].set_visible(False)
    ax_signal.spines["right"].set_visible(False)
    for bar, value in zip(bars, signal_values):
        ax_signal.text(bar.get_x() + bar.get_width() / 2, value + (max(signal_values) * 0.05), f"{value:.2f}%",
                       ha="center", va="bottom", fontsize=9, color="#333333")

    subtitle = (
        "Metrics source: results/validation_evaluation.json | "
        "Model: Gemini 2.5 Pro | Evaluation size: 1,000 meals"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9, color="#666666")

    fig.savefig(OUTPUT_PATH, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    make_plot(metrics)


if __name__ == "__main__":
    main()
