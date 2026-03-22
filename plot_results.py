"""
plot_results.py - Generate training curves and evaluation comparison plots.

Reads CSV outputs produced by sumo-rl during training (via train.py)
and evaluation summaries (via evaluate.py) to produce matplotlib figures.

Usage
    python plot_results.py
    python plot_results.py --training-dir outputs/training
    python plot_results.py --eval-dir outputs/evaluation
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUTS_DIR, RESULTS_DIR


def plot_training_curves(csv_dir: str, output_path: str, window: int = 5) -> None:
    """
    Plot training metrics over simulation steps.

    Produces a 2x2 subplot:
        (0,0) Total waiting time vs step
        (0,1) Mean waiting time vs step
        (1,0) Mean speed vs step
        (1,1) Total stopped vehicles vs step
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        print(f"  No training CSV files found in {csv_dir}")
        return

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    if not frames:
        print("  No valid training data found.")
        return

    data = pd.concat(frames, ignore_index=True)

    metrics = [
        ("system_total_waiting_time", "Total Waiting Time (s)"),
        ("system_mean_waiting_time", "Mean Waiting Time (s)"),
        ("system_mean_speed", "Mean Speed (m/s)"),
        ("system_total_stopped", "Total Stopped Vehicles"),
    ]

    available = [(col, label) for col, label in metrics if col in data.columns]
    if not available:
        print(f"  No recognized metric columns. Found: {list(data.columns)}")
        return

    n = len(available)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(
        "DQN Training Progress — 4-Way Intersection",
        fontsize=14, fontweight="bold",
    )

    for idx, (col, label) in enumerate(available):
        ax = axes[idx // cols, idx % cols]
        values = data[col].values
        ax.plot(values, alpha=0.3, color="steelblue", linewidth=0.5)
        if len(values) > window:
            smoothed = pd.Series(values).rolling(window, min_periods=1).mean()
            ax.plot(
                smoothed, color="darkblue", linewidth=1.5,
                label=f"{window}-step avg",
            )
            ax.legend(fontsize=9)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    for idx in range(len(available), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison_bars(eval_dir: str, output_path: str) -> None:
    """Create grouped bar chart comparing RL vs Fixed-Time."""
    comparison_file = os.path.join(eval_dir, "comparison.csv")
    if not os.path.isfile(comparison_file):
        print(f"  Comparison CSV not found: {comparison_file}")
        print("  Run  python evaluate.py  first.")
        return

    df = pd.read_csv(comparison_file)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(
        x - width / 2, df["rl_mean"], width, label="RL (DQN)",
        color="steelblue", yerr=df["rl_std"], capsize=4,
    )
    ax.bar(
        x + width / 2, df["fixed_mean"], width, label="Fixed-Time",
        color="coral", yerr=df["fixed_std"], capsize=4,
    )

    for i, row in df.iterrows():
        sign = "+" if row["improvement_pct"] > 0 else ""
        ax.annotate(
            f'{sign}{row["improvement_pct"]:.1f}%',
            xy=(i - width / 2, row["rl_mean"]),
            xytext=(0, 10), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
            color="green" if row["improvement_pct"] > 0 else "red",
        )

    labels = [m.replace("_", " ").title() for m in df["metric"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value")
    ax.set_title(
        "RL (DQN) vs Fixed-Time Signal Control — 4-Way Intersection",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_per_episode_comparison(eval_dir: str, output_path: str) -> None:
    """Line plot comparing waiting time per episode for RL vs Fixed-Time."""
    rl_file = os.path.join(eval_dir, "rl_summary.csv")
    fixed_file = os.path.join(eval_dir, "fixed_summary.csv")

    if not os.path.isfile(rl_file) or not os.path.isfile(fixed_file):
        print("  Episode summary CSVs not found. Run  python evaluate.py  first.")
        return

    rl = pd.read_csv(rl_file)
    fixed = pd.read_csv(fixed_file)

    metric = "avg_waiting_time"
    if metric not in rl.columns or metric not in fixed.columns:
        print(f"  Column '{metric}' not found in summary CSVs.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        rl["episode"], rl[metric], "o-", color="steelblue",
        label="RL (DQN)", linewidth=2, markersize=6,
    )
    ax.plot(
        fixed["episode"], fixed[metric], "s--", color="coral",
        label="Fixed-Time", linewidth=2, markersize=6,
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Waiting Time (s)")
    ax.set_title(
        "Per-Episode Waiting Time Comparison",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate result plots.")
    parser.add_argument(
        "--training-dir", type=str,
        default=os.path.join(OUTPUTS_DIR, "training"),
        help="Directory with training CSVs from sumo-rl",
    )
    parser.add_argument(
        "--eval-dir", type=str,
        default=os.path.join(OUTPUTS_DIR, "evaluation"),
        help="Directory with evaluation CSVs from evaluate.py",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Generating Result Plots")
    print("=" * 60)

    print("\n[1/3] Training curves...")
    plot_training_curves(
        args.training_dir,
        os.path.join(RESULTS_DIR, "training_curves.png"),
    )

    print("\n[2/3] Comparison bar chart...")
    plot_comparison_bars(
        args.eval_dir,
        os.path.join(RESULTS_DIR, "comparison_metrics.png"),
    )

    print("\n[3/3] Per-episode comparison...")
    plot_per_episode_comparison(
        args.eval_dir,
        os.path.join(RESULTS_DIR, "episode_comparison.png"),
    )

    print("\n" + "=" * 60)
    print(f"Plots saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
