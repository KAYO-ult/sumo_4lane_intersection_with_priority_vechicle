"""
run_all.py - Execute the complete pipeline end-to-end.

    Step 1: Generate SUMO network + routes   (generate_network.py)
    Step 2: Train DQN agent                  (train.py)
    Step 3: Evaluate RL vs fixed-time        (evaluate.py)
    Step 4: Generate plots                   (plot_results.py)

Usage
    python run_all.py                  # run full pipeline
    python run_all.py --skip-train     # skip training (use existing model)
"""
import subprocess
import sys
import argparse


def _run_step(step_num: int, total: int, description: str,
              cmd: list[str]) -> None:
    """Run a pipeline step, aborting on failure."""
    print(f"\n{'=' * 60}")
    print(f"  [{step_num}/{total}] {description}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nPIPELINE FAILED at step {step_num}: {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full pipeline.")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and use existing model",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override training timesteps",
    )
    args = parser.parse_args()

    py = sys.executable

    steps = [
        ("Generate SUMO network & routes", [py, "generate_network.py"]),
    ]

    if not args.skip_train:
        train_cmd = [py, "train.py"]
        if args.timesteps:
            train_cmd += ["--timesteps", str(args.timesteps)]
        steps.append(("Train DQN agent", train_cmd))

    steps.append(("Evaluate RL vs Fixed-Time", [py, "evaluate.py"]))
    steps.append(("Generate result plots", [py, "plot_results.py"]))

    total = len(steps)
    for i, (desc, cmd) in enumerate(steps, 1):
        _run_step(i, total, desc, cmd)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}")
    print("  Models  : models/")
    print("  CSVs    : outputs/")
    print("  Plots   : results/")


if __name__ == "__main__":
    main()
