"""
evaluate.py - Evaluate trained DQN model and compare against fixed-time baseline.

Runs the trained agent (optionally with SUMO-GUI), collects per-episode
performance metrics, then runs the same scenario with SUMO's built-in
fixed-time signals for a head-to-head comparison.

Usage
    python evaluate.py                                         # defaults
    python evaluate.py --model models/dqn_intersection_final.zip --gui
    python evaluate.py --episodes 10 --no-baseline
"""
import os
import sys
import argparse
import re

import numpy as np
import pandas as pd
import sumo_rl
from stable_baselines3 import DQN

from config import (
    NET_FILE, ROUTE_FILE, MODELS_DIR, OUTPUTS_DIR,
    NUM_SECONDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN,
    REWARD_FN, EVAL_EPISODES, AMBULANCE_SPEED,
)


AMBULANCE_ROUTE_FILE = os.path.join(
    os.path.dirname(ROUTE_FILE),
    "intersection_ambulance.rou.xml",
)


def _build_route_file_arg() -> str:
    """Use normal + ambulance routes when ambulance file is available."""
    if os.path.isfile(AMBULANCE_ROUTE_FILE):
        return f"{ROUTE_FILE},{AMBULANCE_ROUTE_FILE}"
    return ROUTE_FILE


def _sanitize_ambulance_route_file() -> None:
    """Make ambulance route XML SUMO-valid and enforce red emergency vehicles."""
    if not os.path.isfile(AMBULANCE_ROUTE_FILE):
        return

    with open(AMBULANCE_ROUTE_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    updated_content = content

    # Ensure emergency type has explicit max speed + red color.
    emergency_vtype = (
        f'<vType id="emergency" vClass="emergency" '
        f'maxSpeed="{AMBULANCE_SPEED:.2f}" color="255,0,0"/>'
    )
    if re.search(r'<vType\s+id="emergency"[^>]*?/?>', updated_content):
        updated_content = re.sub(
            r'<vType\s+id="emergency"[^>]*?/?>',
            emergency_vtype,
            updated_content,
            count=1,
        )
    else:
        updated_content = re.sub(
            r'(<routes[^>]*>)',
            r'\1\n    ' + emergency_vtype,
            updated_content,
            count=1,
        )

    # SUMO route schema does not allow maxSpeed directly on vehicle elements.
    updated_content = re.sub(
        r'(<vehicle\b[^>]*?)\smaxSpeed="[^"]*"',
        r'\1',
        updated_content,
    )

    if updated_content != content:
        with open(AMBULANCE_ROUTE_FILE, "w", encoding="utf-8") as f:
            f.write(updated_content)


def _ensure_unique_ambulance_vehicle_ids(prefix: str = "amb_") -> None:
    """Ensure ambulance vehicle IDs do not collide with normal route IDs."""
    if not os.path.isfile(AMBULANCE_ROUTE_FILE):
        return

    with open(AMBULANCE_ROUTE_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Prefix every <vehicle id="..."> in ambulance routes to avoid duplicate IDs
    # when loading multiple route files in the same SUMO simulation.
    updated_content = re.sub(
        r'(<vehicle\s+id=")([^\"]+)(")',
        lambda m: f'{m.group(1)}{m.group(2) if m.group(2).startswith(prefix) else prefix + m.group(2)}{m.group(3)}',
        content,
    )

    if updated_content != content:
        with open(AMBULANCE_ROUTE_FILE, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated ambulance vehicle IDs with prefix '{prefix}' to avoid duplicates.")


def _apply_ambulance_priority(env) -> None:
    """Allow emergency vehicles to pass without red-light waiting."""
    if not hasattr(env, "sumo") or env.sumo is None:
        return

    try:
        vehicle_ids = env.sumo.vehicle.getIDList()
    except Exception:
        return

    for veh_id in vehicle_ids:
        try:
            if env.sumo.vehicle.getTypeID(veh_id) != "emergency":
                continue

            # Keep emergency vehicles visually distinct.
            env.sumo.vehicle.setColor(veh_id, (255, 0, 0, 255))

            # 7 disables right-of-way and red-light checks while preserving
            # core safety checks, preventing red-signal waiting for emergency vehicles.
            env.sumo.vehicle.setSpeedMode(veh_id, 7)
        except Exception:
            # Ignore per-vehicle TraCI issues so evaluation continues.
            continue


# ── Metric Collection ─────────────────────────────────────────────────

# Keys sumo-rl puts in the info dict at every step
_INFO_KEYS = {
    "system_mean_waiting_time": "avg_waiting_time",
    "system_mean_speed": "avg_speed",
    "system_total_stopped": "total_stopped",
    "system_total_waiting_time": "total_waiting_time",
}


def _collect_metrics(info: dict, step_records: list[dict]) -> None:
    """Append one row of metrics from the step info dict."""
    row = {}
    for src, dst in _INFO_KEYS.items():
        row[dst] = info.get(src, float("nan"))
    step_records.append(row)


def _aggregate_episode(episode: int, step_records: list[dict]) -> dict:
    """Compute episode-level averages from per-step records."""
    metrics: dict = {"episode": episode}
    if not step_records:
        for dst in _INFO_KEYS.values():
            metrics[dst] = float("nan")
        return metrics

    df = pd.DataFrame(step_records)
    for col in df.columns:
        metrics[col] = df[col].mean()
    return metrics


# ── RL Evaluation ──────────────────────────────────────────────────────

def run_rl_evaluation(
    model_path: str,
    num_episodes: int = EVAL_EPISODES,
    use_gui: bool = False,
) -> pd.DataFrame:
    """Run evaluation episodes with the trained RL model."""
    results = []
    route_file_arg = _build_route_file_arg()

    for ep in range(num_episodes):
        env = sumo_rl.SumoEnvironment(
            net_file=NET_FILE,
            route_file=route_file_arg,
            use_gui=use_gui,
            num_seconds=NUM_SECONDS,
            delta_time=DELTA_TIME,
            yellow_time=YELLOW_TIME,
            min_green=MIN_GREEN,
            max_green=MAX_GREEN,
            reward_fn=REWARD_FN,
            single_agent=True,
        )

        model = DQN.load(model_path, env=env)

        obs, info = env.reset()
        _apply_ambulance_priority(env)
        done = False
        total_reward = 0.0
        step_records: list[dict] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            _apply_ambulance_priority(env)
            total_reward += reward
            _collect_metrics(info, step_records)
            done = terminated or truncated

        env.close()

        metrics = _aggregate_episode(ep, step_records)
        metrics["total_reward"] = total_reward
        results.append(metrics)

        print(
            f"  RL Episode {ep + 1}/{num_episodes} — "
            f"reward={total_reward:.1f}, "
            f"avg_wait={metrics['avg_waiting_time']:.1f}s"
        )

    return pd.DataFrame(results)


# ── Fixed-Time Baseline ───────────────────────────────────────────────

def run_fixed_time_baseline(
    num_episodes: int = EVAL_EPISODES,
    use_gui: bool = False,
) -> pd.DataFrame:
    """Run episodes with SUMO's default fixed-time signal plans."""
    results = []
    route_file_arg = _build_route_file_arg()

    for ep in range(num_episodes):
        env = sumo_rl.SumoEnvironment(
            net_file=NET_FILE,
            route_file=route_file_arg,
            use_gui=use_gui,
            num_seconds=NUM_SECONDS,
            delta_time=DELTA_TIME,
            yellow_time=YELLOW_TIME,
            min_green=MIN_GREEN,
            max_green=MAX_GREEN,
            fixed_ts=True,
            single_agent=True,
        )

        obs, info = env.reset()
        _apply_ambulance_priority(env)
        done = False
        step_records: list[dict] = []

        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            _apply_ambulance_priority(env)
            _collect_metrics(info, step_records)
            done = terminated or truncated
        env.close()

        metrics = _aggregate_episode(ep, step_records)
        metrics["total_reward"] = 0.0
        results.append(metrics)

        print(
            f"  Fixed Episode {ep + 1}/{num_episodes} — "
            f"avg_wait={metrics['avg_waiting_time']:.1f}s"
        )

    return pd.DataFrame(results)


# ── Comparison ─────────────────────────────────────────────────────────

def compute_comparison(
    rl_metrics: pd.DataFrame,
    fixed_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compute improvement percentages of RL over fixed-time."""
    comparisons = []

    for metric, lower_is_better in [
        ("avg_waiting_time", True),
        ("avg_speed", False),
        ("total_stopped", True),
        ("total_waiting_time", True),
    ]:
        if metric not in rl_metrics.columns:
            continue

        rl_mean = rl_metrics[metric].mean()
        rl_std = rl_metrics[metric].std()
        fixed_mean = fixed_metrics[metric].mean()
        fixed_std = fixed_metrics[metric].std()

        if fixed_mean != 0:
            if lower_is_better:
                improvement = (fixed_mean - rl_mean) / fixed_mean * 100
            else:
                improvement = (rl_mean - fixed_mean) / fixed_mean * 100
        else:
            improvement = 0.0

        comparisons.append({
            "metric": metric,
            "rl_mean": rl_mean,
            "rl_std": rl_std,
            "fixed_mean": fixed_mean,
            "fixed_std": fixed_std,
            "improvement_pct": improvement,
        })

    return pd.DataFrame(comparisons)


def print_results(comparison: pd.DataFrame) -> None:
    """Pretty-print the comparison table."""
    print("\n" + "=" * 75)
    print("  RESULTS: RL (DQN) vs Fixed-Time Signals")
    print("=" * 75)
    print(
        f"  {'Metric':<25} {'RL Mean':>10} {'Fixed Mean':>12} "
        f"{'Improvement':>12}"
    )
    print("-" * 75)
    for _, row in comparison.iterrows():
        sign = "+" if row["improvement_pct"] > 0 else ""
        print(
            f"  {row['metric']:<25} {row['rl_mean']:>10.2f} "
            f"{row['fixed_mean']:>12.2f} "
            f"{sign}{row['improvement_pct']:>10.1f}%"
        )
    print("=" * 75)


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN model vs fixed-time baseline."
    )
    parser.add_argument(
        "--model", type=str,
        default=os.path.join(MODELS_DIR, "dqn_intersection_final.zip"),
        help="Path to trained model .zip",
    )
    parser.add_argument(
        "--episodes", type=int, default=EVAL_EPISODES,
        help=f"Number of evaluation episodes (default: {EVAL_EPISODES})",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Show SUMO-GUI during RL evaluation",
    )
    parser.add_argument(
        "--no-baseline", action="store_true",
        help="Skip the fixed-time baseline comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.model):
        sys.exit(
            f"ERROR: Model file not found: {args.model}\n"
            "Train a model first:  python train.py"
        )

    print("=" * 60)
    print("Evaluation — 4-Way Intersection")
    print("=" * 60)

    # Ensure ambulance route XML is SUMO-valid and renders in red in GUI.
    _ensure_unique_ambulance_vehicle_ids()
    _sanitize_ambulance_route_file()

    # RL evaluation
    print(f"\n[1/2] Running RL evaluation ({args.episodes} episodes)...")
    rl_metrics = run_rl_evaluation(
        model_path=args.model,
        num_episodes=args.episodes,
        use_gui=args.gui,
    )

    if args.no_baseline:
        print("\nSkipping fixed-time baseline (--no-baseline).")
        print("\nRL Metrics:")
        print(rl_metrics.describe())
        return

    # Fixed-time baseline
    print(f"\n[2/2] Running fixed-time baseline ({args.episodes} episodes)...")
    fixed_metrics = run_fixed_time_baseline(num_episodes=args.episodes)

    # Compare and display
    comparison = compute_comparison(rl_metrics, fixed_metrics)
    print_results(comparison)

    # Save CSVs for plotting
    eval_dir = os.path.join(OUTPUTS_DIR, "evaluation")
    rl_metrics.to_csv(os.path.join(eval_dir, "rl_summary.csv"), index=False)
    fixed_metrics.to_csv(os.path.join(eval_dir, "fixed_summary.csv"), index=False)
    comparison.to_csv(os.path.join(eval_dir, "comparison.csv"), index=False)

    print(f"\nCSVs saved to: {eval_dir}")

    print("\nNext steps:")
    print("  python plot_results.py          # generate static plots")
    print("  streamlit run dashboard.py      # interactive visualization")


if __name__ == "__main__":
    main()
