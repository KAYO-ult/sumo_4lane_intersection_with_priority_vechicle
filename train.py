"""
train.py - Train a DQN agent for single-intersection traffic signal control.

Uses sumo-rl's Gymnasium (single-agent) interface with Stable-Baselines3 DQN.
The agent observes queue lengths, densities and current phase, then selects
the next green phase for the single traffic light at the intersection.

Usage
    python train.py                          # default 100 000 steps
    python train.py --timesteps 200000       # longer training
    python train.py --gui                    # watch in SUMO-GUI
"""
import os
import sys
import argparse

import sumo_rl
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

from config import (
    NET_FILE, ROUTE_FILE, MODELS_DIR, OUTPUTS_DIR,
    NUM_SECONDS, DELTA_TIME, YELLOW_TIME, MIN_GREEN, MAX_GREEN,
    REWARD_FN,
    LEARNING_RATE, BUFFER_SIZE, LEARNING_STARTS, BATCH_SIZE,
    TARGET_UPDATE_INTERVAL, TRAIN_FREQ,
    EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS, EXPLORATION_FRACTION,
    GAMMA, TOTAL_TIMESTEPS, CHECKPOINT_FREQ, NET_ARCH,
)


def make_env(use_gui: bool = False, out_csv: str | None = None):
    """Create the SUMO-RL single-agent Gymnasium environment."""
    return sumo_rl.SumoEnvironment(
        net_file=NET_FILE,
        route_file=ROUTE_FILE,
        use_gui=use_gui,
        num_seconds=NUM_SECONDS,
        delta_time=DELTA_TIME,
        yellow_time=YELLOW_TIME,
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        reward_fn=REWARD_FN,
        out_csv_name=out_csv,
        single_agent=True,
    )


def train(args: argparse.Namespace) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_dir = os.path.join(OUTPUTS_DIR, "training")
    os.makedirs(train_dir, exist_ok=True)

    out_csv = os.path.join(train_dir, "dqn")
    env = make_env(use_gui=args.gui, out_csv=out_csv)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        train_freq=TRAIN_FREQ,
        exploration_initial_eps=EXPLORATION_INITIAL_EPS,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        exploration_fraction=EXPLORATION_FRACTION,
        gamma=GAMMA,
        policy_kwargs={"net_arch": NET_ARCH},
        verbose=1,
        tensorboard_log=os.path.join(OUTPUTS_DIR, "tb_logs"),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODELS_DIR,
        name_prefix="dqn_intersection",
    )

    print("=" * 60)
    print(f"Training DQN — Single 4-Way Intersection")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  LR        : {LEARNING_RATE}")
    print(f"  Net arch  : {NET_ARCH}")
    print(f"  Reward    : {REWARD_FN}")
    print("=" * 60)

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb],
        progress_bar=True,
    )

    final_path = os.path.join(MODELS_DIR, "dqn_intersection_final")
    model.save(final_path)
    print(f"\nFinal model saved: {final_path}.zip")

    env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DQN for single-intersection signal control."
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Launch SUMO-GUI during training (slow but visual)",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Run with interactive Streamlit dashboard in another terminal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(NET_FILE):
        sys.exit(
            f"ERROR: Network file not found: {NET_FILE}\n"
            "Generate it first:  python generate_network.py"
        )
    if not os.path.isfile(ROUTE_FILE):
        sys.exit(
            f"ERROR: Route file not found: {ROUTE_FILE}\n"
            "Generate it first:  python generate_network.py"
        )

    train(args)

    print("\nNext steps:")
    print("  python evaluate.py              # compare RL vs fixed-time")
    print("  python evaluate.py --gui        # watch the trained agent")
    print("  python plot_results.py          # generate static plots")
    print("  streamlit run dashboard.py      # interactive visualization")

    if args.dashboard:
        print("\n💡 To view real-time progress in another terminal:")
        print("  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
