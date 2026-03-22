"""
visualization_utils.py - Shared utilities for interactive visualization tools.

Provides functions for:
  - Reading live training logs (streaming CSV reader)
  - Parsing TensorBoard event files
  - Tracking SUMO simulation state (vehicles, signals)
  - Creating animation frames
  - Color schemes and constants
"""
import os
import glob
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ── Color Schemes ─────────────────────────────────────────────────────────

COLOR_RL = (70, 130, 180)           # Steel blue
COLOR_FIXED = (255, 127, 80)        # Coral
COLOR_PRIORITY_VEHICLE = (255, 165, 0)  # Orange (Priority/Emergency)
COLOR_AMBULANCE = (255, 0, 128)     # Deep Pink (Priority indicator)
COLOR_VEHICLE_MOVING = (100, 149, 237)  # Cornflower blue
COLOR_VEHICLE_STOPPED = (220, 20, 60)   # Crimson
COLOR_PRIORITY_MOVING = (255, 50, 50)   # Bright Red (Priority moving)
COLOR_PRIORITY_STOPPED = (200, 0, 0)    # Dark Red (Priority stopped)
COLOR_BACKGROUND = (240, 240, 240)  # Light gray
COLOR_ROAD = (200, 200, 200)        # Road gray
COLOR_GREEN_LIGHT = (0, 255, 0)     # Green
COLOR_RED_LIGHT = (255, 0, 0)       # Red
COLOR_YELLOW_LIGHT = (255, 255, 0)  # Yellow
COLOR_TEXT_DARK = (30, 30, 30)      # Dark text
COLOR_TEXT_LIGHT = (200, 200, 200)  # Light text
COLOR_PRIORITY_BORDER = (255, 100, 0)   # Orange border for priority vehicles


# ── Data Structures ───────────────────────────────────────────────────────

@dataclass
class VehicleState:
    """Snapshot of a vehicle's state during simulation."""
    vehicle_id: str
    x: float
    y: float
    speed: float
    is_ambulance: bool = False
    is_stopped: bool = False


@dataclass
class TrafficLightState:
    """Snapshot of traffic light state."""
    phase: int                        # 0-3 for 4-phase signal
    times_in_phase: float             # seconds in current phase
    min_green: float = 5.0
    max_green: float = 60.0

    def is_green_direction(self, direction: int) -> bool:
        """Check if a direction has green light."""
        return self.phase == direction

    def get_color(self, direction: int) -> Tuple[int, int, int]:
        """Get color for traffic light in a direction."""
        if self.phase == direction:
            return COLOR_GREEN_LIGHT
        return COLOR_RED_LIGHT


@dataclass
class SimulationSnapshot:
    """Complete snapshot of simulation state at one timestep."""
    step: int
    sim_time: float
    vehicles: List[VehicleState]
    traffic_light: TrafficLightState
    total_waiting_time: float = 0.0
    mean_waiting_time: float = 0.0
    mean_speed: float = 0.0
    total_stopped: int = 0


# ── CSV Log Reading ───────────────────────────────────────────────────────

def read_training_logs(training_dir: str, max_records: Optional[int] = None) -> pd.DataFrame:
    """
    Read all training CSV files in a directory.

    Args:
        training_dir: Path to directory containing training CSVs (dqn_conn0_epX.csv)
        max_records: Max total records to read (for performance); None = all

    Returns:
        Combined DataFrame with all training records
    """
    csv_files = sorted(glob.glob(os.path.join(training_dir, "*.csv")))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    total_records = 0

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if max_records and total_records + len(df) > max_records:
                df = df.iloc[:max_records - total_records]
            frames.append(df)
            total_records += len(df)
            if max_records and total_records >= max_records:
                break
        except Exception as e:
            print(f"Warning: Could not read {csv_file}: {e}")

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def get_latest_episode_metrics(training_dir: str) -> Optional[Dict[str, float]]:
    """
    Get metrics from the latest training episode CSV.

    Returns:
        Dict with columns as keys, or None if no data
    """
    csv_files = sorted(glob.glob(os.path.join(training_dir, "*.csv")))
    if not csv_files:
        return None

    try:
        df = pd.read_csv(csv_files[-1])
        return df.iloc[-1].to_dict() if len(df) > 0 else None
    except Exception:
        return None


def read_live_csv(csv_path: str, skip_rows_cache: Dict = None) -> pd.DataFrame:
    """
    Read a CSV file, skipping rows already read (for live monitoring).

    Useful for streaming training progress without re-reading entire file.

    Args:
        csv_path: Path to CSV file
        skip_rows_cache: Dict with 'last_row' key tracking last read position

    Returns:
        New rows since last read
    """
    if skip_rows_cache is None:
        skip_rows_cache = {}

    try:
        df = pd.read_csv(csv_path)
        last_row = skip_rows_cache.get('last_row', 0)
        new_df = df.iloc[last_row:]
        skip_rows_cache['last_row'] = len(df)
        return new_df
    except Exception:
        return pd.DataFrame()


def compute_rolling_average(
    series: pd.Series,
    window: int = 5,
    min_periods: int = 1
) -> pd.Series:
    """
    Compute rolling average for smoothing training curves.

    Args:
        series: Data series
        window: Rolling window size
        min_periods: Minimum periods for computation

    Returns:
        Smoothed series
    """
    return series.rolling(window=window, min_periods=min_periods).mean()


# ── TensorBoard Event Parsing ─────────────────────────────────────────────

def get_tensorboard_scalar_event(
    tb_log_dir: str,
    tag: str,
    max_steps: Optional[int] = None
) -> Optional[Tuple[List[int], List[float]]]:
    """
    Extract scalar values from TensorBoard event files.

    Args:
        tb_log_dir: Directory containing TensorBoard logs (e.g., outputs/tb_logs/DQN_5/)
        tag: Scalar tag to extract (e.g., 'rollout/ep_len_mean', 'loss/actor_loss')
        max_steps: Max timesteps to return; None = all

    Returns:
        (steps, values) tuple, or None if not found
    """
    try:
        from tensorboard.compat.tensorflow_stub import io as tb_io
    except ImportError:
        print("Warning: TensorBoard not available for event parsing")
        return None

    steps = []
    values = []

    # Find event files
    event_files = glob.glob(os.path.join(tb_log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None

    try:
        for event_file in event_files:
            for event in tb_io.read_events(event_file):
                if not event.summary.value:
                    continue

                for v in event.summary.value:
                    if v.tag == tag and v.simple_value is not None:
                        steps.append(event.step)
                        values.append(v.simple_value)

                if max_steps and len(steps) >= max_steps:
                    return (steps[:max_steps], values[:max_steps])

        return (steps, values) if steps else None
    except Exception as e:
        print(f"Warning: Could not parse TensorBoard events: {e}")
        return None


# ── Animation Frame Rendering ────────────────────────────────────────────

def create_intersection_frame(
    snapshot: SimulationSnapshot,
    canvas_size: int = 400,
    zoom_factor: float = 1.0,
) -> Image.Image:
    """
    Create a PIL Image showing the intersection from above.

    Draws:
      - Road layout (4 approaches)
      - Vehicles as circles (color = stopped/moving)
      - Traffic lights for each direction
      - Metadata (step, waiting time, speed)

    Args:
        snapshot: SimulationSnapshot with current state
        canvas_size: Size of output image (canvas_size x canvas_size)
        zoom_factor: Zoom level for visualization

    Returns:
        PIL Image object
    """
    img = Image.new("RGB", (canvas_size, canvas_size), COLOR_BACKGROUND)
    draw = ImageDraw.Draw(img)

    # Center of intersection
    cx, cy = canvas_size // 2, canvas_size // 2

    # Road dimensions
    road_width = 60
    road_length = 150

    # Draw roads (N, S, E, W)
    # North road
    draw.rectangle(
        [(cx - road_width // 2, cy - road_length),
         (cx + road_width // 2, cy)],
        fill=COLOR_ROAD
    )
    # South road
    draw.rectangle(
        [(cx - road_width // 2, cy),
         (cx + road_width // 2, cy + road_length)],
        fill=COLOR_ROAD
    )
    # East road
    draw.rectangle(
        [(cx, cy - road_width // 2),
         (cx + road_length, cy + road_width // 2)],
        fill=COLOR_ROAD
    )
    # West road
    draw.rectangle(
        [(cx - road_length, cy - road_width // 2),
         (cx, cy + road_width // 2)],
        fill=COLOR_ROAD
    )

    # Draw intersection box
    draw.rectangle(
        [(cx - road_width // 2, cy - road_width // 2),
         (cx + road_width // 2, cy + road_width // 2)],
        fill=(220, 220, 200),
        outline=(100, 100, 100),
        width=2
    )

    # Draw traffic lights
    tl_offset = 20
    tl_size = 6

    # North light
    north_color = snapshot.traffic_light.get_color(0)
    draw.rectangle(
        [(cx - tl_size // 2, cy - road_length - tl_offset - tl_size),
         (cx + tl_size // 2, cy - road_length - tl_offset)],
        fill=north_color
    )

    # South light
    south_color = snapshot.traffic_light.get_color(2)
    draw.rectangle(
        [(cx - tl_size // 2, cy + road_length + tl_offset),
         (cx + tl_size // 2, cy + road_length + tl_offset + tl_size)],
        fill=south_color
    )

    # East light
    east_color = snapshot.traffic_light.get_color(1)
    draw.rectangle(
        [(cx + road_length + tl_offset, cy - tl_size // 2),
         (cx + road_length + tl_offset + tl_size, cy + tl_size // 2)],
        fill=east_color
    )

    # West light
    west_color = snapshot.traffic_light.get_color(3)
    draw.rectangle(
        [(cx - road_length - tl_offset - tl_size, cy - tl_size // 2),
         (cx - road_length - tl_offset, cy + tl_size // 2)],
        fill=west_color
    )

    # Draw vehicles
    vehicle_radius = 4
    for vehicle in snapshot.vehicles:
        # Simple mapping: normalize position to canvas
        # In real implementation, use SUMO coordinate system
        vx = cx + (vehicle.x - cx) * zoom_factor
        vy = cy + (vehicle.y - cy) * zoom_factor

        # Determine vehicle color
        if vehicle.is_ambulance:
            # Priority/Emergency vehicles
            color = COLOR_PRIORITY_STOPPED if vehicle.is_stopped else COLOR_PRIORITY_MOVING
            # Draw outer border for priority vehicles
            draw.ellipse(
                [(vx - vehicle_radius - 2, vy - vehicle_radius - 2),
                 (vx + vehicle_radius + 2, vy + vehicle_radius + 2)],
                outline=COLOR_PRIORITY_BORDER,
                width=2
            )
        else:
            # Regular vehicles
            color = COLOR_VEHICLE_STOPPED if vehicle.is_stopped else COLOR_VEHICLE_MOVING

        draw.ellipse(
            [(vx - vehicle_radius, vy - vehicle_radius),
             (vx + vehicle_radius, vy + vehicle_radius)],
            fill=color
        )

    # Draw metadata overlay
    try:
        font = ImageFont.load_default()
    except:
        font = None

    text_lines = [
        f"Step: {snapshot.step} | Time: {snapshot.sim_time:.1f}s",
        f"Waiting: {snapshot.mean_waiting_time:.1f}s | Speed: {snapshot.mean_speed:.1f}m/s",
        f"Stopped: {snapshot.total_stopped} | Phase: {snapshot.traffic_light.phase}",
    ]

    y_offset = 10
    for line in text_lines:
        draw.text((10, y_offset), line, fill=COLOR_TEXT_DARK, font=font)
        y_offset += 12

    # Draw legend for vehicle types
    legend_y = canvas_size - 60
    draw.text((10, legend_y), "Legend:", fill=COLOR_TEXT_DARK, font=font)

    # Priority vehicle (moving)
    draw.ellipse(
        [(25, legend_y + 12), (35, legend_y + 22)],
        fill=COLOR_PRIORITY_MOVING,
        outline=COLOR_PRIORITY_BORDER,
        width=1
    )
    draw.text((40, legend_y + 12), "Priority Moving", fill=COLOR_TEXT_DARK, font=font)

    # Priority vehicle (stopped)
    draw.ellipse(
        [(25, legend_y + 25), (35, legend_y + 35)],
        fill=COLOR_PRIORITY_STOPPED,
        outline=COLOR_PRIORITY_BORDER,
        width=1
    )
    draw.text((40, legend_y + 25), "Priority Stopped", fill=COLOR_TEXT_DARK, font=font)

    # Regular vehicle (moving)
    draw.ellipse(
        [(200, legend_y + 12), (210, legend_y + 22)],
        fill=COLOR_VEHICLE_MOVING
    )
    draw.text((215, legend_y + 12), "Regular Moving", fill=COLOR_TEXT_DARK, font=font)

    # Regular vehicle (stopped)
    draw.ellipse(
        [(200, legend_y + 25), (210, legend_y + 35)],
        fill=COLOR_VEHICLE_STOPPED
    )
    draw.text((215, legend_y + 25), "Regular Stopped", fill=COLOR_TEXT_DARK, font=font)

    return img


def create_side_by_side_comparison(
    rl_image: Image.Image,
    fixed_image: Image.Image,
    rl_metrics: Dict[str, float],
    fixed_metrics: Dict[str, float],
) -> Image.Image:
    """
    Create side-by-side comparison image of RL and Fixed-Time simulations.

    Args:
        rl_image: PIL Image of RL scenario
        fixed_image: PIL Image of Fixed-Time scenario
        rl_metrics: Metrics dict for RL scenario
        fixed_metrics: Metrics dict for Fixed-Time scenario

    Returns:
        Composite PIL Image with labels and metrics
    """
    width = rl_image.width + fixed_image.width + 20
    height = max(rl_image.height, fixed_image.height) + 40

    composite = Image.new("RGB", (width, height), COLOR_BACKGROUND)

    # Paste images
    composite.paste(rl_image, (10, 30))
    composite.paste(fixed_image, (rl_image.width + 10, 30))

    # Add labels
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    rl_x = 10 + rl_image.width // 2
    fixed_x = rl_image.width + 10 + fixed_image.width // 2

    # Title line
    rl_wait = rl_metrics.get("avg_waiting_time", 0)
    fixed_wait = fixed_metrics.get("avg_waiting_time", 0)
    improvement = (fixed_wait - rl_wait) / fixed_wait * 100 if fixed_wait > 0 else 0

    draw.text((rl_x - 30, 5), "RL (DQN)", fill=COLOR_RL, font=font)
    draw.text((fixed_x - 50, 5), "Fixed-Time", fill=COLOR_FIXED, font=font)

    # Bottom info
    bottom_y = max(rl_image.height, fixed_image.height) + 35
    draw.text(
        (10, bottom_y),
        f"Improvement: {improvement:+.1f}% (lower waiting time = better)",
        fill=COLOR_TEXT_DARK,
        font=font
    )

    return composite


# ── Utility Functions ─────────────────────────────────────────────────────

def get_latest_model_checkpoint(models_dir: str) -> Optional[str]:
    """
    Get the path to the latest model checkpoint.

    Returns:
        Path to .zip file, or None if not found
    """
    checkpoint_files = sorted(glob.glob(os.path.join(models_dir, "dqn_intersection_*.zip")))
    if not checkpoint_files:
        return None

    # Prefer "final" model, then highest numbered checkpoint
    for f in reversed(checkpoint_files):
        if "final" in f:
            return f

    return checkpoint_files[-1]


def get_checkpoint_step(checkpoint_path: str) -> Optional[int]:
    """
    Extract training step from checkpoint filename.

    e.g., "dqn_intersection_50000_steps.zip" -> 50000
    """
    try:
        filename = Path(checkpoint_path).stem
        parts = filename.split("_")
        for i, part in enumerate(parts):
            if part == "steps" and i > 0:
                return int(parts[i - 1])
    except (ValueError, IndexError):
        pass
    return None


def format_time(seconds: float) -> str:
    """Format time duration as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    """Apply rolling average smoothing to a series."""
    if len(series) < window:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


# ── File and Directory Utilities ──────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def get_animation_output_dir(results_dir: str) -> str:
    """Get or create animation output directory."""
    anim_dir = os.path.join(results_dir, "animations")
    return ensure_dir(anim_dir)


def cleanup_old_animations(anim_dir: str, keep_count: int = 10) -> None:
    """Keep only the most recent N animation files."""
    gif_files = sorted(glob.glob(os.path.join(anim_dir, "*.gif")))
    if len(gif_files) > keep_count:
        for f in gif_files[:-keep_count]:
            try:
                os.remove(f)
            except Exception:
                pass
