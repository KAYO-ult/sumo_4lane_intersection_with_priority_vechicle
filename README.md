# AI Traffic Signal Control — 4-Way Intersection with Priority Vehicles

A Deep Q-Network (DQN) agent that learns to optimize traffic signal timings at a single 4-way intersection, built with **SUMO** (Simulation of Urban Mobility) and **SUMO-RL**. This project includes priority vehicle (ambulance) detection and handling for emergency scenarios.

## Overview

Traditional traffic signals use fixed-time cycles regardless of actual traffic conditions. This project replaces fixed-time control with a reinforcement learning agent (DQN) that observes real-time traffic state — queue lengths, vehicle densities, current signal phase — and dynamically selects the optimal green phase to minimize vehicle waiting time.

The system also includes **priority vehicle (ambulance) detection** that allows emergency vehicles to pass through the intersection with minimal delay by dynamically adjusting traffic signals when approaching.

### Key Features

- **Single 4-way intersection** with 4 approaches (North, South, East, West), 3 lanes each
- **Indian left-hand traffic** (LHT) configuration
- **Priority vehicles (ambulance)** - Emergency vehicles with higher speed and priority handling
- **DQN agent** trained via Stable-Baselines3 with experience replay and target networks
- **Reward function**: `diff-waiting-time` — agent is rewarded for reducing cumulative vehicle delay
- **Automated evaluation**: head-to-head comparison of RL agent vs fixed-time signals
- **Result visualization**: training curves, bar charts, per-episode comparisons, and interactive dashboards

## Architecture

```
┌──────────────┐     TraCI      ┌──────────────┐    Gymnasium   ┌──────────────┐
│     SUMO     │ ◄────────────► │   sumo-rl    │ ◄────────────► │   SB3 DQN    │
│  (Simulator) │                │  (Env Wrap)  │                │   (Agent)    │
└──────────────┘                └──────────────┘                └──────────────┘
```

**Observation space**: queue lengths, vehicle densities, current phase (per lane)
**Action space**: select next green phase for the traffic light
**Reward**: reduction in total cumulative waiting time between decisions

## Project Structure

```
sumo_4lane_intersection_with_priority_vechicle/
├── config.py                    # All tunable parameters (network, DQN, evaluation)
├── generate_network.py          # Builds SUMO network (.net.xml) + vehicle routes
├── randomTrips.py               # Custom random trip generator with Poisson arrival
├── train.py                     # DQN training (single-agent, Gymnasium interface)
├── evaluate.py                  # RL vs fixed-time baseline comparison
├── plot_results.py              # Static plots (training curves, comparison charts)
├── dashboard.py                 # Interactive Streamlit visualization dashboard
├── visualization_utils.py       # Shared utilities for visualization tools
├── run_all.py                   # One-command full pipeline
├── requirements.txt             # Python dependencies
├── export_explanation.py        # Generate project explanation DOCX
├── nets/                        # Generated SUMO network and route files
│   ├── intersection.net.xml     # Main network file
│   ├── intersection.rou.xml     # Normal traffic demand
│   ├── intersection_low.rou.xml    # Low traffic demand
│   ├── intersection_high.rou.xml   # High traffic demand
│   └── intersection_ambulance.rou.xml  # Priority/ambulance vehicles
├── models/                      # Saved DQN model checkpoints
├── outputs/                     # Training CSVs, TensorBoard logs, evaluation data
│   ├── training/                # Per-episode training records (CSVs)
│   ├── evaluation/              # RL vs Fixed-Time comparison results
│   └── tb_logs/                 # TensorBoard event files
└── results/                    # Output PNG charts
    └── *.png                    # Static plot images
```

## Prerequisites

1. **SUMO** (Simulation of Urban Mobility)
   - Download: https://sumo.dlr.de/docs/Downloads.php
   - Set the `SUMO_HOME` environment variable:
     ```bash
     # Windows (System Environment Variables)
     SUMO_HOME = C:\Program Files (x86)\Eclipse\Sumo

     # Linux/Mac
     export SUMO_HOME=/usr/share/sumo
     ```

2. **Python 3.10+**

## Installation

```bash
cd sumo_4lane_intersection_with_priority_vechicle
pip install -r requirements.txt
```

## Usage

### Full Pipeline (one command)

```bash
python run_all.py
```

This runs all four steps sequentially: generate → train → evaluate → plot.

### Step-by-Step

```bash
# Step 1: Generate the SUMO network and vehicle demand
python generate_network.py

# Step 2: Train the DQN agent (100,000 timesteps)
python train.py

# Step 3: Evaluate RL vs fixed-time signals (5 episodes each)
python evaluate.py

# Step 4: Generate result plots
python plot_results.py
```

### Useful Flags

| Command | Flag | Description |
|---|---|---|
| `generate_network.py` | (no flags) | Generates network + routes (normal, low, high, ambulance) |
| `train.py` | `--gui` | Open SUMO-GUI to watch vehicles during training |
| `train.py` | `--dashboard` | Reminder to run dashboard in another terminal for real-time monitoring |
| `train.py` | `--timesteps N` | Set total training timesteps (default: 100,000) |
| `evaluate.py` | `--gui` | Watch the trained agent control traffic in SUMO-GUI |
| `evaluate.py` | `--episodes N` | Number of evaluation episodes (default: 5) |
| `evaluate.py` | `--no-baseline` | Skip fixed-time comparison |

### Visual Inspection

```bash
# View the intersection in SUMO-GUI (after generating network)
sumo-gui nets/intersection.sumocfg
```

## Interactive Visualization

This project includes **multiple visualization tools** for monitoring training and analyzing results:

### 1. Interactive Dashboard (Streamlit)

The most comprehensive way to explore results in real-time:

```bash
streamlit run dashboard.py
```

**Features:**
- 📊 **Training Progress Tab** — Live training curves with rolling averages
  - Total/mean waiting time over training steps
  - Mean speed and stopped vehicles metrics
  - Auto-refresh every 5-60 seconds while training

- 📈 **Comparison Metrics Tab** — RL vs Fixed-Time analysis
  - Grouped bar charts with improvement percentages
  - Per-episode trend lines
  - Statistical summaries

- 🏆 **Model Checkpoints Tab** — Training progression explorer
  - Browse all saved checkpoints by step
  - Track performance across training milestones
  - File information (size, modification time)

- 📺 **TensorBoard Tab** — Integration with TensorBoard
  - Instructions to launch TensorBoard
  - Live metric exploration
  - Direct access to event files

- ⚙️ **Help & Settings Tab** — Documentation and system info

### 2. Real-Time Training Monitoring

Monitor training progress in the dashboard while training runs:

**Terminal 1: Start dashboard**
```bash
streamlit run dashboard.py
```
Dashboard opens at: http://localhost:8501

**Terminal 2: Run training**
```bash
python train.py
```

Dashboard automatically detects new training data and refreshes every 5s.

### 3. Static Plots

Generate high-quality PNG charts for reports:

```bash
python plot_results.py
```

**Outputs:**
- `results/training_curves.png` — 4-subplot training progress
- `results/comparison_metrics.png` — Bar chart of metrics comparison
- `results/episode_comparison.png` — Line plot of per-episode performance

### 4. TensorBoard

Detailed metrics logging for model analysis:

```bash
tensorboard --logdir outputs/tb_logs
```

Opens at: http://localhost:6006

**Metrics:**
- Episode length and rewards
- DQN loss function and learning rate
- Network gradients and weight distributions
- Custom scalars and histograms

## Priority Vehicles (Ambulance)

This project includes **priority vehicle handling** for emergency scenarios:

- **Ambulance generation**: One ambulance every 30 seconds
- **Ambulance speed**: 25 m/s (~90 km/h) - significantly faster than regular traffic
- **Priority detection**: The RL agent can be extended to detect and prioritize emergency vehicles
- **Emergency routes**: Separate route file (`intersection_ambulance.rou.xml`) for priority vehicles

The ambulance vehicles are visually distinguished in the simulation and can trigger special signal handling.

## Results

After evaluation, the comparison between RL (DQN) and Fixed-Time signals shows significant improvements:

| Metric | RL (DQN) | Fixed-Time | Improvement |
|---|---|---|---|
| Avg Waiting Time (s) | 1.33 | 4.00 | **+66.6%** |
| Avg Speed (m/s) | 8.43 | 7.96 | **+5.8%** |
| Total Stopped Vehicles | 7.32 | 10.31 | **+29.0%** |
| Total Waiting Time (s) | 56.86 | 155.23 | **+63.4%** |

Result charts are saved to `results/`:
- `training_curves.png` — DQN training progress over 100K steps
- `comparison_metrics.png` — Grouped bar chart with improvement percentages
- `episode_comparison.png` — Per-episode waiting time line plot

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_LANES` | 3 | Lanes per direction on each approach |
| `SPEED_LIMIT` | 13.89 m/s | ~50 km/h, typical Indian urban road |
| `LEFT_HAND_TRAFFIC` | True | Indian left-hand driving |
| `AMBULANCE_PERIOD` | 30.0s | Time between ambulance arrivals |
| `AMBULANCE_SPEED` | 25.0 m/s | ~90 km/h, priority vehicle speed |
| `DELTA_TIME` | 5s | Seconds between agent decisions |
| `YELLOW_TIME` | 3s | Yellow phase duration |
| `MIN_GREEN` | 5s | Minimum green phase duration |
| `MAX_GREEN` | 60s | Maximum green phase duration |
| `LEARNING_RATE` | 0.001 | DQN learning rate |
| `NET_ARCH` | [256, 256] | MLP hidden layer sizes |
| `TOTAL_TIMESTEPS` | 100,000 | Training duration |
| `REWARD_FN` | diff-waiting-time | Reward = reduction in cumulative delay |
| `VEHICLE_PERIOD` | 1.5s | ~2400 vehicles/hour |
| `SIMULATION_DURATION` | 3600s | 1 hour per episode |

## Technologies

- **SUMO** — microscopic traffic simulation
- **sumo-rl** — Gymnasium wrapper for SUMO traffic signal control
- **Stable-Baselines3** — DQN implementation with experience replay
- **Gymnasium** — RL environment interface
- **Streamlit** — interactive web dashboard for real-time monitoring
- **Plotly / Altair** — interactive charts in the dashboard
- **Matplotlib / Pandas** — static visualization and data processing
- **TensorBoard** — detailed training metrics and debugging
- **PIL/Pillow** — image generation utilities
- **python-docx** — DOCX report generation

## Generate Project Documentation

Create a detailed DOCX explanation of the project:

```bash
python export_explanation.py
```

This generates `project_explanation.docx` with:
- Problem statement and solution overview
- Step-by-step explanation of the system
- Technology stack details
- Architecture diagram
- Training parameters
- Usage instructions