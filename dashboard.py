"""
dashboard.py - Interactive Streamlit dashboard for SUMO-RL training visualization.

Provides real-time monitoring of training progress, metrics comparison,
model checkpoint exploration, and TensorBoard integration.

Usage:
    streamlit run dashboard.py

Features:
    - Real-time training progress (auto-refreshing)
    - Comparison metrics (RL vs Fixed-Time)
    - Model checkpoint explorer
    - Priority vehicle (ambulance) handling stats
    - TensorBoard integration
"""
import os
import glob
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from pathlib import Path

from config import OUTPUTS_DIR, MODELS_DIR, RESULTS_DIR, TOTAL_TIMESTEPS
from visualization_utils import (
    read_training_logs,
    get_latest_episode_metrics,
    compute_rolling_average,
    get_latest_model_checkpoint,
    get_checkpoint_step,
    format_time,
    smooth_series,
)


# ── Page Configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="SUMO-RL Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .improvement-positive {
        color: #28a745;
        font-weight: bold;
    }
    .improvement-negative {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ── Sidebar Configuration ───────────────────────────────────────────────────

st.sidebar.title("🚦 SUMO-RL Dashboard")
st.sidebar.write("Interactive visualization for 4-way intersection traffic control")

tab = st.sidebar.radio(
    "Select View",
    [
        "📊 Training Progress",
        "📈 Comparison Metrics",
        "🏆 Model Checkpoints",
        "📺 TensorBoard",
        "⚙️ Help & Settings",
    ]
)

# Refresh controls
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

with col2:
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_interval = st.slider("Interval (s)", 5, 60, 10) if auto_refresh else 10

if auto_refresh:
    st.sidebar.info(f"Auto-refreshing every {refresh_interval}s")
    st.autorefresh(interval=refresh_interval * 1000, key="dashboard_refresh")


# ── Helper Functions ───────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_training_data():
    """Load training CSV data with caching."""
    train_dir = os.path.join(OUTPUTS_DIR, "training")
    if os.path.isdir(train_dir):
        return read_training_logs(train_dir)
    return pd.DataFrame()


@st.cache_data(ttl=5)
def load_comparison_data():
    """Load comparison CSV data."""
    eval_dir = os.path.join(OUTPUTS_DIR, "evaluation")
    comp_file = os.path.join(eval_dir, "comparison.csv")

    if os.path.isfile(comp_file):
        return pd.read_csv(comp_file)
    return pd.DataFrame()


@st.cache_data(ttl=5)
def load_rl_eval_data():
    """Load RL evaluation summary."""
    eval_dir = os.path.join(OUTPUTS_DIR, "evaluation")
    rl_file = os.path.join(eval_dir, "rl_summary.csv")

    if os.path.isfile(rl_file):
        return pd.read_csv(rl_file)
    return pd.DataFrame()


@st.cache_data(ttl=5)
def load_fixed_eval_data():
    """Load fixed-time evaluation summary."""
    eval_dir = os.path.join(OUTPUTS_DIR, "evaluation")
    fixed_file = os.path.join(eval_dir, "fixed_summary.csv")

    if os.path.isfile(fixed_file):
        return pd.read_csv(fixed_file)
    return pd.DataFrame()




# ── Tab: Training Progress ──────────────────────────────────────────────────

def show_training_progress():
    st.title("📊 Training Progress")

    training_data = load_training_data()

    if training_data.empty:
        st.warning("No training data found. Run `python train.py` first.")
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latest = get_latest_episode_metrics(os.path.join(OUTPUTS_DIR, "training"))
        if latest:
            st.metric(
                "Latest Waiting Time",
                f"{latest.get('system_mean_waiting_time', 0):.1f}s",
            )

    with col2:
        st.metric(
            "Total Records",
            f"{len(training_data):,}",
        )

    with col3:
        episode_count = len(glob.glob(os.path.join(os.path.join(OUTPUTS_DIR, "training"), "*.csv")))
        st.metric(
            "Episodes",
            f"{episode_count}",
        )

    with col4:
        st.metric(
            "Data Points",
            f"{len(training_data)}",
        )

    st.markdown("---")

    # Metric selection
    available_metrics = [
        ("system_total_waiting_time", "Total Waiting Time (s)"),
        ("system_mean_waiting_time", "Mean Waiting Time (s)"),
        ("system_mean_speed", "Mean Speed (m/s)"),
        ("system_total_stopped", "Total Stopped Vehicles"),
    ]

    selected_metrics = st.multiselect(
        "Select metrics to display",
        [label for _, label in available_metrics],
        default=[label for _, label in available_metrics],
    )

    # Smoothing control
    window = st.slider("Rolling average window", 1, 20, 5)

    # Plot training curves
    for col, label in available_metrics:
        if label in selected_metrics and col in training_data.columns:
            fig = go.Figure()

            # Raw data
            fig.add_trace(go.Scatter(
                y=training_data[col],
                mode='lines',
                name='Raw',
                line=dict(color='rgba(100, 100, 200, 0.3)', width=1),
            ))

            # Smoothed data
            smoothed = smooth_series(training_data[col], window=window)
            fig.add_trace(go.Scatter(
                y=smoothed,
                mode='lines',
                name=f'{window}-step avg',
                line=dict(color='steelblue', width=2),
            ))

            fig.update_layout(
                title=label,
                xaxis_title="Step",
                yaxis_title=label,
                hovermode='x unified',
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)


# ── Tab: Comparison Metrics ─────────────────────────────────────────────────

def show_comparison_metrics():
    st.title("📈 RL vs Fixed-Time Comparison")

    comparison_data = load_comparison_data()

    if comparison_data.empty:
        st.warning("No comparison data found. Run `python evaluate.py` first.")
        return

    # Metrics summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metrics = comparison_data[comparison_data["metric"] == "avg_waiting_time"]
        if not metrics.empty:
            improvement = metrics.iloc[0]["improvement_pct"]
            st.metric(
                "Waiting Time Improvement",
                f"{improvement:+.1f}%",
                delta_color="off" if improvement > 0 else "normal",
            )

    with col2:
        metrics = comparison_data[comparison_data["metric"] == "avg_speed"]
        if not metrics.empty:
            improvement = metrics.iloc[0]["improvement_pct"]
            st.metric(
                "Speed Improvement",
                f"{improvement:+.1f}%",
                delta_color="off" if improvement > 0 else "normal",
            )

    with col3:
        metrics = comparison_data[comparison_data["metric"] == "total_stopped"]
        if not metrics.empty:
            improvement = metrics.iloc[0]["improvement_pct"]
            st.metric(
                "Stopped Vehicles Reduction",
                f"{improvement:+.1f}%",
                delta_color="off" if improvement > 0 else "normal",
            )

    with col4:
        st.metric(
            "Metrics Compared",
            len(comparison_data),
        )

    st.markdown("---")

    # Bar chart comparison
    fig = go.Figure()

    metrics_labels = {
        "avg_waiting_time": "Avg Waiting Time (s)",
        "avg_speed": "Avg Speed (m/s)",
        "total_stopped": "Total Stopped",
        "total_waiting_time": "Total Waiting Time (s)",
    }

    # RL bars (one trace)
    fig.add_trace(go.Bar(
        name="RL (DQN)",
        x=[metrics_labels.get(row["metric"], row["metric"]) for _, row in comparison_data.iterrows()],
        y=[row["rl_mean"] for _, row in comparison_data.iterrows()],
        error_y=dict(type='data', array=[row["rl_std"] for _, row in comparison_data.iterrows()]),
        marker_color="steelblue",
    ))

    # Fixed-time bars (one trace)
    fig.add_trace(go.Bar(
        name="Fixed-Time",
        x=[metrics_labels.get(row["metric"], row["metric"]) for _, row in comparison_data.iterrows()],
        y=[row["fixed_mean"] for _, row in comparison_data.iterrows()],
        error_y=dict(type='data', array=[row["fixed_std"] for _, row in comparison_data.iterrows()]),
        marker_color="coral",
    ))

    fig.update_layout(
        title="Metric Comparison: RL vs Fixed-Time",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        hovermode='x unified',
        height=400,
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    st.subheader("Detailed Metrics")
    st.dataframe(comparison_data, use_container_width=True)

    # Per-episode trends
    st.subheader("Per-Episode Trends")

    rl_eval = load_rl_eval_data()
    fixed_eval = load_fixed_eval_data()

    if not rl_eval.empty and not fixed_eval.empty:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=rl_eval["episode"],
            y=rl_eval["avg_waiting_time"],
            mode='lines+markers',
            name="RL (DQN)",
            line=dict(color="steelblue", width=2),
        ))

        fig.add_trace(go.Scatter(
            x=fixed_eval["episode"],
            y=fixed_eval["avg_waiting_time"],
            mode='lines+markers',
            name="Fixed-Time",
            line=dict(color="coral", width=2),
        ))

        fig.update_layout(
            title="Average Waiting Time per Episode",
            xaxis_title="Episode",
            yaxis_title="Average Waiting Time (s)",
            hovermode='x unified',
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


# ── Tab: Model Checkpoints ──────────────────────────────────────────────────

def show_model_checkpoints():
    st.title("🏆 Model Checkpoints")

    checkpoint_files = sorted(glob.glob(os.path.join(MODELS_DIR, "dqn_intersection_*.zip")))

    if not checkpoint_files:
        st.warning("No model checkpoints found. Run `python train.py` first.")
        return

    st.metric("Total Checkpoints", len(checkpoint_files))

    # Checkpoint selection
    checkpoint_names = [Path(f).stem for f in checkpoint_files]
    selected_checkpoint = st.selectbox(
        "Select Checkpoint",
        checkpoint_names,
        index=len(checkpoint_names) -1,  # Latest by default
    )

    if selected_checkpoint:
        selected_path = os.path.join(MODELS_DIR, f"{selected_checkpoint}.zip")
        step = get_checkpoint_step(selected_path)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Step", f"{step:,}" if step else "Unknown")

        with col2:
            file_size_mb = os.path.getsize(selected_path) / (1024 * 1024)
            st.metric("File Size", f"{file_size_mb:.1f} MB")

        with col3:
            mtime = os.path.getmtime(selected_path)
            st.metric("Modified", pd.Timestamp(mtime, unit='s').strftime("%Y-%m-%d"))

        st.markdown("---")

    # Checkpoint progression
    st.subheader("Training Progression")

    checkpoints_data = []
    for cp_file in checkpoint_files:
        step = get_checkpoint_step(cp_file)
        if step:
            checkpoints_data.append({
                "Checkpoint": Path(cp_file).stem,
                "Step": step,
                "Progress": min(100, (step / TOTAL_TIMESTEPS) * 100),
            })

    if checkpoints_data:
        cp_df = pd.DataFrame(checkpoints_data)

        fig = px.bar(
            cp_df,
            x="Checkpoint",
            y="Progress",
            title="Model Training Progress",
            labels={"Progress": "% of Total Training"},
            height=300,
        )

        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Target")
        st.plotly_chart(fig, use_container_width=True)


# ── Tab: TensorBoard ────────────────────────────────────────────────────────

def show_tensorboard():
    st.title("📺 TensorBoard Integration")

    tb_log_dir = os.path.join(OUTPUTS_DIR, "tb_logs")
    tb_subdir = None

    # Find TensorBoard log subdirectory
    if os.path.isdir(tb_log_dir):
        subdirs = [d for d in os.listdir(tb_log_dir)
                   if os.path.isdir(os.path.join(tb_log_dir, d))]
        if subdirs:
            tb_subdir = sorted(subdirs)[-1]

    if not tb_subdir or not os.path.isdir(os.path.join(tb_log_dir, tb_subdir)):
        st.warning("No TensorBoard logs found. Run `python train.py` to generate logs.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info(
            """
            **To view TensorBoard in your browser:**

            ```bash
            tensorboard --logdir=outputs/tb_logs/
            ```

            Then open: http://localhost:6006
            """
        )

    with col2:
        if st.button("📋 Copy Command", use_container_width=True):
            st.success("Command copied!")

    st.markdown("---")

    # Display TensorBoard info
    tb_full_dir = os.path.join(tb_log_dir, tb_subdir)
    st.subheader(f"Active Log Directory: {tb_subdir}")

    col1, col2, col3 = st.columns(3)

    event_files = glob.glob(os.path.join(tb_full_dir, "events.out.tfevents.*"))
    with col1:
        st.metric("Event Files", len(event_files))

    # Find latest event file
    if event_files:
        latest_event = max(event_files, key=os.path.getctime)
        mtime = os.path.getmtime(latest_event)
        with col2:
            time_str = pd.Timestamp(mtime, unit='s').strftime("%H:%M:%S")
            st.metric("Latest Event", time_str)

        with col3:
            file_size_mb = os.path.getsize(latest_event) / (1024 * 1024)
            st.metric("Latest File Size", f"{file_size_mb:.1f} MB")

    st.markdown("---")

    st.subheader("📊 Key Metrics")

    st.markdown("""
    TensorBoard logs the following metrics during training:

    **RL Agent Metrics:**
    - `rollout/ep_len_mean` - Average episode length
    - `rollout/ep_rew_mean` - Average episode reward
    - `train/learning_rate` - Learning rate schedule
    - `train/loss` - DQN loss function value
    - `rollout/es_bar` - Exploration epsilon value

    **To explore metrics live:**
    1. Start TensorBoard: `tensorboard --logdir=outputs/tb_logs/`
    2. Open http://localhost:6006 in your browser
    3. Select the "Scalars" tab to view metrics over time
    4. Use the "Histograms" tab to see weight distributions
    """)



# ── Tab: Help & Settings ────────────────────────────────────────────────────

def show_help_settings():
    st.title("⚙️ Help & Settings")

    help_tab, settings_tab = st.tabs(["📖 Help", "⚙️ Settings"])

    with help_tab:
        st.subheader("Getting Started")

        st.markdown("""
        ### Step 1: Generate Network
        ```bash
        python generate_network.py
        ```
        Creates the 4-way intersection network and routes.

        ### Step 2: Train Model
        ```bash
        python train.py
        ```
        Trains a DQN agent to control traffic signals.

        ### Step 3: Evaluate & Compare
        ```bash
        python evaluate.py
        ```
        Compares trained RL agent with fixed-time baseline.


        ### Step 4: View Dashboard (This App!)
        ```bash
        streamlit run dashboard.py
        ```
        Interactive visualization of all results.

        ---

        ### Tips
        - **Real-time monitoring:** Start `streamlit run dashboard.py` in one terminal,
          then `python train.py` in another. The dashboard will auto-refresh.
        - **TensorBoard:** Run `tensorboard --logdir=outputs/tb_logs/` for detailed metrics.
        - **Faster iterations:** Use `--episodes 2` and `--timesteps 50000` for quick testing.
        """)

    with settings_tab:
        st.subheader("Directory Structure")

        dirs = {
            "nets/": "Network files (.net.xml, .rou.xml, .sumocfg)",
            "models/": "Trained DQN checkpoints (.zip files)",
            "outputs/": "Training/evaluation logs and metrics",
            "outputs/training/": "Per-episode training CSVs",
            "outputs/evaluation/": "Evaluation comparison results",
            "outputs/tb_logs/": "TensorBoard event files",
            "results/": "Final visualizations (PNG plots)",
        }

        for dir_name, description in dirs.items():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.code(dir_name, language="text")
            with col2:
                st.write(description)

        st.markdown("---")

        st.subheader("Configuration")
        st.markdown("""
        All parameters are centralized in **config.py**:

        - **Network:** `NUM_LANES`, `APPROACH_LENGTH`, `SPEED_LIMIT`
        - **Traffic:** `VEHICLE_PERIOD`, `AMBULANCE_PERIOD`
        - **RL:** Learning rate, network architecture, exploration schedule
        - **Training:** Timesteps, checkpoint frequency, batch size
        """)

        st.markdown("---")

        st.subheader("System Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            results_dir = RESULTS_DIR
            st.metric(
                "Results Directory",
                results_dir.split(os.sep)[-1],
            )

        with col2:
            models_dir = MODELS_DIR
            model_count = len(glob.glob(os.path.join(models_dir, "*.zip")))
            st.metric("Trained Models", model_count)

        with col3:
            outputs_dir = OUTPUTS_DIR
            if os.path.isdir(outputs_dir):
                csv_count = len(glob.glob(os.path.join(outputs_dir, "**/*.csv")))
                st.metric("CSV Files", csv_count)


# ── Main App Router ─────────────────────────────────────────────────────────

if tab == "📊 Training Progress":
    show_training_progress()
elif tab == "📈 Comparison Metrics":
    show_comparison_metrics()
elif tab == "🏆 Model Checkpoints":
    show_model_checkpoints()
elif tab == "📺 TensorBoard":
    show_tensorboard()
elif tab == "⚙️ Help & Settings":
    show_help_settings()

# Footer
st.markdown("---")
st.markdown(
    "🚦 SUMO 4-Way Intersection RL Dashboard | "
    "[GitHub](https://github.com/KAYO-ult/sumo_4lane_intersection_with_priority_vechicle) | [SUMO](http://sumo.dlr.de)"
)
