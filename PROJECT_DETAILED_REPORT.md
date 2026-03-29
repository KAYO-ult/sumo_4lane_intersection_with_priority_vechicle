# Abstract

This project presents an AI-based adaptive traffic signal control system for a single 4-way intersection using SUMO simulation and Deep Q-Network (DQN) reinforcement learning. The system replaces fixed-time traffic signaling with a data-driven policy that observes real-time traffic state and selects green phases dynamically to reduce delay and improve flow. A priority-vehicle module is integrated for ambulance handling, including route generation, XML sanitization safeguards, and runtime emergency preemption logic to minimize ambulance delay at signals. The implementation combines SUMO, sumo-rl, Stable-Baselines3, and Gymnasium, with output analysis through CSV logs, static plots, TensorBoard, and a Streamlit dashboard. Experimental comparison against fixed-time signaling shows strong gains: 76.81% reduction in average waiting time, 78.10% reduction in total waiting time, 54.75% reduction in total stopped vehicles, and 15.26% increase in average speed.

---

# Chapter 1

## 1.1 Introduction

Urban traffic congestion is a persistent challenge caused by growing vehicle populations and static traffic control strategies. Conventional fixed-time traffic lights run pre-defined signal cycles regardless of current traffic conditions, leading to avoidable delay, queue spillback, and inefficient intersection utilization.

This project addresses the problem through reinforcement learning, where a DQN agent learns control actions from interaction with a simulated 4-way intersection environment. The agent repeatedly observes lane-level traffic conditions and chooses a signal phase to optimize traffic flow.

The key objectives of the project are:
1. Build a configurable, reproducible SUMO-based intersection model.
2. Train an RL agent to minimize waiting-time-based congestion metrics.
3. Benchmark RL performance against fixed-time control.
4. Include priority ambulance traffic and emergency pass-through behavior.

The implemented system targets practical experimentation and reporting, with automated scripts for network generation, training, evaluation, plotting, dashboarding, and dataset preparation.

---

# Chapter 2

## 2.1 Literature Survey

Traffic signal optimization has historically progressed through three major categories:

1. Fixed-time control:
Precomputed cycle plans based on historical averages. These methods are easy to deploy but weak under demand variability and incident-driven traffic shocks.

2. Actuated/adaptive rule-based control:
Sensor-triggered phase adjustments improve responsiveness but still depend on handcrafted heuristics and limited adaptation depth.

3. Reinforcement learning-based control:
RL models learn policies directly from interaction data and can adapt to complex, time-varying patterns. Deep RL approaches, especially value-based methods such as DQN, are widely used for single-intersection traffic control due to manageable action spaces and stable off-policy training.

In this project, DQN is selected for the single-agent setting, leveraging experience replay and target-network updates for stable learning. The work extends conventional RL intersection control by explicitly incorporating emergency-vehicle handling and runtime safety-compatible priority logic.

---

# Chapter 3

## 3.1 Experimental Dataset

The project uses simulation-generated traffic data rather than externally collected datasets.

### 3.1.1 Data Generation Source
Traffic demand is generated in SUMO using route generation scripts for three non-emergency demand profiles:
1. Normal traffic: period = 1.5 s.
2. Low traffic: period = 3.0 s.
3. High traffic: period = 0.8 s.

Priority vehicles (ambulances) are generated separately:
1. Ambulance period = 30 s.
2. Vehicle class = emergency.
3. Target speed = 25 m/s.

### 3.1.2 Simulation and Logging Data
Training and evaluation logs are written to CSV files under outputs, including:
1. outputs/training/*.csv
2. outputs/evaluation/rl_summary.csv
3. outputs/evaluation/fixed_summary.csv
4. outputs/evaluation/comparison.csv

### 3.1.3 Features and Metrics
The primary monitored metrics are:
1. system_mean_waiting_time
2. system_total_waiting_time
3. system_mean_speed
4. system_total_stopped

### 3.1.4 Prepared Dataset Splits
The project provides script-based split generation into:
1. outputs/dataset/train.csv
2. outputs/dataset/validation.csv
3. outputs/dataset/test.csv
4. outputs/dataset/metadata.json

---

# Chapter 4

## 4.1 Proposed Methodology

### 4.1.1 System Architecture
The control loop follows:

┌──────────────┐     TraCI      ┌──────────────┐    Gymnasium   ┌──────────────┐
│     SUMO     │ ◄────────────► │   sumo-rl    │ ◄────────────► │   SB3 DQN    │
│  (Simulator) │                │  (Env Wrap)  │                │   (Agent)    │
└──────────────┘                └──────────────┘                └──────────────┘

### 4.1.2 Environment Configuration
Key environment parameters:
1. Intersection type: single 4-way, 3 lanes per direction.
2. Left-hand traffic: enabled.
3. Episode duration: 3600s.
4. Decision interval: 5s.
5. Yellow time: 3s.    
6. Green time constraints: 5s to 60s.

### 4.1.3 DQN Training Setup
Model uses MLP policy with architecture [256, 256], learning rate 1e-3, replay buffer 50,000, batch size 128, gamma 0.99, and 100,000 training timesteps by default.

### 4.1.4 Reward Formulation
The reward function diff-waiting-time is used to encourage reductions in cumulative waiting time over control steps.

### 4.1.5 Priority Ambulance Logic
The implemented emergency module includes:
1. Dedicated ambulance route file.
2. XML sanitation for valid SUMO parsing.
3. Duplicate ID prevention for multi-route loading.
4. Emergency visual coding (red vehicle type).
5. Runtime speed-mode override to reduce red-signal waiting for ambulances.

### 4.1.6 Workflow
1. Generate network and routes.
2. Train RL model.
3. Evaluate RL and fixed-time baselines.
4. Produce plots and dashboard analytics.

---

# Chapter 5

## 5.1 Results and Discussion

Evaluation is performed over 5 episodes for RL and fixed-time policies under equivalent simulation settings.

### 5.1.1 Quantitative Results

| Metric | RL Mean | Fixed Mean | Improvement |
|---|---:|---:|---:|
| avg_waiting_time | 0.9037 | 3.8970 | 76.81% |
| avg_speed | 9.5405 | 8.2775 | 15.26% |
| total_stopped | 4.4872 | 9.9175 | 54.75% |
| total_waiting_time | 32.5033 | 148.4064 | 78.10% |

### 5.1.2 Discussion
1. Delay metrics improve strongly, indicating that RL effectively reallocates green time according to live traffic conditions.
2. Speed increase confirms reduced interruption and smoother corridor movement through the intersection.
3. Reduction in stopped vehicles shows lower queue buildup and less stop-and-go behavior.
4. Emergency handling enhancements improve practical realism and support ambulance pass-through requirements.

### 5.1.3 Visualization Outputs
Generated analysis figures:
1. results/training_curves.png
2. results/comparison_metrics.png
3. results/episode_comparison.png

Interactive monitoring:
1. Streamlit dashboard for trend and checkpoint analysis.
2. TensorBoard logs for training diagnostics.

---

# Chapter 6

## 6.1 Conclusions

The project successfully demonstrates that DQN-based adaptive signal control significantly outperforms fixed-time control for the modeled 4-way intersection. The implementation is reproducible, modular, and experimentally validated through episode-level comparisons. The system also includes working ambulance-priority mechanisms with route integrity safeguards and runtime emergency preemption.

## 6.2 Future Work

1. Extend from single intersection to multi-intersection coordinated control.
2. Evaluate robustness under unseen traffic distributions and incident scenarios.
3. Add richer emergency KPIs such as ambulance travel-time percentiles.
4. Compare DQN with PPO, A2C, and multi-agent RL alternatives.
5. Incorporate fairness constraints to balance emergency priority with general traffic impact.

---

# References

1. SUMO Documentation. Eclipse SUMO. https://sumo.dlr.de/docs/
2. sumo-rl Documentation and Repository. https://github.com/LucasAlegre/sumo-rl
3. Stable-Baselines3 Documentation. https://stable-baselines3.readthedocs.io/
4. Gymnasium Documentation. https://gymnasium.farama.org/
5. Mnih, V., et al. "Human-level control through deep reinforcement learning." Nature, 2015.
