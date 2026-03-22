"""
config.py - Centralized configuration for single 4-way intersection
              AI traffic signal control using SUMO-RL + DQN.

All tunable parameters live here so every script imports from one place.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NETS_DIR = os.path.join(BASE_DIR, "nets")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

NET_FILE = os.path.join(NETS_DIR, "intersection.net.xml")
ROUTE_FILE = os.path.join(NETS_DIR, "intersection.rou.xml")
SUMOCFG_FILE = os.path.join(NETS_DIR, "intersection.sumocfg")

# ── Network Geometry ───────────────────────────────────────────────────
APPROACH_LENGTH = 300          # meters — length of each approach road
NUM_LANES = 3                  # lanes per direction on each approach
SPEED_LIMIT = 13.89            # m/s  (≈ 50 km/h, typical Indian urban road)
LEFT_HAND_TRAFFIC = True       # Indian left-hand traffic

# ── Priority Vehicles (Ambulance) ──────────────────────────────────
AMBULANCE_PERIOD = 30.0        # one ambulance every 30 seconds
AMBULANCE_SPEED = 25.0         # m/s (≈ 90 km/h, priority speed)

# ── Route / Demand Generation ─────────────────────────────────────────
SIMULATION_DURATION = 3600     # seconds (1 hour)
VEHICLE_PERIOD = 1.5           # one vehicle generated every 1.5 s (≈ 2 400 veh/hr)
RANDOM_SEED = 42

# ── SUMO-RL Environment ───────────────────────────────────────────────
NUM_SECONDS = 3600             # episode length (sim seconds)
DELTA_TIME = 5                 # seconds between RL agent decisions
YELLOW_TIME = 3                # yellow phase duration
MIN_GREEN = 5                  # minimum green phase
MAX_GREEN = 60                 # maximum green phase
REWARD_FN = "diff-waiting-time"  # reward = reduction in cumulative delay

# ── DQN Hyper-parameters ──────────────────────────────────────────────
LEARNING_RATE = 1e-3
BUFFER_SIZE = 50_000
LEARNING_STARTS = 800          # steps of random exploration before training
BATCH_SIZE = 128
TARGET_UPDATE_INTERVAL = 500   # steps between target-network syncs
TRAIN_FREQ = 1                 # update model every step
EXPLORATION_INITIAL_EPS = 0.10
EXPLORATION_FINAL_EPS = 0.02
EXPLORATION_FRACTION = 0.30    # fraction of total steps for ε-decay
GAMMA = 0.99                   # discount factor
TOTAL_TIMESTEPS = 100_000
CHECKPOINT_FREQ = 10_000       # save checkpoint every N steps
NET_ARCH = [256, 256]          # MLP hidden layers

# ── Evaluation ─────────────────────────────────────────────────────────
EVAL_EPISODES = 5
