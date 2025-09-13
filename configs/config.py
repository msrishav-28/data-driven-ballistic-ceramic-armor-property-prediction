import os
from pathlib import Path
class Config:
"""Central configuration for the ceramic armor ML project"""
# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
BALLISTIC_DATA_DIR = DATA_DIR / "ballistic"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# API Keys (set as environment variables)
MP_API_KEY = os.getenv("MP_API_KEY", "your_mp_api_key_here")

# Target ceramic systems
CERAMIC_SYSTEMS = ["Si-C", "B-C", "W-C", "Ti-C", "Al-O"]
TARGET_MATERIALS = ["SiC", "B4C", "WC", "TiC", "Al2O3"]

# Model performance targets
MECHANICAL_R2_TARGET = 0.85
BALLISTIC_R2_TARGET = 0.80
SCREENING_REDUCTION_TARGET = 0.60

# Model hyperparameters
RANDOM_STATE = 42
CV_FOLDS = 5
N_JOBS = -1  # Use all CPU cores

# Feature engineering settings
MIN_FEATURES = 150
FEATURE_SELECTION_THRESHOLD = 0.01

# Optimization settings
OPTUNA_N_TRIALS = 200
EARLY_STOPPING_ROUNDS = 50