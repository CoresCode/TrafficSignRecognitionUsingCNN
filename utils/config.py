"""
============================================================
Traffic Sign Detection using CNN
Configuration File
============================================================
"""

from pathlib import Path
import torch

# ============================================================
# Project Paths
# ============================================================

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Paths
RAW_DATASET_PATH = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed"

# Output Paths
OUTPUT_PATH = PROJECT_ROOT / "outputs"
GRAPHS_PATH = OUTPUT_PATH / "graphs"
REPORTS_PATH = OUTPUT_PATH / "reports"
SAMPLES_PATH = OUTPUT_PATH / "sample_annotations"

# Model Path
MODEL_PATH = PROJECT_ROOT / "models"

# ============================================================
# Dataset Information
# ============================================================

NUM_CLASSES = 43

TRAIN_FOLDER = RAW_DATASET_PATH / "train"
VALID_FOLDER = RAW_DATASET_PATH / "valid"
TEST_FOLDER = RAW_DATASET_PATH / "test"

# ============================================================
# Image Parameters
# ============================================================

IMAGE_SIZE = 128

# ============================================================
# Training Parameters
# ============================================================

BATCH_SIZE = 64

LEARNING_RATE = 0.001

EPOCHS = 100

RANDOM_SEED = 42

# ============================================================
# Hardware
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Console Information
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Traffic Sign Detection Configuration")
    print("=" * 60)

    print(f"Project Root : {PROJECT_ROOT}")
    print(f"Dataset Path : {RAW_DATASET_PATH}")
    print(f"Device       : {DEVICE}")
    print(f"Image Size   : {IMAGE_SIZE}")
    print(f"Epochs       : {EPOCHS}")