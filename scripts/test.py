from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import json

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from tqdm import tqdm

from utils.transforms import get_valid_transforms
from utils.model import TrafficSignCNN


# ==========================================================
# Configuration
# ==========================================================

TEST_DIR = Path("dataset/raw/GTSRB/Final_Test/Images")

CSV_PATH = Path("dataset/raw/GT-final_test.csv")

MODEL_PATH = Path("models/best_model.pt")

OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 43

BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Dataset
# ==========================================================

class GTSRBTestDataset(Dataset):

    def __init__(
        self,
        image_dir,
        csv_file,
        transform=None
    ):

        self.image_dir = Path(image_dir)

        self.transform = transform

        self.annotations = pd.read_csv(
            csv_file,
            sep=";"
        )

    def __len__(self):

        return len(self.annotations)

    def __getitem__(self, index):

        row = self.annotations.iloc[index]

        image_path = self.image_dir / row["Filename"]

        image = cv2.imread(str(image_path))

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform:

            image = self.transform(
                image=image
            )["image"]

        label = torch.tensor(
            int(row["ClassId"]),
            dtype=torch.long
        )

        return image, label, row["Filename"]

# ==========================================================
# Load Dataset
# ==========================================================

test_dataset = GTSRBTestDataset(
    TEST_DIR,
    CSV_PATH,
    transform=get_valid_transforms()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ==========================================================
# Load Model
# ==========================================================

model = TrafficSignCNN(NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("=" * 60)
print("Traffic Sign Recognition Test")
print("=" * 60)
print(f"Device      : {DEVICE}")
print(f"Test Images : {len(test_dataset)}")
print("=" * 60)


# ==========================================================
# Inference
# ==========================================================

all_predictions = []
all_labels = []
prediction_rows = []

with torch.no_grad():

    progress_bar = tqdm(
        test_loader,
        desc="Testing"
    )

    for images, labels, filenames in progress_bar:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        for filename, pred, true in zip(
            filenames,
            predictions.cpu().numpy(),
            labels.cpu().numpy()
        ):

            prediction_rows.append(
                {
                    "Filename": filename,
                    "GroundTruth": int(true),
                    "Prediction": int(pred)
                }
            )

# ==========================================================
# Metrics
# ==========================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels,
    all_predictions,
    average="weighted"
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

report = classification_report(
    all_labels,
    all_predictions,
    digits=4
)

print(f"\nTest Accuracy : {accuracy * 100:.2f}%")
print(f"Precision    : {precision * 100:.2f}%")
print(f"Recall       : {recall * 100:.2f}%")
print(f"F1 Score     : {f1 * 100:.2f}%")


# ==========================================================
# Save Results
# ==========================================================

pd.DataFrame(
    prediction_rows
).to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as f:

    f.write(report)

with open(
    OUTPUT_DIR / "test_results.json",
    "w"
) as f:

    json.dump(
        {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        f,
        indent=4
    )


# ==========================================================
# Confusion Matrix
# ==========================================================

plt.figure(figsize=(14, 14))

plt.imshow(
    cm,
    interpolation="nearest",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


print("\n" + "=" * 60)
print("Testing Complete")
print("=" * 60)
print(f"Accuracy      : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")
print()
print(f"Predictions   : {OUTPUT_DIR / 'predictions.csv'}")
print(f"Report        : {OUTPUT_DIR / 'classification_report.txt'}")
print(f"Results       : {OUTPUT_DIR / 'test_results.json'}")
print(f"Confusion Mat : {OUTPUT_DIR / 'confusion_matrix.png'}")
print("=" * 60)