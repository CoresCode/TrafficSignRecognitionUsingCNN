import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt

from utils.dataset import TrafficSignDataset
from utils.transforms import get_train_transforms

transform = get_train_transforms()

dataset = TrafficSignDataset(
    dataset_folder="dataset/raw/train",
    transform=transform
)

print(f"Total Samples: {len(dataset)}")

plt.figure(figsize=(10, 10))

for i in range(9):
    image, label = dataset[i]

    plt.subplot(3, 3, i + 1)

    # Albumentations returns a Tensor directly
    plt.imshow(image.permute(1, 2, 0))

    plt.title(dataset.get_class_names()[label.item()], fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.show()