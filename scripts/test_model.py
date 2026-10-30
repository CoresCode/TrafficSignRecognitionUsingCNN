import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch

from utils.model import TrafficSignCNN

model = TrafficSignCNN(num_classes=57)

print(model)

dummy = torch.randn(4, 3, 96, 96)

output = model(dummy)

print("\nOutput Shape:", output.shape)