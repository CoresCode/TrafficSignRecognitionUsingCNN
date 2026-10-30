from pathlib import Path
import json
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    random_split
)
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from utils.dataset import TrafficSignDataset
from utils.transforms import (
    get_train_transforms,
    get_valid_transforms
)
from utils.model import TrafficSignCNN


# ==========================================================
# Configuration
# ==========================================================

TRAIN_DIR = "dataset/raw/GTSRB/Final_Training/Images"

NUM_CLASSES = 43

IMAGE_SIZE = 128

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("models/best_model.pt")
MODEL_PATH.parent.mkdir(exist_ok=True)


# ==========================================================
# Training Function
# ==========================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress_bar:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================================
# Validation Function
# ==========================================================

def validate_one_epoch(
    model,
    valid_loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        valid_loader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for images, labels in progress_bar:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device.type):

                outputs = model(images)

                loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

    epoch_loss = running_loss / len(valid_loader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


# ==========================================================
# Main
# ==========================================================

def main():

    train_dataset = TrafficSignDataset(
    TRAIN_DIR,
    transform=get_train_transforms()
    )

    valid_dataset = TrafficSignDataset(
        TRAIN_DIR,
        transform=get_valid_transforms()
    )

    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_subset, _ = random_split(
        train_dataset,
        [train_size, valid_size],
        generator=generator
    )

    _, valid_subset = random_split(
        valid_dataset,
        [train_size, valid_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = TrafficSignCNN(NUM_CLASSES).to(DEVICE)

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = GradScaler(
        enabled=(DEVICE.type == "cuda")
    )

    best_accuracy = 0.0
    early_stop_counter = 0

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "learning_rate": []
    }

    print("=" * 65)
    print("Traffic Sign Recognition Training")
    print("=" * 65)
    print(f"Device      : {DEVICE}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch Size  : {BATCH_SIZE}")
    print(f"Image Size  : {IMAGE_SIZE}")
    print("=" * 65)

    for epoch in range(EPOCHS):

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            DEVICE
        )

        valid_loss, valid_acc = validate_one_epoch(
            model,
            valid_loader,
            criterion,
            DEVICE
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_acc)
        history["learning_rate"].append(current_lr)

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Train Acc  : {train_acc:.2f}%")
        print(f"Valid Loss : {valid_loss:.4f}")
        print(f"Valid Acc  : {valid_acc:.2f}%")
        print(f"LR         : {current_lr:.8f}")

        if valid_acc > best_accuracy:

            best_accuracy = valid_acc
            early_stop_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_accuracy": best_accuracy,
                },
                MODEL_PATH,
            )

            print("\n✅ Best model saved!")

        else:

            early_stop_counter += 1

            print(
                f"\nNo improvement ({early_stop_counter}/{PATIENCE})"
            )

        if early_stop_counter >= PATIENCE:

            print("\n🛑 Early stopping triggered.")
            break

    history_path = Path("outputs/training_history.json")
    history_path.parent.mkdir(exist_ok=True)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\n" + "=" * 65)
    print("Training Complete")
    print("=" * 65)
    print(f"Best Validation Accuracy : {best_accuracy:.2f}%")
    print(f"Model Saved To           : {MODEL_PATH}")
    print(f"History Saved To         : {history_path}")
    print("=" * 65)


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()