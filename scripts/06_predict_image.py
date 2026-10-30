"""
============================================================
Traffic Sign Detection - Standalone Image Predictor CLI
============================================================
Usage:
    python scripts/06_predict_image.py --image path/to/image.jpg
"""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import cv2
import torch
import torch.nn.functional as F

from utils.model import TrafficSignCNN
from utils.transforms import get_valid_transforms
from utils.sign_names import get_sign_info

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
NUM_CLASSES = 43
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model():
    """Load the TrafficSignCNN model with trained weights."""
    model = TrafficSignCNN(NUM_CLASSES).to(DEVICE)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def predict_image(image_path: str, top_k: int = 5):
    """Predict traffic sign class for a single image file."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image using OpenCV and convert BGR to RGB
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image file: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess image
    transform = get_valid_transforms()
    transformed = transform(image=image_rgb)["image"]
    tensor = transformed.unsqueeze(0).to(DEVICE)  # Add batch dim

    # Load model and predict
    model = load_trained_model()
    with torch.no_grad():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    # Extract Top-K predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_id = idx.item()
        confidence = prob.item() * 100
        sign_info = get_sign_info(class_id)
        results.append({
            "class_id": class_id,
            "confidence": confidence,
            "sign_info": sign_info
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Traffic Sign Image Classifier")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top candidate classes to display")
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("Traffic Sign Detection - Single Image Prediction")
    print("=" * 65)
    print(f"Device : {DEVICE}")
    print(f"Image  : {args.image}")
    print("=" * 65)

    try:
        predictions = predict_image(args.image, top_k=args.top_k)
        top = predictions[0]
        
        print(f"\n🏆 Top Prediction:")
        print(f"   Class ID   : {top['class_id']}")
        print(f"   Sign Name  : {top['sign_info']['name']}")
        print(f"   German Name: {top['sign_info']['german_name']}")
        print(f"   Category   : {top['sign_info']['category']}")
        print(f"   Confidence : {top['confidence']:.2f}%")
        print(f"   Rule       : {top['sign_info']['description']}")

        print(f"\n📊 Top {len(predictions)} Candidate Breakdown:")
        for i, pred in enumerate(predictions, 1):
            print(f"   {i}. Class {pred['class_id']:02d} | {pred['sign_info']['name']:<45} | {pred['confidence']:6.2f}%")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"❌ Error during prediction: {e}")


if __name__ == "__main__":
    main()
