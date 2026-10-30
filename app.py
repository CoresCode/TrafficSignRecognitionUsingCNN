"""
============================================================
German Traffic Sign Recognition - Flask Web Backend Server
============================================================
Serves the web application UI and REST API for model inference.
"""

import os
import io
import sys
import time
import json
import base64
from pathlib import Path

# Fix Windows console UTF-8 printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from utils.model import TrafficSignCNN
from utils.transforms import get_valid_transforms
from utils.sign_names import GTSRB_CLASSES, get_sign_info

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DATASET_DIR = PROJECT_ROOT / "dataset" / "raw"

# PyTorch setup
NUM_CLASSES = 43
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global model instance
model = None


def load_model():
    """Load PyTorch CNN model once at application startup."""
    global model
    if model is not None:
        return model

    print(f"🔄 Loading PyTorch CNN Model on device: {DEVICE}...")
    model_inst = TrafficSignCNN(NUM_CLASSES).to(DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model_inst.load_state_dict(checkpoint["model_state_dict"])
    else:
        model_inst.load_state_dict(checkpoint)

    model_inst.eval()
    model = model_inst
    print("✅ Model loaded successfully!")
    return model


# Warm up model on startup
try:
    load_model()
except Exception as err:
    print(f"⚠️ Warning: Could not auto-load model on startup ({err}). Will attempt on request.")


@app.route("/")
def index():
    """Render main web application interface."""
    return render_template("index.html")


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return dictionary of all 43 GTSRB classes with metadata."""
    return jsonify({
        "status": "success",
        "count": len(GTSRB_CLASSES),
        "classes": GTSRB_CLASSES
    })


@app.route("/api/history", methods=["GET"])
def get_training_history():
    """Return model training history for interactive Chart.js visualization."""
    history_file = OUTPUTS_DIR / "training_history.json"
    if not history_file.exists():
        return jsonify({"status": "error", "message": "Training history not found"}), 4404
    
    with open(history_file, "r") as f:
        data = json.load(f)
    
    return jsonify({
        "status": "success",
        "data": data
    })


@app.route("/api/samples", methods=["GET"])
def get_sample_images():
    """Return list of real GTSRB test images converted to PNG base64 data URLs for web browsers."""
    test_dir = RAW_DATASET_DIR / "GTSRB" / "Final_Test" / "Images"
    csv_file = RAW_DATASET_DIR / "GT-final_test.csv"
    
    samples = []
    if csv_file.exists() and test_dir.exists():
        import pandas as pd
        df = pd.read_csv(csv_file, sep=";")
        
        # Pick 6 diverse samples
        indices = [1, 4, 7, 8, 14, 18]
        for idx in indices:
            if idx < len(df):
                row = df.iloc[idx]
                filename = row["Filename"]
                class_id = int(row["ClassId"])
                img_path = test_dir / filename
                if img_path.exists():
                    try:
                        # Open PPM image and convert to PNG in-memory
                        pil_img = Image.open(img_path).convert("RGB")
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                        
                        info = get_sign_info(class_id)
                        samples.append({
                            "filename": filename,
                            "class_id": class_id,
                            "name": info["name"],
                            "icon": info["icon"],
                            "data_url": f"data:image/png;base64,{b64_data}"
                        })
                    except Exception as err:
                        print(f"Error encoding sample {filename}: {err}")
    
    return jsonify({"status": "success", "samples": samples})




@app.route("/outputs/<path:filename>")
def serve_output_file(filename):
    """Serve output artifacts like confusion_matrix.png."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept image file upload or base64 image string, preprocess,
    run model inference, and return top predictions.
    """
    start_time = time.time()
    image_bytes = None

    # Handle FormData file upload or Base64 JSON payload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected"}), 400
        image_bytes = file.read()

    elif request.is_json:
        req_data = request.get_json()
        if "image" in req_data:
            base64_str = req_data["image"]
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_bytes = base64.b64decode(base64_str)

    if not image_bytes:
        return jsonify({"status": "error", "message": "No valid image provided"}), 400

    try:
        # Convert bytes to PIL RGB Image -> OpenCV numpy array
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_img)

        # Preprocess image using project validation transforms
        transform = get_valid_transforms()
        transformed = transform(image=image_np)["image"]
        input_tensor = transformed.unsqueeze(0).to(DEVICE)

        # Perform Inference
        active_model = load_model()
        with torch.no_grad():
            logits = active_model(input_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        # Top-5 predictions
        top_probs, top_indices = torch.topk(probabilities, k=min(5, NUM_CLASSES))

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_id = idx.item()
            conf_val = float(prob.item() * 100)
            sign_meta = get_sign_info(class_id)
            predictions.append({
                "class_id": class_id,
                "confidence": round(conf_val, 2),
                "info": sign_meta
            })

        latency_ms = round((time.time() - start_time) * 1000, 2)
        top_prediction = predictions[0]
        
        # Smart Top-3 Cumulative Energy Rule (Non-Traffic Sign Detector)
        # Real signs concentrate energy in Top-3 (e.g. 37% + 27% + 9% = 73%), while random noise is flat (< 35%)
        top_3_energy = sum(p["confidence"] for p in predictions[:3])
        ENERGY_THRESHOLD = 35.0
        is_valid_sign = top_3_energy >= ENERGY_THRESHOLD

        return jsonify({
            "status": "success",
            "inference_time_ms": latency_ms,
            "is_valid_sign": is_valid_sign,
            "top_3_energy": round(top_3_energy, 2),
            "energy_threshold": ENERGY_THRESHOLD,
            "top_prediction": top_prediction,
            "top_k": predictions
        })



    except Exception as e:
        print(f"❌ Error in /api/predict: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Starting German Traffic Sign Recognition Web App Server...")
    print("=" * 65)
    print(f"Listening on: http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host="127.0.0.1", port=5000, debug=True)
