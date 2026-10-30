# 🚦 Traffic Sign Recognition using Deep Learning (PyTorch + Flask)

A full-stack, high-performance Traffic Sign Recognition web application powered by **PyTorch** and **Flask**. The system leverages a custom Convolutional Neural Network (CNN) trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset to accurately classify traffic signs across **43 distinct categories** with **98.92% accuracy**.

---

## 🌟 Key Features

- **⚡ Real-Time Classification:** Upload or drag-and-drop any traffic sign image to get instantaneous inference results with class names, categories, and confidence percentages.
- **🧠 Custom CNN Architecture:** Multi-stage convolutional pipeline designed with PyTorch featuring Batch Normalization, Dropout regularization, and Adaptive Average Pooling.
- **🛡️ Invalid / OOD Sign Detection:** Incorporates energy-based confidence thresholding to detect and flag non-sign images, corrupted inputs, or out-of-distribution uploads.
- **📊 Top-5 Predictions:** Detailed probabilistic breakdown with interactive confidence bars and sign metadata.
- **📈 Comprehensive Model Dashboard:** Built-in dashboard showcasing training/validation loss & accuracy curves, test metrics, and a high-resolution confusion matrix.
- **🧪 Sample Testing Gallery:** Ready-to-use sample traffic signs included for one-click testing without needing to download external files.
- **🎨 Modern Responsive UI:** Clean, dark-mode inspired web interface built with modern HTML5, CSS3, Vanilla JS, and Chart.js.

---

## 📊 Model Performance

Evaluated rigorously on the official GTSRB benchmark test set:

| Metric | Score |
|:-------|------:|
| **Accuracy** | **98.92%** |
| **Precision** | **98.95%** |
| **Recall** | **98.92%** |
| **F1-Score** | **98.91%** |

Evaluation artifacts including `confusion_matrix.png`, `classification_report.txt`, and `training_history.json` are available in the `outputs/` directory.

---

## 🛠️ Tech Stack

- **Deep Learning Framework:** PyTorch, Torchvision
- **Backend Server:** Flask, Flask-CORS
- **Computer Vision & Preprocessing:** OpenCV, Albumentations, Pillow, NumPy
- **Evaluation & Analytics:** Scikit-learn, Matplotlib, Pandas
- **Frontend Interface:** HTML5, CSS3, JavaScript (ES6+), Chart.js

---

## 📁 Project Structure

```text
TrafficSignRecognitionUsingCNN/
├── app.py                     # Flask web server & inference REST API
├── train.py                   # PyTorch CNN training pipeline
├── extract_samples.py         # Utility to extract representative samples
├── requirements.txt           # Project dependencies
├── .gitignore                 # Git ignore configuration
├── LICENSE                    # Unlicense (Public Domain)
├── README.md                  # Documentation
│
├── models/
│   └── best_model.pt          # Trained CNN model weights (98.92% accuracy)
│
├── outputs/                   # Evaluation results & visualizations
│   ├── confusion_matrix.png   # High-resolution confusion matrix plot
│   ├── classification_report.txt # Per-class precision, recall, f1-scores
│   ├── test_results.json      # Overall summary metrics
│   └── training_history.json  # Epoch-by-epoch loss and accuracy metrics
│
├── ppt_samples/               # Sample sign images for quick testing
│   ├── No_Entry.png
│   ├── Right_of_Way.png
│   ├── Speed_Limit_30.png
│   ├── Stop.png
│   ├── Turn_Right_Ahead.png
│   └── Yield.png
│
├── scripts/                   # Standalone testing & prediction scripts
│   ├── 06_predict_image.py    # CLI prediction tool for single images
│   ├── 07_streamlit_app.py    # Legacy launcher redirecting to app.py
│   ├── test.py                # Full evaluation script on test set
│   ├── test_dataset.py        # Dataset verification helper
│   └── test_model.py          # Model architecture verification
│
├── static/                    # Frontend static assets
│   ├── css/
│   │   └── style.css          # Modern UI stylesheet
│   └── js/
│       └── app.js             # Client-side interaction & API caller
│
├── templates/
│   └── index.html             # Main dashboard template
│
└── utils/                     # Modular backend utilities
    ├── config.py              # Central configuration & hyperparameters
    ├── dataset.py             # PyTorch Dataset implementation
    ├── model.py               # TrafficSignCNN neural network definition
    ├── sign_names.py          # Mapping of 43 GTSRB class IDs & descriptions
    └── transforms.py          # Albumentations data augmentation pipelines
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/CoresCode/TrafficSignRecognitionUsingCNN.git
cd TrafficSignRecognitionUsingCNN
```

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Web Application

Launch the Flask server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 💻 CLI & Additional Scripts

### Single Image Prediction CLI
You can test any image directly from the terminal:
```bash
python scripts/06_predict_image.py --image ppt_samples/Stop.png
```

### Evaluate Model on Test Set
To compute full classification metrics and regenerate the confusion matrix:
```bash
python scripts/test.py
```

### Train the Model from Scratch
To train the CNN model using the GTSRB dataset:
```bash
python train.py
```

---

## 🚦 Supported Classes (GTSRB 43 Categories)

The model recognizes all 43 classes of the German Traffic Sign Recognition Benchmark, including:
- **Speed Limits:** 20, 30, 50, 60, 70, 80, 100, 120 km/h
- **Prohibitory:** No passing, No passing for vehicles > 3.5t, No entry, General danger
- **Mandatory:** Turn right, Turn left, Ahead only, Roundabout mandatory, Keep right/left
- **Danger Warnings:** Pedestrians, Children crossing, Bicycles, Road work, Traffic signals, Slippery road
- **Priority:** Right-of-way, Priority road, Yield, Stop

---

## 📄 License

This project is released into the **public domain** under **The Unlicense**.
You are free to use, modify, distribute, and commercialize this software without restriction.
