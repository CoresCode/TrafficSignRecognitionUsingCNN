# 🚦 Traffic Sign Detection using Deep Learning

A web-based Traffic Sign Detection application developed using **PyTorch**. The system utilizes a custom Convolutional Neural Network (CNN) trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset to classify traffic signs into 43 different categories.

## Features

- 🚦 Traffic Sign Detection
- 🧠 Custom CNN Architecture
- 📤 Image Upload Interface
- 📊 Confidence Score & Top-5 Predictions
- 📈 Model Performance Dashboard
- 📉 Training & Validation Curves
- 📋 Confusion Matrix Visualization
- 🌙 Modern Streamlit User Interface

## Dataset

- **German Traffic Sign Recognition Benchmark (GTSRB)**
- 43 Traffic Sign Classes

## Model Performance

| Metric | Score |
|--------|------:|
| Accuracy | **98.92%** |
| Precision | **98.95%** |
| Recall | **98.92%** |
| F1 Score | **98.91%** |

## Technologies Used

- Python
- PyTorch
- OpenCV
- Albumentations
- Streamlit
- NumPy
- Matplotlib
- Scikit-learn


## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/traffic-sign-detection.git
cd traffic-sign-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Results

The trained CNN achieves **98.92% test accuracy** on the GTSRB dataset while providing real-time traffic sign detection through an interactive web interface.

## License

This project is released into the **public domain** under **The Unlicense**.

You are free to use, modify, distribute, and commercialize this project without restriction or attribution.

---

Developed as a Deep Learning project using **PyTorch**
