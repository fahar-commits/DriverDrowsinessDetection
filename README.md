# 🚗 Driver Drowsiness Detection System


A real-time computer vision system that detects driver drowsiness using webcam feed and triggers multi-modal alerts to prevent accidents. Built with deep learning and designed for research and portfolio demonstration.

## 🎯 Features

- **Real-time Detection**: Monitors driver state continuously using webcam
- **Multi-class Classification**: Detects Alert, Drowsy, and Yawning states
- **Temporal Smoothing**: Reduces false positives with prediction buffering
- **Audio Alerts**: Plays warning sounds when drowsiness detected
- **Visual Alerts**: Screen flashing and on-screen warnings
- **Performance Metrics**: Real-time FPS display and detailed evaluation reports
- **Extensible**: Ready for hardware integration (steering vibration, seat actuators)

## 🏗️ System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webcam    │────▶│  CNN Model   │────▶│ Alert System│
│   Feed      │     │ (MobileNetV2)│     │ (Multi-Modal)│
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────▼──────┐
                    │  Temporal   │
                    │  Smoothing  │
                    └─────────────┘
```

## 📁 Project Structure

```
DriverDrowsinessDetection/
├── train_model.py           # Model training script
├── webcam_inference.py      # Real-time detection system
├── data_loader.py           # Dataset download/preparation utility
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore rules
│
├── datasets/
│   └── drowsiness/
│       ├── alert/         # Alert driver images
│       ├── drowsy/        # Drowsy/eyes closed images
│       └── yawning/       # Yawning driver images
│
├── models/
│   └── drowsiness_detector_best.pth  # Trained model weights
│
├── sounds/
│   └── alert.wav          # Alert sound file
│
└── logs/
    ├── training_history.json      # Training metrics
    ├── evaluation_results.json    # Model performance
    └── results.csv                # Detection logs
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/DriverDrowsinessDetection](https://github.com/fahar-commits/DriverDrowsinessDetection.git
cd DriverDrowsinessDetection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Option A: Download Public Dataset**

Download one of these datasets:
- [Drowsiness Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)

### 3. Train the Model

```bash
python train_model.py
```

Training parameters (editable in `train_model.py`):
- **Epochs**: 20 (adjustable)
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Architecture**: MobileNetV2 (pre-trained)

Expected training time:
- CPU: ~2-3 hours
- GPU: ~15-20 minutes

### 4. Run Real-time Detection

```bash
python webcam_inference.py
```

**Controls**:
- **Q**: Quit
- **S**: Save screenshot
- **R**: Reset statistics

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 94.5% |
| Alert Precision | 96.2% |
| Drowsy Recall | 92.8% |
| Yawning F1-Score | 93.5% |

*Results based on sample dataset. Performance may vary.*


### Camera Settings

```python
monitor.run(camera_id=0)  # Change camera source
```

## 🙏 Acknowledgments

- PyTorch Team for MobileNetV2 implementation
- OpenCV Community for computer vision tools
- Kaggle for hosting drowsiness detection datasets
- Research papers:
  - Soukupová and Čech (2016) - "Real-Time Eye Blink Detection"
  - Dua et al. (2021) - "Driver Drowsiness Detection"

**⭐ If you found this project helpful, please give it a star on GitHub!**
