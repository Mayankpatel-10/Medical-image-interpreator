# Medical Image Interpretation System

A high-performance medical image classification framework built with PyTorch and EfficientNet-B0. This system is designed to detect brain tumors and pneumonia from medical imaging (MRI and X-Ray) with high confidence and transparency.

## Project Overview

This project provides a complete pipeline from training to prediction. It uses transfer learning with EfficientNet-B0 to achieve high accuracy while remaining lightweight enough to run on consumer hardware or Google Colab T4 runtimes.

### Key Features
- Multi-disease detection (Brain Tumor and Pneumonia).
- Automatic environment detection (Local Machine or Google Colab).
- Native browser file uploads in Colab sessions.
- Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations to show AI focus areas.
- Automated training and best-model checkpointing.

## Project Structure

- **Medical-image-interpreator/**: Contains the core source code, including training and prediction scripts.
- **Data/**: Directory containing the organized dataset for training and testing.
- **train.py**: Script to train the models on a T4 GPU.
- **predict.py**: Hybrid script for interactive analysis and report generation.

## Installation

To set up the environment, ensure you have Python 3.8+ installed, then run:

```bash
pip install -r requirements.txt
```

### Dependencies
- torch >= 2.5.0
- torchvision >= 0.20.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- opencv-python >= 4.8.0
- scikit-learn >= 1.3.0
- pillow >= 10.0.0
- seaborn >= 0.12.0
- tqdm >= 4.66.0

## Usage Instructions

### 1. Training (Optional)
If you have the dataset in the `Data` folder, you can train the models by running:

```bash
python train.py
```

The script will automatically detect an available GPU and save the best models as `best_brain_tumor_model.pth` and `best_pnemonia_model.pth`.

### 2. Prediction and Analysis
To analyze a new medical image, run:

```bash
python predict.py
```

- **Local Use**: A standard file dialog will open to select your image.
- **Colab Use**: A browser-native file uploader will appear.
- **Output**: The system generates a 4-panel report containing the input image, the primary detection result, a Grad-CAM heatmap, and detailed probability breakdowns. Every report is saved as a `.png` file.

## Visualization: Grad-CAM
The system includes a Grad-CAM implementation that highlights the specific regions in the medical image that influenced the AI's decision. This feature is critical for medical interpretability, allowing users to verify if the model is focusing on relevant pathological features (e.g., a specific lung region for pneumonia).

## Medical Disclaimer
Important: This system is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider for any questions regarding a medical condition.
