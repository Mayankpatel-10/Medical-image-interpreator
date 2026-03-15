# 🏥 Medical Image Classification System

Ek advanced deep learning system jo medical images se brain tumor aur pneumonia detect karta hai using EfficientNet-B0 architecture.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Prediction](#prediction)
- [File Structure](#file-structure)
- [Performance](#performance)
- [Contributing](#contributing)

## ✨ Features

### 🧠 **Brain Tumor Detection**
- **4 Classes**: Glioma, Meningioma, No Tumor, Pituitary
- **High Accuracy**: EfficientNet-B0 based model
- **Grad-CAM Visualization**: Model attention heatmap

### 🫁 **Pneumonia Detection**
- **2 Classes**: Normal, Pneumonia
- **Fast Inference**: Optimized for real-time use
- **Confidence Scores**: Detailed probability analysis

### 🚀 **Advanced Features**
- **GPU Acceleration**: CUDA support with mixed precision
- **Transfer Learning**: Pre-trained EfficientNet-B0
- **Data Augmentation**: Robust training pipeline
- **Automatic Batch Size**: GPU memory based optimization
- **Progress Tracking**: Real-time training metrics
- **Model Saving**: Best models automatically saved

## 📦 Requirements

### Hardware Requirements
- **Minimum**: 4GB RAM, CPU
- **Recommended**: 8GB+ GPU RAM, NVIDIA GPU with CUDA

### Software Requirements
```bash
# Python 3.8+
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
opencv-python>=4.5.0
Pillow>=8.3.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

## 🛠️ Installation

### 1. Repository Clone Karein
```bash
git clone <repository-url>
cd EL-Project
```

### 2. Python Environment Setup
```bash
# Virtual environment banayein
python -m venv medical_env

# Activate karein
# Windows:
medical_env\Scripts\activate
# Linux/Mac:
source medical_env/bin/activate

# Dependencies install karein
pip install -r requirements.txt
```

### 3. GPU Setup (Optional)
```bash
# NVIDIA GPU drivers install karein
# CUDA aur cuDNN setup karein
# PyTorch GPU version check karein
python -c "import torch; print(torch.cuda.is_available())"
```

## 📁 Dataset Structure

```
Data/
├── brain_tumor/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
└── pnemonia/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

## 🚀 Usage

### 1. **Training Models**
```bash
# Brain tumor aur pneumonia dono models train karein
python train.py

# Output:
# - best_brain_tumor_model.pth
# - best_pnemonia_model.pth
```

### 2. **Making Predictions**
```bash
# Interactive prediction tool
python predict.py

# File browser se image select karein
# Automatic disease detection
# Grad-CAM visualization
# Detailed analysis report
```

### 3. **Backend API (Optional)**
```bash
# Flask server start karein
cd backend
python app.py

# API Endpoints:
# POST /api/predict - Single image prediction
# GET /api/health - System status
```

## 🏗️ Model Architecture

### **EfficientNet-B0 Architecture**
```
Input (224x224x3)
    ↓
Stem Convolution + BatchNorm + Swish
    ↓
MBConv Blocks (7 stages)
    ↓
Head Convolution + BatchNorm + Swish
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Dense Layer (Custom Classes)
    ↓
Output (4 for Brain Tumor, 2 for Pneumonia)
```

### **Transfer Learning**
- **Base Model**: ImageNet pre-trained EfficientNet-B0
- **Fine-tuning**: Custom classification head
- **Optimization**: AdamW optimizer with weight decay

## 🎯 Training

### **Training Pipeline**
```python
# 1. Data Loading
train_loader, val_loader, test_loader = create_data_loaders()

# 2. Model Creation
model = EfficientNet-B0(pretrained=True)
model.classifier[1] = Linear(num_features, num_classes)

# 3. Training Loop
for epoch in range(epochs):
    # Training phase
    model.train()
    for batch in train_loader:
        # Mixed precision training
        # Forward pass
        # Loss calculation
        # Backward pass
        # Optimizer step
    
    # Validation phase
    model.eval()
    # Calculate validation metrics
    
    # Learning rate scheduling
    scheduler.step(val_accuracy)
```

### **Training Parameters**
- **Epochs**: 20
- **Batch Size**: Auto-adjusted (16-64 based on GPU memory)
- **Learning Rate**: 0.001 with AdamW optimizer
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau

### **Data Augmentation**
- Random Rotation (±20°)
- Random Horizontal Flip
- Random Affine Transform
- Color Jittering
- Normalization

## 🔮 Prediction

### **Prediction Pipeline**
```python
# 1. Image Preprocessing
image = load_image(path)
image = resize(224x224)
image = normalize(ImageNet stats)
image = tensor + batch_dim

# 2. Model Inference
model.eval()
with torch.no_grad():
    output = model(image)
    probabilities = softmax(output)

# 3. Post Processing
prediction = argmax(probabilities)
confidence = max(probabilities)

# 4. Visualization
grad_cam = generate_grad_cam(model, image, target_layer)
heatmap = apply_colormap(grad_cam)
superimposed = blend(image, heatmap)
```

### **Output Format**
```
Medical Image Analysis Report
├── Original Image
├── Primary Detection
│   ├── Disease: Brain Tumor
│   ├── Prediction: Meningioma
│   └── Confidence: 94.2%
├── Attention Map (Grad-CAM)
└── Detailed Analysis
    ├── Glioma: 2.1%
    ├── Meningioma: 94.2%
    ├── No Tumor: 1.8%
    └── Pituitary: 1.9%
```

## 📂 File Structure

```
EL-Project/
├── train.py              # Training script
├── predict.py            # Prediction script
├── README.md             # Documentation
├── requirements.txt      # Dependencies
├── Data/               # Dataset folder
├── backend/            # Flask API (optional)
│   ├── app.py         # Main Flask app
│   ├── config.py      # Configuration
│   ├── database.py    # Database operations
│   └── models.py      # Database models
├── best_brain_tumor_model.pth    # Trained brain tumor model
├── best_pnemonia_model.pth      # Trained pneumonia model
└── prediction_*.png             # Generated predictions
```

## 📊 Performance

### **Brain Tumor Detection**
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~95%
- **F1-Score**: ~94%

### **Pneumonia Detection**
- **Accuracy**: ~96%
- **Precision**: ~95%
- **Recall**: ~96%
- **F1-Score**: ~95%

### **Inference Speed**
- **CPU**: ~500ms per image
- **GPU**: ~50ms per image
- **Memory Usage**: ~2GB GPU memory

## 🔧 Configuration

### **GPU Memory Optimization**
```python
# Automatic batch size adjustment
if gpu_memory >= 8GB:
    batch_size = 64
elif gpu_memory >= 4GB:
    batch_size = 32
else:
    batch_size = 16
```

### **Mixed Precision Training**
```python
# GPU memory bachane ke liye
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 🐛 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```python
   # Batch size kam karein
   classifier = GPUMedicalClassifier(batch_size=16)
   ```

2. **Model Not Found**
   ```bash
   # Pehle train karein
   python train.py
   ```

3. **Dataset Path Error**
   ```python
   # Data folder check karein
   # Correct structure follow karein
   ```

4. **Import Errors**
   ```bash
   # Dependencies install karein
   pip install -r requirements.txt
   ```

## 🤝 Contributing

1. Fork karein repository
2. Feature branch banayein (`git checkout -b feature/AmazingFeature`)
3. Commit karein changes (`git commit -m 'Add some AmazingFeature'`)
4. Push karein branch (`git push origin feature/AmazingFeature`)
5. Open karein Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍⚕️ Medical Disclaimer

⚠️ **Important**: Ye system educational aur research purposes ke liye hai. Medical diagnosis ke liye qualified healthcare professional se consult karein.

- Ye tool medical decisions ke liye nahi hai
- Always doctor se verified diagnosis lein
- Emergency situations mein immediate medical help lein

## 📞 Contact

- **Project Maintainer**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [github.com/yourusername]

---

⭐ **Star** the repository if this project helped you!

🚀 **Happy Coding!** 🏥
