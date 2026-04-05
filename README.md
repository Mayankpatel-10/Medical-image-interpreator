# Medical Image Interpreter

A medical image analysis system that can detect brain tumors and pneumonia from medical images using deep learning models.

## What It Does

This application can:
- Detect brain tumors (glioma, meningioma, no tumor, pituitary)
- Identify pneumonia in chest X-rays (normal, pneumonia)
- Automatically classify which disease type your image contains
- Show detailed analysis with visual explanations

## Key Features

**Smart Disease Classification**
The system first determines what type of medical image you uploaded, then uses the correct AI model for analysis. If you select "Brain Tumor" but upload a chest X-ray, it will detect this and warn you.

**Visual Analysis**
Each prediction includes a 4-panel report showing:
- Your original medical image
- Main prediction result with confidence score
- Heat map showing what the AI focused on
- Detailed probability breakdown

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python predict.py`
3. Select disease type (Brain Tumor or Pneumonia)
4. Choose your medical image file
5. View the analysis report

## How It Works

The system uses two separate AI models:
- Brain Tumor Model: EfficientNet-B0 trained on brain MRI scans
- Pneumonia Model: EfficientNet-B0 trained on chest X-rays

When you analyze an image, the system runs both models to determine what type of image you actually uploaded, then uses the appropriate model for detailed analysis.

## Example Usage

```
Select option (1-3): 1
Selected: Brain Tumor Prediction
Analyzing: brain_scan.jpg

Image Classification: Brain Tumor (Confidence: 98.5%)
User Selection: Brain Tumor
Report saved as: prediction_brain_scan.png
```

If there's a mismatch:
```
Image Classification: Pneumonia (Confidence: 92.3%)
User Selection: Brain Tumor

WARNING: Image appears to be Pneumonia, but you selected Brain Tumor
The system will analyze with the appropriate model for the actual disease type.
```

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- PIL (Pillow)
- Matplotlib
- NumPy

## Files Needed

- `best_brain_tumor_model.pth` - Brain tumor detection model
- `best_pnemonia_model.pth` - Pneumonia detection model

## Performance

- Brain Tumor Detection: ~95% accuracy
- Pneumonia Detection: ~96% accuracy

## Important Note

This tool is for educational purposes only and should not replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Recent Updates

**Version 2.0**: Added smart disease classification that automatically detects image type and prevents incorrect model usage.
