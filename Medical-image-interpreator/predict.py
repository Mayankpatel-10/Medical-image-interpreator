# Importing required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Detect if running in Google Colab
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# Import relevant GUI tool based on environment
if IS_COLAB:
    from google.colab import files
else:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("Warning: Tkinter not found. File browser will not be available locally.")

# GPU Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EasyMedicalPredictor:
    """Main class for medical image prediction"""
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.device = device
        self.is_colab = IS_COLAB
        
        # Image preprocessing - Identical to original
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Identical class lists
        self.brain_tumor_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        self.pneumonia_classes = ['NORMAL', 'PNEUMONIA']
        
        self.brain_tumor_model = None
        self.pneumonia_model = None
        
        print("Easy Medical Image Predictor")
        print("="*40)
        print(f"Environment: {'Google Colab' if self.is_colab else 'Local Machine'}")
        print("="*40)
    
    def load_models(self):
        """Load trained models from the local directory"""
        models_loaded = 0
        
        if os.path.exists('best_brain_tumor_model.pth'):
            try:
                self.brain_tumor_model = models.efficientnet_b0(pretrained=False)
                num_ftrs = self.brain_tumor_model.classifier[1].in_features
                self.brain_tumor_model.classifier[1] = nn.Linear(num_ftrs, 4)
                self.brain_tumor_model.load_state_dict(torch.load('best_brain_tumor_model.pth', map_location=self.device))
                self.brain_tumor_model = self.brain_tumor_model.to(self.device).eval()
                print("✓ Brain tumor model loaded")
                models_loaded += 1
            except Exception as e: print(f"✗ Error loading brain tumor model: {e}")
        
        if os.path.exists('best_pnemonia_model.pth'):
            try:
                self.pneumonia_model = models.efficientnet_b0(pretrained=False)
                num_ftrs = self.pneumonia_model.classifier[1].in_features
                self.pneumonia_model.classifier[1] = nn.Linear(num_ftrs, 2)
                self.pneumonia_model.load_state_dict(torch.load('best_pnemonia_model.pth', map_location=self.device))
                self.pneumonia_model = self.pneumonia_model.to(self.device).eval()
                print("✓ Pneumonia model loaded")
                models_loaded += 1
            except Exception as e: print(f"✗ Error loading pneumonia model: {e}")
        
        return models_loaded
    
    def select_image_file(self):
        """Select image file based on environment"""
        if self.is_colab:
            print("\nSelect image to upload:")
            uploaded = files.upload()
            return list(uploaded.keys()) if uploaded else None
        else:
            try:
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)
                file_path = filedialog.askopenfilename(title="Select medical image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
                root.destroy()
                return [file_path] if file_path else None
            except:
                path = input("Enter path to image: ").strip()
                return [path] if path else None

    def preprocess_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            return image_tensor, original_image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None, None
    
    def predict_brain_tumor(self, image_tensor):
        if not self.brain_tumor_model: return None, None, None
        with torch.no_grad():
            outputs = self.brain_tumor_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
            confidence, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            return self.brain_tumor_classes[predicted.item()], confidence.item(), probabilities
    
    def predict_pneumonia(self, image_tensor):
        if not self.pneumonia_model: return None, None, None
        with torch.no_grad():
            outputs = self.pneumonia_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy().flatten()
            confidence, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            return self.pneumonia_classes[predicted.item()], confidence.item(), probabilities
    
    def create_grad_cam(self, model, image_tensor):
        """Standard PyTorch Grad-CAM implementation (Fixes the mixed TF/Torch bug)"""
        try:
            gradients = []
            activations = []
            def save_grad(m, gi, go): gradients.append(go[0])
            def save_act(m, i, o): activations.append(o)
            
            # Find last conv layer
            target_layer = None
            for module in model.modules():
                if isinstance(module, nn.Conv2d): target_layer = module
            if not target_layer: return None
            
            h1 = target_layer.register_forward_hook(save_act)
            h2 = target_layer.register_full_backward_hook(save_grad)
            
            model.zero_grad()
            output = model(image_tensor)
            pred_idx = output.argmax().item()
            output[0, pred_idx].backward()
            
            h1.remove(); h2.remove()
            
            weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
            cam = torch.relu(torch.sum(weights * activations[0], dim=1)).squeeze().detach().cpu().numpy()
            if cam.max() > cam.min(): cam = (cam - cam.min()) / (cam.max() - cam.min())
            return cam
        except Exception as e:
            print(f"Visualization error: {e}")
            return None

    def predict_and_visualize(self, image_path):
        """Main analysis function - preserved functionality and visuals"""
        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        image_tensor, original_image = self.preprocess_image(image_path)
        if image_tensor is None: return
        
        # Get predictions from both models
        b_class, b_conf, b_probs = self.predict_brain_tumor(image_tensor)
        p_class, p_conf, p_probs = self.predict_pneumonia(image_tensor)
        
        results = []
        if b_class: results.append({'disease': 'Brain Tumor', 'prediction': b_class, 'confidence': b_conf, 'model': self.brain_tumor_model, 'details': b_probs, 'classes': self.brain_tumor_classes})
        if p_class: results.append({'disease': 'Pneumonia', 'prediction': p_class, 'confidence': p_conf, 'model': self.pneumonia_model, 'details': p_probs, 'classes': self.pneumonia_classes})
        
        if not results:
            print("No models available for prediction")
            return

        # Selection of best result
        best = max(results, key=lambda x: x['confidence'])
        
        # Identical 4-panel split visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Medical Image Analysis Report', fontsize=18, fontweight='bold')
        
        # 1. Input Image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Main Prediction Text
        summary = f"Detection Result\n\n{best['disease']}\n\n{best['prediction'].upper()}\n\nConfidence: {best['confidence']:.1%}"
        bg_col = 'lightgreen' if best['confidence'] > 0.8 else 'lightyellow' if best['confidence'] > 0.6 else 'lightcoral'
        axes[0, 1].text(0.5, 0.5, summary, ha='center', va='center', fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_col))
        axes[0, 1].set_title('Primary Detection', fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Grad-CAM Visualization
        cam = self.create_grad_cam(best['model'], image_tensor)
        if cam is not None:
            cam_map = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (224, 224))), cv2.COLORMAP_JET)
            cam_map = cv2.cvtColor(cam_map, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(np.array(original_image.resize((224, 224))), 0.6, cam_map, 0.4, 0)
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Attention Map (Grad-CAM)', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Grad-CAM Generation Failed', ha='center', va='center')
        axes[1, 0].axis('off')
        
        # 4. Detailed Analysis
        detail_txt = f"Detailed Predictions ({best['disease']}):\n\n"
        for label, prob in zip(best['classes'], best['details']):
            detail_txt += f"• {label}: {prob:.1%}\n"
        axes[1, 1].text(0.1, 0.5, detail_txt, ha='left', va='center', fontsize=11, family='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Detailed Analysis', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save output image - Added back for identical behavior
        output_name = f"prediction_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Report saved as: {output_name}")
        
        plt.show()

def main():
    predictor = EasyMedicalPredictor()
    if predictor.load_models() == 0:
        print("\nFATAL: No trained models found in the current directory.")
        return

    while True:
        print("\n" + "="*50)
        print("1. Select Image (Analyze)")
        print("2. Exit")
        choice = input("Select option (1-2): ").strip()
        
        if choice == '2': break
        if choice == '1':
            file_paths = predictor.select_image_file()
            if file_paths:
                for path in file_paths:
                    try:
                        predictor.predict_and_visualize(path)
                    except Exception as e:
                        print(f"Error analyzing {path}: {e}")

if __name__ == "__main__":
    main()
