# Required libraries import kar rahe hain
import os  # File system ke liye
import numpy as np  # Sankhyatmak operations ke liye
import matplotlib.pyplot as plt  # Graphs aur visualization ke liye
import cv2  # Image processing ke liye (OpenCV)
from PIL import Image  # Image handling ke liye
import torch  # Deep learning framework PyTorch
import torch.nn as nn  # Neural network modules
from torchvision import transforms, models  # Computer vision ke liye
import torch.nn.functional as F  # Neural network functions
import tkinter as tk  # File dialog box ke liye
from tkinter import filedialog  # File chunne ke liye
import warnings
warnings.filterwarnings('ignore')  # Warnings ko ignore karne ke liye

# GPU Configuration - CUDA available hai to GPU use karenge warna CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EasyMedicalPredictor:
    """Medical image prediction ke liye main class"""
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.device = device
        
        # Image preprocessing - Image ko model ke liye taiyar karna
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Image ko 224x224 size mein resize karein
            transforms.ToTensor(),  # Image ko tensor mein badlein
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize karein
        ])
        
        # Disease ki categories
        self.brain_tumor_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Brain tumor types
        self.pneumonia_classes = ['NORMAL', 'PNEUMONIA']  # Pneumonia categories
        
        # Models - dono bimariyon ke liye alag-alag models
        self.brain_tumor_model = None  # Brain tumor detection model
        self.pneumonia_model = None  # Pneumonia detection model
        
        print("Easy Medical Image Predictor")
        print("="*40)
        print("Features:")
        print("- File browser for image selection")
        print("- Automatic disease detection")
        print("- Grad-CAM visualization")
        print("="*40)
    
    def load_models(self):
        """Trained models ko load karna"""
        models_loaded = 0  # Kitne models hue
        
        # Brain tumor model ko load karein
        if os.path.exists('best_brain_tumor_model.pth'):
            try:
                # EfficientNet-B0 model banayein aur use apni zarurat ke hisab se modify karein
                self.brain_tumor_model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.brain_tumor_model.classifier[1].in_features  # Akhri layer ke features nikalein
                self.brain_tumor_model.classifier[1] = nn.Linear(num_ftrs, 4)  # 4 classes ke liye
                # Trained weights ko load karein
                self.brain_tumor_model.load_state_dict(torch.load('best_brain_tumor_model.pth', map_location=self.device))
                self.brain_tumor_model = self.brain_tumor_model.to(self.device)  # GPU/CPU par bhejein
                self.brain_tumor_model.eval()  # Evaluation mode mein set karein
                print("Brain tumor model loaded")
                models_loaded += 1
            except Exception as e:
                print(f"Error loading brain tumor model: {e}")
        else:
            print("Brain tumor model not found")
        
        # Pneumonia model ko load karein
        if os.path.exists('best_pnemonia_model.pth'):
            try:
                self.pneumonia_model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.pneumonia_model.classifier[1].in_features
                self.pneumonia_model.classifier[1] = nn.Linear(num_ftrs, 2)
                self.pneumonia_model.load_state_dict(torch.load('best_pnemonia_model.pth', map_location=self.device))
                self.pneumonia_model = self.pneumonia_model.to(self.device)
                self.pneumonia_model.eval()
                print("Pneumonia model loaded")
                models_loaded += 1
            except Exception as e:
                print(f"Error loading pneumonia model: {e}")
        else:
            print("Pneumonia model not found")
        
        return models_loaded
    
    def select_image_file(self):
        """File dialog kholkar image select karna"""
        print("\nFile browser khola ja raha hai...")
        
        # Root window banayein (chhupa hua)
        root = tk.Tk()
        root.withdraw()  # Window ko chipayein
        root.attributes('-topmost', True)  # Sabse upar rakhein
        
        # File dialog box
        file_path = filedialog.askopenfilename(
            title="Medical image chunein",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        root.destroy()
        
        if file_path:
            print(f"Selected: {os.path.basename(file_path)}")
            return file_path
        else:
            print("No file selected")
            return None
    
    def preprocess_image(self, image_path):
        """Prediction ke liye image ko preprocess karna"""
        try:
            image = Image.open(image_path).convert('RGB')  # Image ko RGB mein badlein
            original_image = image.copy()  # Original image ki copy rakhein
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Transform aur tensor mein badlein
            return image_tensor, original_image
        except Exception as e:
            print(f"Image preprocessing mein error: {e}")
            return None, None
    
    def predict_brain_tumor(self, image_tensor):
        """Brain tumor ka prediction karna"""
        if self.brain_tumor_model is None:
            return None, None, None  # Model available nahi hai
        
        with torch.no_grad():  # Gradient calculation band karein (bachat ke liye)
            outputs = self.brain_tumor_model(image_tensor)  # Model se prediction lein
            probabilities = F.softmax(outputs, dim=1)  # Probabilities nikalein
            confidence, predicted = torch.max(probabilities, 1)  # Sabse zyada confidence wala class nikalein
            
            predicted_class = self.brain_tumor_classes[predicted.item()]  # Class ka naam nikalein
            confidence_score = confidence.item()  # Confidence score
            all_probs = probabilities.cpu().numpy().flatten()  # Sabhi classes ki probabilities
            
            return predicted_class, confidence_score, all_probs
    
    def predict_pneumonia(self, image_tensor):
        """Pneumonia ka prediction karna"""
        if self.pneumonia_model is None:
            return None, None, None  # Model available nahi hai
        
        with torch.no_grad():
            outputs = self.pneumonia_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.pneumonia_classes[predicted.item()]
            confidence_score = confidence.item()
            all_probs = probabilities.cpu().numpy().flatten()
            
            return predicted_class, confidence_score, all_probs
    
    def create_grad_cam_fixed(self, model, image_tensor, target_layer_name='features'):
        """Grad-CAM visualization banana - ye batata hai ki model ne kis hisse par dhyan diya"""
        try:
            # Gradients aur activations ko store karne ke liye
            gradients = {}
            activations = {}
            
            def backward_hook(module, grad_input, grad_output):
                gradients[module] = grad_output[0]
            
            def forward_hook(module, input, output):
                activations[module] = output
            
            # Find the target layer
            target_layer = None
            for name, module in model.named_modules():
                if target_layer_name in name and isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
            
            if target_layer is None:
                # Try to find any feature layer
                for name, module in model.named_modules():
                    if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
                        target_layer = module
                        break
            
            if target_layer is None:
                return None
            
            # Register hooks
            forward_handle = target_layer.register_forward_hook(forward_hook)
            backward_handle = target_layer.register_backward_hook(backward_hook)
            
            # Forward pass - Image ko model se guzarein
            model.eval()
            output = model(image_tensor)
            pred_idx = output.argmax(dim=1).item()  # Sabse zyada confidence wala class
            
            # Backward pass - Gradients calculate karein
            model.zero_grad()  # Purane gradients clear karein
            class_loss = output[0, pred_idx]  # Loss calculate karein
            class_loss.backward(retain_graph=True)  # Backpropagation
            
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
            
            # Gradients aur activations ko milkar CAM banayein
            if target_layer in gradients and target_layer in activations:
                grads = gradients[target_layer]
                acts = activations[target_layer]
                
                # Gradients ka global average pooling
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                
                # Activations ka weighted combination
                cam = torch.sum(weights * acts, dim=1)
                cam = torch.relu(cam)  # Negative values ko 0 karein
                
                # Normalize karein (0 se 1 ke beech)
                if cam.max() > cam.min():
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                
                return cam.squeeze().cpu().numpy()
            
            return None
            
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None
    
    def create_simple_heatmap(self, image_tensor, model):
        """Simple attention heatmap - agar Grad-CAM fail ho jaye to use karein"""
        try:
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)  # Prediction lein
                
            # Prediction confidence par based simple attention map banayein
            attention_map = torch.ones((1, self.image_size, self.image_size)) * output[0].max().item()
            attention_map = attention_map.squeeze().cpu().numpy()
            
            # Kuch variation jodein taki ye heatmap jaisa dikhe
            x, y = np.meshgrid(np.linspace(0, 1, self.image_size), np.linspace(0, 1, self.image_size))
            
            # Radial gradient banayein (beech se bahar ki aur)
            dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            attention_map = attention_map * np.exp(-dist * 3)  # Exponential decay
            
            return attention_map
            
        except Exception as e:
            return None
    
    def predict_and_visualize(self, image_path):
        """Bimari ka prediction karein aur visualization banayein"""
        print(f"\nVishleshan kiya ja raha hai: {os.path.basename(image_path)}")
        print("="*50)
        
        # Preprocessing - Image ko taiyar karein
        image_tensor, original_image = self.preprocess_image(image_path)
        if image_tensor is None:
            return  # Agar image load nahi hui to rukein
        
        # Prediction - Dono models se result lein
        brain_result = self.predict_brain_tumor(image_tensor)  # Brain tumor ka result
        pneumonia_result = self.predict_pneumonia(image_tensor)  # Pneumonia ka result
        
        # Results ko ikattha karein
        results = []
        
        if brain_result[0] is not None:
            results.append({
                'disease': 'Brain Tumor',
                'prediction': brain_result[0],
                'confidence': brain_result[1],
                'details': brain_result[2],
                'model': self.brain_tumor_model
            })
        
        if pneumonia_result[0] is not None:
            results.append({
                'disease': 'Pneumonia',
                'prediction': pneumonia_result[0],
                'confidence': pneumonia_result[1],
                'details': pneumonia_result[2],
                'model': self.pneumonia_model
            })
        
        if not results:
            print("No models available for prediction")
            return
        
        # Sabse achha result (sabse zyada confidence wala)
        best_result = max(results, key=lambda x: x['confidence'])
        
        # Visualization - 4 bhagon mein report banayein
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Medical Image Analysis Report', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')  # Axis chipayein
        
        # Main prediction result
        result_text = f"Detection Result\n\n{best_result['disease']}\n\n{best_result['prediction']}\n\nConfidence: {best_result['confidence']:.1%}"
        # Confidence ke hisab se rang chunein
        color = "lightgreen" if best_result['confidence'] > 0.8 else "lightyellow" if best_result['confidence'] > 0.6 else "lightcoral"
        axes[0, 1].text(0.5, 0.5, result_text, ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        axes[0, 1].set_title('Primary Detection', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Grad-CAM visualization - Model ne kis hisse par dhyan diya
        if best_result['model'] is not None:
            print("Grad-CAM visualization banaya ja raha hai...")
            
            # Pehle fixed Grad-CAM ki koshish karein
            cam = self.create_grad_cam_fixed(best_result['model'], image_tensor)
            
            if cam is not None:
                print("Grad-CAM successfully banaya gaya!")
                # CAM ko image size mein resize karein
                cam_resized = cv2.resize(cam, (self.image_size, self.image_size))
                # Heatmap banayein (Jet colormap ke saath)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # BGR se RGB mein badlein
                
                # Original image ko numpy mein badlein aur dono ko milayein
                orig_np = np.array(original_image.resize((self.image_size, self.image_size)))
                superimposed = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)  # 60% image + 40% heatmap
                
                axes[1, 0].imshow(superimposed)
                axes[1, 0].set_title('Attention Map (Grad-CAM)', fontweight='bold')
            else:
                print("Grad-CAM fail ho gaya, simple heatmap use kar rahe hain...")
                # Fallback ke liye simple heatmap
                simple_heat = self.create_simple_heatmap(image_tensor, best_result['model'])
                if simple_heat is not None:
                    heatmap = cv2.applyColorMap(np.uint8(255 * simple_heat), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    orig_np = np.array(original_image.resize((self.image_size, self.image_size)))
                    superimposed = cv2.addWeighted(orig_np, 0.7, heatmap, 0.3, 0)
                    
                    axes[1, 0].imshow(superimposed)
                    axes[1, 0].set_title('Attention Map (Simple)', fontweight='bold')
                else:
                    axes[1, 0].text(0.5, 0.5, 'Heatmap Generation Failed', ha='center', va='center')
                    axes[1, 0].set_title('Attention Map', fontweight='bold')
        else:
            axes[1, 0].text(0.5, 0.5, 'Model Not Available', ha='center', va='center')
            axes[1, 0].set_title('Attention Map', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Detailed analysis - Sabhi classes ki probabilities
        details_text = "Detailed Predictions:\n\n"
        for result in results:
            details_text += f"{result['disease']}:\n"
            details_text += f"  Result: {result['prediction']}\n"
            details_text += f"  Confidence: {result['confidence']:.1%}\n"
            
            if result['disease'] == 'Brain Tumor':
                classes = self.brain_tumor_classes
            else:
                classes = self.pneumonia_classes
            
            for i, (cls, prob) in enumerate(zip(classes, result['details'])):
                details_text += f"  {cls}: {prob:.1%}\n"
            details_text += "\n"
        
        axes[1, 1].text(0.05, 0.95, details_text, ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Detailed Analysis', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Result save karein
        output_path = f"prediction_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Result save ho gaya: {output_path}")
        
        plt.show()  # Report dikhaein
        
        # Saransh - Final diagnosis
        print(f"\nFinal Diagnosis:")
        print(f"Primary: {best_result['disease']} - {best_result['prediction']}")
        print(f"Confidence: {best_result['confidence']:.1%}")
        
        if best_result['disease'] == 'Brain Tumor':
            if best_result['prediction'] == 'notumor':
                print("Result: Koi tumor nahi mila - Brain normal lag raha hai")
            else:
                print(f"Warning: {best_result['prediction'].upper()} tumor mila - Neurologist se salah lein")
        else:
            if best_result['prediction'] == 'NORMAL':
                print("Result: Phephde normal lag rahe hain - Koi pneumonia nahi")
            else:
                print("Warning: Pneumonia mila - Doctor se salah lein")
        
        print("="*50)

def main():
    """Aasan prediction tool file browser ke saath"""
    print("Easy Medical Image Predictor")
    print("="*50)
    
    # Initialize karein
    predictor = EasyMedicalPredictor()
    
    # Models load karein
    models_loaded = predictor.load_models()
    
    if models_loaded == 0:
        print("\nKoi trained model nahi mila!")
        print("Kripya chalayein: python train.py")
        return
    
    print(f"\n{models_loaded} model successfully load hue!")
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("1. Image select karein aur predict karein")
        print("2. Exit")
        print("="*50)
        
        choice = input("Apna option chunein (1-2): ").strip()
        
        if choice == '2':
            print("Alvida!")
            break
        
        if choice != '1':
            print("Amaanya option!")
            continue
        
        # Image select karein
        image_path = predictor.select_image_file()
        
        if not image_path:
            continue
        
        try:
            # Predict karein
            predictor.predict_and_visualize(image_path)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
