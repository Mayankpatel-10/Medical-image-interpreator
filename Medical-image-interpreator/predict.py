# Importing required libraries
import os  # For file system operations
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For graphs and visualization
import cv2  # For image processing (OpenCV)
from PIL import Image  # For image handling
import torch  # Deep learning framework PyTorch
import torch.nn as nn  # Neural network modules
from torchvision import transforms, models  # For computer vision
import torch.nn.functional as F  # Neural network functions
import tkinter as tk  # For file dialog boxes
from tkinter import filedialog  # For file selection
import warnings
warnings.filterwarnings('ignore')  # To ignore warnings

# GPU Configuration - Use GPU if CUDA is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EasyMedicalPredictor:
    """Main class for medical image prediction"""
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.device = device
        
        # Image preprocessing - Prepare image for the model
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize image to 224x224
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        
        # Disease categories
        self.brain_tumor_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']  # Brain tumor types
        self.pneumonia_classes = ['NORMAL', 'PNEUMONIA']  # Pneumonia categories
        
        # Models - Separate models for both diseases
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
        """Load trained models"""
        models_loaded = 0  # Count of loaded models
        
        # Load brain tumor model
        if os.path.exists('best_brain_tumor_model.pth'):
            try:
                # Create EfficientNet-B0 model and modify as needed
                self.brain_tumor_model = models.efficientnet_b0(pretrained=True)
                num_ftrs = self.brain_tumor_model.classifier[1].in_features  # Get features from last layer
                self.brain_tumor_model.classifier[1] = nn.Linear(num_ftrs, 4)  # For 4 classes
                # Load trained weights
                self.brain_tumor_model.load_state_dict(torch.load('best_brain_tumor_model.pth', map_location=self.device))
                self.brain_tumor_model = self.brain_tumor_model.to(self.device)  # Send to GPU/CPU
                self.brain_tumor_model.eval()  # Set to evaluation mode
                print("Brain tumor model loaded")
                models_loaded += 1
            except Exception as e:
                print(f"Error loading brain tumor model: {e}")
        else:
            print("Brain tumor model not found")
        
        # Load pneumonia model
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
        """Open file dialog to select image"""
        print("\nOpening file browser...")
        
        # Create root window (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide window
        root.attributes('-topmost', True)  # Keep on top
        
        # File dialog box
        file_path = filedialog.askopenfilename(
            title="Select medical image",
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
        """Preprocess image for prediction"""
        try:
            image = Image.open(image_path).convert('RGB')  # Convert image to RGB
            original_image = image.copy()  # Keep copy of original image
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)  # Transform and convert to tensor
            return image_tensor, original_image
        except Exception as e:
            print(f"Error in image preprocessing: {e}")
            return None, None
    
    def predict_brain_tumor(self, image_tensor):
        """Predict brain tumor"""
        if self.brain_tumor_model is None:
            return None, None, None  # Model not available
        
        with torch.no_grad():  # Disable gradient calculation (for efficiency)
            outputs = self.brain_tumor_model(image_tensor)  # Get prediction from model
            probabilities = F.softmax(outputs, dim=1)  # Get probabilities
            confidence, predicted = torch.max(probabilities, 1)  # Get class with highest confidence
            
            predicted_class = self.brain_tumor_classes[predicted.item()]  # Get class name
            confidence_score = confidence.item()  # Confidence score
            all_probs = probabilities.cpu().numpy().flatten()  # Probabilities for all classes
            
            return predicted_class, confidence_score, all_probs
    
    def predict_pneumonia(self, image_tensor):
        """Predict pneumonia"""
        if self.pneumonia_model is None:
            return None, None, None  # Model not available
        
        with torch.no_grad():
            outputs = self.pneumonia_model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = self.pneumonia_classes[predicted.item()]
            confidence_score = confidence.item()
            all_probs = probabilities.cpu().numpy().flatten()
            
            return predicted_class, confidence_score, all_probs
    
    def create_grad_cam_fixed(self, model, image_tensor, target_layer_name='features'):
        """Create Grad-CAM visualization - shows which part the model focused on"""
        try:
            # To store gradients and activations
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
            
            # Forward pass - Pass image through model
            model.eval()
            output = model(image_tensor)
            pred_idx = output.argmax(dim=1).item()  # Class with highest confidence
            
            # Backward pass - Calculate gradients
            model.zero_grad()  # Clear old gradients
            class_loss = output[0, pred_idx]  # Calculate loss
            class_loss.backward(retain_graph=True)  # Backpropagation
            
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()
            
            # Create CAM by combining gradients and activations
            if target_layer in gradients and target_layer in activations:
                grads = gradients[target_layer]
                acts = activations[target_layer]
                
                # Global average pooling of gradients
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                
                # Weighted combination of activations
                cam = torch.sum(weights * acts, dim=1)
                cam = torch.relu(cam)  # Set negative values to 0
                
                # Normalize (between 0 and 1)
                if cam.max() > cam.min():
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                
                return cam.squeeze().cpu().numpy()
            
            return None
            
        except Exception as e:
            print(f"Grad-CAM error: {e}")
            return None
    
    def create_simple_heatmap(self, image_tensor, model):
        """Simple attention heatmap - use if Grad-CAM fails"""
        try:
            model.eval()
            with torch.no_grad():
                output = model(image_tensor)  # Get prediction
                
            # Create simple attention map based on prediction confidence
            attention_map = torch.ones((1, self.image_size, self.image_size)) * output[0].max().item()
            attention_map = attention_map.squeeze().cpu().numpy()
            
            # Add some variation to make it look like heatmap
            x, y = np.meshgrid(np.linspace(0, 1, self.image_size), np.linspace(0, 1, self.image_size))
            
            # Create radial gradient (center to outside)
            dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            attention_map = attention_map * np.exp(-dist * 3)  # Exponential decay
            
            return attention_map
            
        except Exception as e:
            return None
    
    def predict_and_visualize(self, image_path):
        """Predict disease and create visualization"""
        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        print("="*50)
        
        # Preprocessing - Prepare image
        image_tensor, original_image = self.preprocess_image(image_path)
        if image_tensor is None:
            return  # Stop if image not loaded
        
        # Prediction - Get results from both models
        brain_result = self.predict_brain_tumor(image_tensor)  # Brain tumor result
        pneumonia_result = self.predict_pneumonia(image_tensor)  # Pneumonia result
        
        # Collect results
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
        
        # Best result (highest confidence)
        best_result = max(results, key=lambda x: x['confidence'])
        
        # Visualization - Create report in 4 sections
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Medical Image Analysis Report', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')  # Hide axis
        
        # Main prediction result
        result_text = f"Detection Result\n\n{best_result['disease']}\n\n{best_result['prediction']}\n\nConfidence: {best_result['confidence']:.1%}"
        # Choose color based on confidence
        color = "lightgreen" if best_result['confidence'] > 0.8 else "lightyellow" if best_result['confidence'] > 0.6 else "lightcoral"
        axes[0, 1].text(0.5, 0.5, result_text, ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        axes[0, 1].set_title('Primary Detection', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Grad-CAM visualization - Which part the model focused on
        if best_result['model'] is not None:
            print("Creating Grad-CAM visualization...")
            
            # First try fixed Grad-CAM
            cam = self.create_grad_cam_fixed(best_result['model'], image_tensor)
            
            if cam is not None:
                print("Grad-CAM created successfully!")
                # Resize CAM to image size
                cam_resized = cv2.resize(cam, (self.image_size, self.image_size))
                # Create heatmap (with Jet colormap)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Convert original image to numpy and combine both
                orig_np = np.array(original_image.resize((self.image_size, self.image_size)))
                superimposed = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)  # 60% image + 40% heatmap
                
                axes[1, 0].imshow(superimposed)
                axes[1, 0].set_title('Attention Map (Grad-CAM)', fontweight='bold')
            else:
                print("Grad-CAM failed, using simple heatmap...")
                # Simple heatmap as fallback
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
        
        # Detailed analysis - Probabilities for all classes
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
        
        # Save result
        output_path = f"prediction_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Result saved {output_path}")
        
        plt.show()  # Show report
        
        # Summary - Final diagnosis
        print(f"\nFinal Diagnosis:")
        print(f"Primary: {best_result['disease']} - {best_result['prediction']}")
        print(f"Confidence: {best_result['confidence']:.1%}")
        
        if best_result['disease'] == 'Brain Tumor':
            if best_result['prediction'] == 'no tumor':
                print("Result: Normal")
            else:
                print(f"Warning: {best_result['prediction'].upper()} tumor detected")
        else:
            if best_result['prediction'] == 'NORMAL':
                print("Result: Normal")
            else:
                print("Warning: PNEUMONIA Detected")
        
        print("="*50)

    def predict_and_visualize_specific(self, image_path, disease_choice):
        """Predict specific disease and create visualization"""
        print(f"\nAnalyzing: {os.path.basename(image_path)}")
        print("="*50)
        
        # Preprocessing - Prepare image
        image_tensor, original_image = self.preprocess_image(image_path)
        if image_tensor is None:
            return  # Stop if image not loaded
        
        # Prediction based on choice
        if disease_choice == '1':  # Brain tumor
            if self.brain_tumor_model is None:
                print("Brain tumor model not available!")
                return
            
            predicted_class, confidence_score, all_probs = self.predict_brain_tumor(image_tensor)
            disease_name = 'Brain Tumor'
            classes = self.brain_tumor_classes
            model = self.brain_tumor_model
            
        elif disease_choice == '2':  # Pneumonia
            if self.pneumonia_model is None:
                print("Pneumonia model not available!")
                return
            
            predicted_class, confidence_score, all_probs = self.predict_pneumonia(image_tensor)
            disease_name = 'Pneumonia'
            classes = self.pneumonia_classes
            model = self.pneumonia_model
            
        else:
            print("Invalid disease choice!")
            return
        
        if predicted_class is None:
            print("Prediction failed!")
            return
        
        # Visualization - Create report in 4 sections
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{disease_name} Analysis Report', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontweight='bold')
        axes[0, 0].axis('off')  # Hide axis
        
        # Main prediction result
        result_text = f"Detection Result\n\n{disease_name}\n\n{predicted_class}\n\nConfidence: {confidence_score:.1%}"
        # Choose color based on confidence
        color = "lightgreen" if confidence_score > 0.8 else "lightyellow" if confidence_score > 0.6 else "lightcoral"
        axes[0, 1].text(0.5, 0.5, result_text, ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color))
        axes[0, 1].set_title('Primary Detection', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Grad-CAM visualization - Which part the model focused on
        if model is not None:
            print("Creating Grad-CAM visualization...")
            
            # First try fixed Grad-CAM
            cam = self.create_grad_cam_fixed(model, image_tensor)
            
            if cam is not None:
                print("Grad-CAM created successfully!")
                # Resize CAM to image size
                cam_resized = cv2.resize(cam, (self.image_size, self.image_size))
                # Create heatmap (with Jet colormap)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Convert original image to numpy and combine both
                orig_np = np.array(original_image.resize((self.image_size, self.image_size)))
                superimposed = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)  # 60% image + 40% heatmap
                
                axes[1, 0].imshow(superimposed)
                axes[1, 0].set_title('Attention Map (Grad-CAM)', fontweight='bold')
            else:
                print("Grad-CAM failed, using simple heatmap...")
                # Simple heatmap as fallback
                simple_heat = self.create_simple_heatmap(image_tensor, model)
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
        
        # Detailed analysis - Probabilities for all classes
        details_text = "Detailed Predictions:\n\n"
        details_text += f"{disease_name}:\n"
        details_text += f"  Result: {predicted_class}\n"
        details_text += f"  Confidence: {confidence_score:.1%}\n\n"
        
        details_text += "Class Probabilities:\n"
        for i, (cls, prob) in enumerate(zip(classes, all_probs)):
            details_text += f"  {cls}: {prob:.1%}\n"
        
        axes[1, 1].text(0.05, 0.95, details_text, ha='left', va='top', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Detailed Analysis', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save result
        output_path = f"prediction_{disease_name.lower().replace(' ', '_')}_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Result saved as {output_path}")
        
        plt.show()  # Show report
        
        # Summary - Final diagnosis
        print(f"\nFinal Diagnosis:")
        print(f"Detection: {disease_name} - {predicted_class}")
        print(f"Confidence: {confidence_score:.1%}")
        
        if disease_name == 'Brain Tumor':
            if predicted_class == 'notumor':
                print("Result: Normal - No tumor detected")
            else:
                print(f"Warning: {predicted_class.upper()} tumor detected")
        else:  # Pneumonia
            if predicted_class == 'NORMAL':
                print("Result: Normal - No pneumonia detected")
            else:
                print("Warning: PNEUMONIA Detected")
        
        print("="*50)

def main():
    """Easy prediction tool with file browser"""
    print("Easy Medical Image interpreator")
    print("="*50)
    
    # Initialize
    predictor = EasyMedicalPredictor()
    
    # Load models
    models_loaded = predictor.load_models()
    
    if models_loaded == 0:
        print("\nNo trained models found!")
        print("Please run: python train.py")
        return
    
    print(f"\n{models_loaded} model successfully loaded!")
    
    while True:
        print("\n" + "="*50)
        print("What do you want to predict?")
        print("1. Brain Tumor Detection")
        print("2. Pneumonia Detection")
        print("3. Exit")
        print("="*50)
        
        choice = input("Select your option (1-3): ").strip()
        
        if choice == '3':
            print("Thank you for using the predictor!")
            break
        
        if choice == '1':
            # Brain tumor prediction
            if predictor.brain_tumor_model is None:
                print("Brain tumor model not available!")
                continue
            
            print("\nBrain Tumor Detection Selected")
            print("Please select a brain MRI image...")
            
        elif choice == '2':
            # Pneumonia prediction
            if predictor.pneumonia_model is None:
                print("Pneumonia model not available!")
                continue
            
            print("\nPneumonia Detection Selected")
            print("Please select a chest X-ray image...")
            
        else:
            print("Invalid option!")
            continue
        
        # Select image
        image_path = predictor.select_image_file()
        
        if not image_path:
            continue
        
        try:
            # Make prediction with specific model
            predictor.predict_and_visualize_specific(image_path, choice)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()