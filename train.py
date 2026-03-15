# Required libraries import kar rahe hain
import os  # File system ke liye
import numpy as np  # Sankhyatmak operations ke liye
import matplotlib.pyplot as plt  # Graphs aur visualization ke liye
import seaborn as sns  # Advanced graphs ke liye
import cv2  # Image processing ke liye (OpenCV)
from PIL import Image  # Image handling ke liye
from tqdm import tqdm  # Progress bar ke liye
import torch  # Deep learning framework PyTorch
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizers ke liye
from torch.utils.data import DataLoader, Dataset  # Data loading ke liye
from torchvision import transforms, models  # Computer vision ke liye
from sklearn.metrics import classification_report, confusion_matrix  # Metrics ke liye
import warnings
warnings.filterwarnings('ignore')  # Warnings ko ignore karne ke liye

# Better looks ke liye style set karein
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")  # Color palette set karein

# GPU Configuration (silent mode) - CUDA available hai to GPU use karenge
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MedicalImageDataset(Dataset):
    """Medical images ke liye custom dataset class"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        self.labels = []
        
        # Image paths ko load karein
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Sirf image files lein
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Image ko load karein
        image = Image.open(img_path).convert('RGB')  # RGB format mein convert karein
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class GPUMedicalClassifier:
    """GPU ke liye optimized medical image classifier"""
    def __init__(self, image_size=224, batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device
        
        # GPU memory ke hisab se batch size adjust karein
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GPU memory in GB
            if gpu_memory >= 8:
                self.batch_size = min(64, batch_size * 2)  # High memory mein larger batch
            elif gpu_memory >= 4:
                self.batch_size = batch_size  # Medium memory
            else:
                self.batch_size = min(16, batch_size)  # Low memory mein smaller batch
        else:
            self.batch_size = min(16, batch_size)  # CPU ke liye small batch
        
        # Data transforms - Training ke liye data augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Image ko resize karein
            transforms.RandomRotation(20),  # Random rotation
            transforms.RandomHorizontalFlip(),  # Random flip
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2, scale=(0.8, 1.2)),  # Random transformations
            transforms.ToTensor(),  # Tensor mein convert karein
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize karein
        ])
        
        # Test transforms - Validation/testing ke liye (no augmentation)
        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Sirf resize
            transforms.ToTensor(),  # Tensor mein convert
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        
        # Models aur class names ko store karein
        self.models = {}
        self.class_names = {}
    
    def create_model(self, num_classes):
        """Transfer learning ke saath model banana"""
        # EfficientNet-B0 use karein (pretrained)
        model = models.efficientnet_b0(pretrained=True)
        
        # Final layer ko modify karein
        num_ftrs = model.classifier[1].in_features  # Last layer ke features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)  # Apne classes ke liye
        
        # GPU par bhejein
        model = model.to(self.device)
        
        # Mixed precision training enable karein agar available hai
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()  # GPU memory bachane ke liye
        else:
            scaler = None
        
        return model, scaler
    
    def create_data_loaders(self, data_path, disease_type):
        """Create data loaders for training and testing"""
        if disease_type == 'brain_tumor':
            train_dir = os.path.join(data_path, 'brain_tumor', 'train')
            test_dir = os.path.join(data_path, 'brain_tumor', 'test')
        else:  # pneumonia
            train_dir = os.path.join(data_path, 'pnemonia', 'train')
            test_dir = os.path.join(data_path, 'pnemonia', 'test')
        
        # Create datasets
        train_dataset = MedicalImageDataset(train_dir, transform=self.train_transform)
        test_dataset = MedicalImageDataset(test_dir, transform=self.test_transform)
        
        # Split train into train/validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        
        # Store class names
        self.class_names[disease_type] = train_dataset.dataset.classes
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, disease_type, epochs=20):
        """Train the model for a specific disease"""
        print(f"\nProcessing {disease_type.replace('_', ' ').title()} Detection")
        print("="*60)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders('Data', disease_type)
        
        # Get number of classes
        num_classes = len(self.class_names[disease_type])
        print(f"Number of classes: {num_classes}")
        print(f"Class names: {self.class_names[disease_type]}")
        
        # Create model
        model, scaler = self.create_model(num_classes)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        
        # Training history
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                              leave=False, ncols=100, colour='green', 
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                if scaler:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', 
                                leave=False, ncols=100, colour='blue',
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            scheduler.step(val_acc)
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_{disease_type}_model.pth')
        
        # Store model and history
        self.models[disease_type] = model
        
        # Plot training history
        self.plot_training_history(disease_type, train_losses, val_losses, train_accs, val_accs)
        
        # Evaluate on test set
        self.evaluate_model(disease_type, test_loader)
        
        return model, test_loader
    
    def plot_training_history(self, disease_type, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_title(f'{disease_type.title()} - Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Training Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_title(f'{disease_type.title()} - Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{disease_type}_training_history.png')
        plt.show()
    
    def evaluate_model(self, disease_type, test_loader):
        """Evaluate model on test set"""
        model = self.models[disease_type]
        model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Classification report
        print(f"\n{disease_type.title()} - Test Results:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=self.class_names[disease_type]))
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names[disease_type], 
                   yticklabels=self.class_names[disease_type])
        plt.title(f'{disease_type.title()} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'{disease_type}_confusion_matrix.png')
        plt.show()
    
    def create_grad_cam(self, model, image_tensor, target_layer='out_relu'):
        """Create Grad-CAM visualization"""
        try:
            # Find the last convolutional layer
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower() or 'out_relu' in layer.name:
                    layer_name = layer.name
                    break
            
            # Create Grad-CAM model
            grad_model = models.Model(
                [model.inputs],
                [model.get_layer(layer_name).output, model.output]
            )
            
            # Compute gradients
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                pred_index = tf.argmax(predictions[0])
                loss = predictions[:, pred_index]
            
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
        
        except Exception as e:
            print(f"Error creating Grad-CAM: {e}")
        # Get sample images
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(images))):
            # Original image
            img_tensor = images[i]
            original_img = img_tensor.permute(1, 2, 0).numpy()
            
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            original_img = original_img * std + mean
            original_img = np.clip(original_img, 0, 1)
            
            # Create Grad-CAM
            cam, pred_idx = self.create_grad_cam(model, img_tensor, target_layer)
            
            # Resize CAM to image size
            cam_resized = cv2.resize(cam, (self.image_size, self.image_size))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Superimpose
            superimposed = cv2.addWeighted(np.uint8(original_img * 255), 0.6, heatmap, 0.4, 0)
            
            # Plot
            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(cam_resized, cmap='jet')
            axes[i, 1].set_title('Grad-CAM Heatmap')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(superimposed)
            true_class = self.class_names[disease_type][labels[i].item()]
            pred_class = self.class_names[disease_type][pred_idx]
            axes[i, 2].set_title(f'Superimposed\nTrue: {true_class}\nPred: {pred_class}')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{disease_type}_grad_cam_visualization.png')
        plt.show()

def main():
    """Main function to run the complete pipeline"""
    print("\n" + "="*60)
    print("MEDICAL IMAGE CLASSIFICATION SYSTEM")
    print("="*60)
    
    # Initialize classifier
    classifier = GPUMedicalClassifier(image_size=224, batch_size=32)
    
    # Train models
    diseases = ['brain_tumor', 'pnemonia']
    
    for disease in diseases:
        try:
            # Train model
            model, test_loader = classifier.train_model(disease, epochs=20)
            
            print(f"\n{disease.replace('_', ' ').title()} detection model trained successfully!")
            
        except Exception as e:
            print(f"Error training {disease} model: {e}")
            continue
    
    print(f"\nTRAINING COMPLETE!")
    print("="*60)
    print("Models saved as:")
    print("   best_brain_tumor_model.pth")
    print("   best_pnemonia_model.pth")
    print("="*60)
    print("Ready for predictions! Run: python predict.py")

if __name__ == "__main__":
    main()
