"""
Driver Drowsiness Detection - Model Training Script
Trains a CNN model to classify driver states: Alert, Drowsy, Yawning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import json

class DrowsinessDataset(Dataset):
    """Custom dataset for drowsiness detection"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DrowsinessDetector(nn.Module):
    """CNN model based on MobileNetV2"""
    def __init__(self, num_classes=3):
        super(DrowsinessDetector, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

def load_dataset(data_dir):
    """
    Load dataset from directory structure:
    data_dir/
        alert/
        drowsy/
        yawning/
    """
    classes = ['alert', 'drowsy', 'yawning']
    image_paths = []
    labels = []
    
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found")
            continue
            
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(idx)
    
    return image_paths, labels

def train_model(data_dir='datasets/drowsiness', epochs=20, batch_size=32, learning_rate=0.001):
    """Train the drowsiness detection model"""
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    image_paths, labels = load_dataset(data_dir)
    
    if len(image_paths) == 0:
        print("ERROR: No images found. Please organize your dataset as:")
        print("datasets/drowsiness/alert/, datasets/drowsiness/drowsy/, datasets/drowsiness/yawning/")
        return
    
    print(f"Total images: {len(image_paths)}")
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets and dataloaders
    train_dataset = DrowsinessDataset(X_train, y_train, train_transform)
    val_dataset = DrowsinessDataset(X_val, y_val, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = DrowsinessDetector(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print("\nStarting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'\nEpoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'models/drowsiness_detector_best.pth')
            print(f'Saved best model with validation accuracy: {val_acc:.2f}%')
        
        scheduler.step(val_acc)
    
    # Save training history
    os.makedirs('logs', exist_ok=True)
    with open('logs/training_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    class_names = ['Alert', 'Drowsy', 'Yawning']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'best_val_acc': best_val_acc,
        'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': cm.tolist()
    }
    
    with open('logs/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("Model saved to: models/drowsiness_detector_best.pth")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('datasets/drowsiness/alert', exist_ok=True)
    os.makedirs('datasets/drowsiness/drowsy', exist_ok=True)
    os.makedirs('datasets/drowsiness/yawning', exist_ok=True)
    
    print("Driver Drowsiness Detection - Training Script")
    print("=" * 50)
    print("\nDataset Structure Required:")
    print("datasets/drowsiness/")
    print("  ├── alert/     (images of alert drivers)")
    print("  ├── drowsy/    (images of drowsy/eyes closed)")
    print("  └── yawning/   (images of yawning drivers)")
    print("\nPlace your images in the appropriate folders before training.")
    print("=" * 50)
    
    # Check if dataset exists
    data_dir = 'datasets/drowsiness'
    image_count = sum([len(os.listdir(os.path.join(data_dir, cls))) 
                      for cls in ['alert', 'drowsy', 'yawning'] 
                      if os.path.exists(os.path.join(data_dir, cls))])
    
    if image_count > 0:
        train_model(data_dir=data_dir, epochs=20, batch_size=32)
    else:
        print("\n⚠️  No images found in dataset directories.")
        print("Download a dataset from:")
        print("- Kaggle: 'Drowsiness Detection Dataset'")
        print("- Roboflow: 'Driver Drowsiness Dataset'")
        print("\nOr use the provided data_loader.py script to download sample data.")
