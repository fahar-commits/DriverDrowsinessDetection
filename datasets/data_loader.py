"""
Driver Drowsiness Detection - Dataset Loader
Downloads and prepares sample dataset for training
"""

import os
import urllib.request
import zipfile
import shutil
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def setup_sample_dataset():
    """
    Download and setup sample drowsiness detection dataset
    """
    print("="*60)
    print("Driver Drowsiness Detection - Dataset Loader")
    print("="*60)
    
    dataset_dir = 'datasets/drowsiness'
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\n📦 Dataset Options:")
    print("\n1. Manual Setup (Recommended)")
    print("   - Download dataset from Kaggle or Roboflow")
    print("   - Extract and organize into alert/drowsy/yawning folders")
    print("\n2. Create Sample Structure")
    print("   - Creates folder structure for manual organization")
    print("\n3. Download Sample Images (Limited)")
    print("   - Downloads small sample set for testing only")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        print("\n📥 Manual Download Instructions:")
        print("\nRecommended Datasets:")
        print("\n1. Kaggle - Drowsiness Detection Dataset")
        print("   URL: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset")
        print("   Steps:")
        print("   - Download the dataset ZIP")
        print("   - Extract to datasets/drowsiness/")
        print("   - Organize images into alert/, drowsy/, yawning/ folders")
        
        print("\n2. Roboflow - Driver Drowsiness Dataset")
        print("   URL: https://universe.roboflow.com/drowsiness-detection")
        print("   Steps:")
        print("   - Sign up for free account")
        print("   - Download dataset in 'Folder Structure' format")
        print("   - Extract to datasets/drowsiness/")
        
        print("\n3. YawnDD Dataset (Research Grade)")
        print("   URL: https://ieee-dataport.org/open-access/yawndd-yawning-detection-dataset")
        
        print("\n✅ After downloading:")
        print("   Run: python train_model.py")
        
    elif choice == '2':
        print("\n📁 Creating folder structure...")
        
        folders = [
            'datasets/drowsiness/alert',
            'datasets/drowsiness/drowsy',
            'datasets/drowsiness/yawning'
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            
            # Create README in each folder
            readme_path = os.path.join(folder, 'README.txt')
            with open(readme_path, 'w') as f:
                f.write(f"Place {os.path.basename(folder)} driver images here.\n")
                f.write(f"Recommended: Minimum 300 images per category.\n")
                f.write(f"Supported formats: .jpg, .jpeg, .png\n")
        
        print("✅ Folder structure created!")
        print("\nNext steps:")
        print("1. Add images to each folder:")
        for folder in folders:
            print(f"   - {folder}")
        print("2. Run: python train_model.py")
        
    elif choice == '3':
        print("\n⚠️  Note: Sample images are for TESTING ONLY")
        print("For actual training, use option 1 or 2 with real datasets.")
        
        confirm = input("\nContinue with sample setup? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Setup cancelled.")
            return
        
        print("\n🔧 Creating sample dataset structure...")
        
        # Create sample structure
        folders = {
            'alert': 'datasets/drowsiness/alert',
            'drowsy': 'datasets/drowsiness/drowsy',
            'yawning': 'datasets/drowsiness/yawning'
        }
        
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Create README with instructions
        readme_content = """
Driver Drowsiness Detection - Sample Dataset

⚠️  IMPORTANT: This is a MINIMAL sample structure for testing only.

For actual training, you need:
- Minimum 300-500 images per category
- High quality, diverse images
- Proper lighting conditions
- Multiple subjects

Recommended Sources:
1. Kaggle: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
2. Roboflow: https://universe.roboflow.com/drowsiness-detection
3. YawnDD: https://ieee-dataport.org/open-access/yawndd-yawning-detection-dataset

Dataset Structure:
datasets/drowsiness/
  ├── alert/     - Images of alert, attentive drivers (eyes open)
  ├── drowsy/    - Images of drowsy drivers (eyes closed/heavy)
  └── yawning/   - Images of drivers yawning

Image Requirements:
- Format: JPEG or PNG
- Resolution: Minimum 224x224 pixels
- Content: Clear face visibility
- Diversity: Multiple subjects, lighting conditions, angles

After adding images, run: python train_model.py
"""
        
        with open('datasets/drowsiness/README.txt', 'w') as f:
            f.write(readme_content)
        
        print("✅ Sample structure created!")
        print("\n📝 README file created with instructions")
        print("\nNext steps:")
        print("1. Download a proper dataset (see README.txt)")
        print("2. Organize images into the folders")
        print("3. Run: python train_model.py")
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)

def validate_dataset():
    """Validate dataset structure and count images"""
    print("\n🔍 Validating dataset...")
    
    dataset_dir = 'datasets/drowsiness'
    classes = ['alert', 'drowsy', 'yawning']
    
    total_images = 0
    validation_passed = True
    
    print("\nDataset Statistics:")
    print("-" * 40)
    
    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        
        if not os.path.exists(class_dir):
            print(f"❌ {cls}: Folder not found")
            validation_passed = False
            continue
        
        image_count = len([f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        total_images += image_count
        status = "✅" if image_count >= 300 else "⚠️ "
        print(f"{status} {cls.capitalize()}: {image_count} images")
        
        if image_count < 300:
            print(f"   Recommended: Minimum 300 images (currently {image_count})")
            validation_passed = False
    
    print("-" * 40)
    print(f"Total: {total_images} images")
    
    if validation_passed:
        print("\n✅ Dataset validation passed!")
        print("Ready to train. Run: python train_model.py")
    else:
        print("\n⚠️  Dataset needs more images for optimal training")
        print("Minimum recommended: 300 images per category")
        print("\nYou can still train with less data, but accuracy may be lower.")
    
    return validation_passed

def create_sample_alert_sound():
    """Create a simple beep sound for alerts"""
    print("\n🔊 Setting up alert sound...")
    
    sounds_dir = 'sounds'
    os.makedirs(sounds_dir, exist_ok=True)
    
    alert_path = os.path.join(sounds_dir, 'alert.wav')
    
    if os.path.exists(alert_path):
        print(f"Alert sound already exists: {alert_path}")
        return
    
    try:
        import numpy as np
        from scipy.io import wavfile
        
        # Generate simple beep (440 Hz for 1 second)
        sample_rate = 44100
        duration = 1.0
        frequency = 440
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add fade in/out to avoid clicks
        fade_samples = int(0.01 * sample_rate)
        audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
        audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        # Convert to 16-bit integer
        audio = (audio * 32767).astype(np.int16)
        
        wavfile.write(alert_path, sample_rate, audio)
        print(f"✅ Alert sound created: {alert_path}")
        
    except ImportError:
        print("⚠️  scipy not installed. Skipping alert sound generation.")
        print("You can add your own alert.wav file to the sounds/ folder")
        
        # Create a placeholder file
        with open(alert_path + '.txt', 'w') as f:
            f.write("Place your alert.wav file here.\n")
            f.write("You can use any audio editing software to create an alert sound.\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Driver Drowsiness Detection - Dataset Setup Utility")
    print("="*60)
    
    # Run setup
    setup_sample_dataset()
    
    # Validate if dataset exists
    if os.path.exists('datasets/drowsiness/alert'):
        print("\n")
        validate_dataset()
    
    # Create alert sound
    create_sample_alert_sound()
    
    print("\n" + "="*60)
    print("All done! Check README.txt in datasets/drowsiness/ for details")
    print("="*60 + "\n")
