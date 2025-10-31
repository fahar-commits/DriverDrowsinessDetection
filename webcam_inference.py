"""
Driver Drowsiness Detection - Real-time Webcam Inference
Detects drowsiness in real-time and triggers alerts
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import time
import threading
import os
from collections import deque
try:
    from playsound import playsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("Warning: playsound not installed. Audio alerts disabled.")

class DrowsinessDetector(nn.Module):
    """CNN model based on MobileNetV2"""
    def __init__(self, num_classes=3):
        super(DrowsinessDetector, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

class AlertSystem:
    """Handles audio and visual alerts"""
    def __init__(self, sound_path='sounds/alert.wav'):
        self.sound_path = sound_path
        self.is_alerting = False
        self.alert_cooldown = 3  # seconds between alerts
        self.last_alert_time = 0
        
    def trigger_alert(self, alert_type='DROWSY'):
        """Trigger alert with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.last_alert_time = current_time
        self.is_alerting = True
        
        # Play sound in separate thread
        if SOUND_AVAILABLE and os.path.exists(self.sound_path):
            threading.Thread(target=self._play_sound, daemon=True).start()
        
        # Console alert
        print(f"\n{'='*50}")
        print(f"⚠️  ALERT: DRIVER {alert_type}! ⚠️")
        print(f"{'='*50}\n")
        
        # In real hardware, this is where you'd send signals to:
        # - Vibration motors in steering wheel
        # - Seat vibration actuators
        # - Dashboard warning lights
        
        # Reset alert flag after delay
        threading.Timer(2.0, self._reset_alert).start()
    
    def _play_sound(self):
        """Play alert sound"""
        try:
            playsound(self.sound_path)
        except Exception as e:
            print(f"Error playing sound: {e}")
    
    def _reset_alert(self):
        """Reset alert flag"""
        self.is_alerting = False

class DrowsinessMonitor:
    """Real-time drowsiness monitoring system"""
    def __init__(self, model_path='models/drowsiness_detector_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = DrowsinessDetector(num_classes=3)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print(f"WARNING: Model file not found at {model_path}")
            print("Using untrained model. Please train the model first using train_model.py")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Alert system
        self.alert_system = AlertSystem()
        
        # Class names
        self.classes = ['Alert', 'Drowsy', 'Yawning']
        
        # Smoothing buffer (temporal smoothing)
        self.prediction_buffer = deque(maxlen=10)
        self.drowsy_threshold = 0.6  # 60% drowsy predictions trigger alert
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
    def preprocess_frame(self, frame):
        """Preprocess webcam frame for model"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, frame):
        """Make prediction on frame"""
        tensor = self.preprocess_frame(frame)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = self.classes[predicted.item()]
        confidence_value = confidence.item()
        
        return predicted_class, confidence_value, probabilities[0].cpu().numpy()
    
    def get_smoothed_prediction(self, current_pred):
        """Apply temporal smoothing to reduce false positives"""
        self.prediction_buffer.append(current_pred)
        
        if len(self.prediction_buffer) < 5:
            return current_pred
        
        # Count predictions in buffer
        pred_counts = {'Alert': 0, 'Drowsy': 0, 'Yawning': 0}
        for pred in self.prediction_buffer:
            pred_counts[pred] += 1
        
        # Return most common prediction
        return max(pred_counts, key=pred_counts.get)
    
    def should_alert(self):
        """Check if alert should be triggered based on buffer"""
        if len(self.prediction_buffer) < 5:
            return False
        
        drowsy_count = sum(1 for pred in self.prediction_buffer if pred in ['Drowsy', 'Yawning'])
        drowsy_ratio = drowsy_count / len(self.prediction_buffer)
        
        return drowsy_ratio >= self.drowsy_threshold
    
    def draw_ui(self, frame, predicted_class, confidence, probabilities):
        """Draw UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Status box
        status_color = (0, 255, 0) if predicted_class == 'Alert' else (0, 0, 255)
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Status text
        cv2.putText(frame, f"Status: {predicted_class}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Probability bars
        y_offset = 130
        for i, (cls, prob) in enumerate(zip(self.classes, probabilities)):
            bar_width = int(prob * 300)
            color = (0, 255, 0) if cls == 'Alert' else (0, 165, 255)
            
            cv2.rectangle(frame, (20, y_offset + i*30), (20 + bar_width, y_offset + i*30 + 20),
                         color, -1)
            cv2.putText(frame, f"{cls}: {prob:.2%}", (330, y_offset + i*30 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Alert indicator
        if self.alert_system.is_alerting:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 20)
            cv2.putText(frame, "!!! ALERT !!!", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        return frame
    
    def run(self, camera_id=0):
        """Run real-time monitoring"""
        print("\n" + "="*60)
        print("Driver Drowsiness Detection System - ACTIVE")
        print("="*60)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'r' to reset statistics")
        print("\n⚠️  Ensure good lighting for best results")
        print("="*60 + "\n")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Webcam initialized. Starting detection...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Make prediction
                predicted_class, confidence, probabilities = self.predict(frame)
                
                # Apply temporal smoothing
                smoothed_prediction = self.get_smoothed_prediction(predicted_class)
                
                # Check if alert needed
                if self.should_alert() and smoothed_prediction != 'Alert':
                    self.alert_system.trigger_alert(smoothed_prediction)
                
                # Draw UI
                frame = self.draw_ui(frame, smoothed_prediction, confidence, probabilities)
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = self.frame_count / elapsed
                
                # Display
                cv2.imshow('Driver Drowsiness Detection', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nShutting down...")
                    break
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.frame_count = 0
                    self.start_time = time.time()
                    self.prediction_buffer.clear()
                    print("Statistics reset")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\nSession Statistics:")
            print(f"  Total frames processed: {self.frame_count}")
            print(f"  Average FPS: {self.fps:.2f}")
            print(f"  Runtime: {time.time() - self.start_time:.2f} seconds")
            print("\nThank you for using Driver Drowsiness Detection System!")

if __name__ == "__main__":
    # Check for model file
    if not os.path.exists('models/drowsiness_detector_best.pth'):
        print("⚠️  Warning: Trained model not found!")
        print("Please train the model first using: python train_model.py")
        print("\nContinuing with untrained model (predictions will be random)...\n")
        input("Press Enter to continue...")
    
    # Create sounds directory if it doesn't exist
    os.makedirs('sounds', exist_ok=True)
    
    # Initialize and run monitor
    monitor = DrowsinessMonitor()
    monitor.run(camera_id=0)
