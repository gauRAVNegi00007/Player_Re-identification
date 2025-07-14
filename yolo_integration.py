import cv2
import numpy as np
import torch
import os
from urllib.request import urlretrieve

class YOLOPlayerDetector:
    """YOLO-based player detector using Ultralytics YOLOv5"""
    def __init__(self, model_url: str = "https://drive.google.com/file/d/1-SfOSHOSB9UXP_enOzNAMScrePVcMDview"):
        self.model_path = "player_detection_model.pt"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        if not os.path.exists(self.model_path):
            print("Downloading YOLO model...")
            self.download_model(model_url)
        
        self.load_model()
    
    def download_model(self, url: str):
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading from: {download_url}")
        urlretrieve(download_url, self.model_path)
        print("Model downloaded successfully!")
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = torch.hub.load(
                    'ultralytics/yolov5',
                    'custom',
                    path=self.model_path,
                    force_reload=True   # ✅ Fix for grid attribute bug!
                ).autoshape()  # Add autoshape for numpy support
                self.model.to(self.device)
                print("YOLO model loaded successfully!")
            else:
                print("Model file not found. Using fallback detection.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def detect_players(self, frame: np.ndarray):
        if self.model is None:
            return self.fallback_detection(frame)
        
        try:
            results = self.model(frame)
            detections = []
            for *box, conf, cls in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                detections.append((x1, y1, x2, y2, float(conf)))
            return detections
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return self.fallback_detection(frame)
    
    def fallback_detection(self, frame: np.ndarray):
        if not hasattr(self, 'bg_subtractor'):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 1.2 < aspect_ratio < 4.0:
                    confidence = min(1.0, area / 5000)
                    detections.append((x, y, x + w, y + h, confidence))
        return detections
    
class EnhancedFeatureExtractor:
    """Enhanced feature extraction for re-identification"""
    def extract_features(self, frame: np.ndarray, bbox):
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(256)
        player_region = frame[y1:y2, x1:x2]
        if player_region.size == 0:
            return np.zeros(256)
        
        color = self.extract_color_features(player_region)
        texture = self.extract_texture_features(player_region)
        shape = self.extract_shape_features(player_region)
        features = np.concatenate([color, texture, shape])
        features /= np.linalg.norm(features) + 1e-7
        return features

    def extract_color_features(self, region):
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        return np.concatenate([hist_h, hist_s, hist_v]) / (np.sum(hist_h) + np.sum(hist_s) + np.sum(hist_v) + 1e-7)

    def extract_texture_features(self, region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.array([np.mean(mag), np.std(mag)])

    def extract_shape_features(self, region):
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            return np.array([area/10000, perimeter/1000])
        else:
            return np.array([0, 0])