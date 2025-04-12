from ultralytics import YOLO
import torch
import cv2
import numpy as np
from flask import jsonify
import time

class Camera:
    def __init__(self):
        self.camera = None
        self.stream_active = False
        self.frame_count = 0
        self.skip_frames = 2
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self.load_models()
        
    def load_models(self):
        try:
            # Load YOLOv8s
            self.yolo_model = YOLO("yolov8s.pt").to(self.device)
            print("YOLOv8s model loaded successfully!")
            
            # Load MiDaS
            print("Loading MiDaS Small...")
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            self.midas.eval()
            self.midas = self.midas.to(self.device)
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = self.midas_transforms.small_transform
            print("MiDaS Small loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def initialize_camera(self):
        if self.camera is not None:
            self.camera.release()
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise RuntimeError("Could not start camera")
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return self.camera

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def detect_objects(self, frame):
        results = self.yolo_model(frame, conf=0.45)
        return results[0].boxes.data.cpu().numpy()

    def estimate_depth(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        return depth_map

    def process_frame(self, frame):
        # Detect objects
        detections = self.detect_objects(frame)
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_name = self.yolo_model.names[int(cls)]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f'{class_name}: {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Estimate depth
        depth_map = self.estimate_depth(frame)
        depth_colored = cv2.applyColorMap((depth_map * 255).astype(np.uint8), 
                                        cv2.COLORMAP_MAGMA)
        
        cv2.putText(depth_colored, 'Depth Map', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        combined = np.hstack([frame, depth_colored])
        return combined

    def generate_frames(self):
        while self.stream_active:
            if self.camera is None:
                try:
                    self.initialize_camera()
                except RuntimeError:
                    print("Failed to initialize camera")
                    break
            
            self.frame_count += 1
            success, frame = self.camera.read()
            
            if not success:
                print("Failed to read frame")
                break
            else:
                if self.frame_count % self.skip_frames != 0:
                    continue
                    
                try:
                    processed_frame = self.process_frame(frame)
                    ret, buffer = cv2.imencode('.jpg', processed_frame)
                    if not ret:
                        continue
                    
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
            
            time.sleep(0.01)

    def start_stream(self):
        try:
            self.initialize_camera()
            self.stream_active = True
            return jsonify({'status': 'success', 'message': 'Stream started'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})

    def stop_stream(self):
        self.stream_active = False
        self.release_camera()
        return jsonify({'status': 'success', 'message': 'Stream stopped'})