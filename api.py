from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO
import torch
import time
import os

app = Flask(__name__)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models
try:
    # Load YOLOv8s
    yolo_model = YOLO("yolov8s.pt").to(device)
    print("YOLOv8s model loaded successfully!")
    
    # Load MiDaS
    print("Loading MiDaS Small...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    midas = midas.to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    print("MiDaS Small loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

def detect_objects(image, conf=0.45):
    """Perform object detection on the provided image."""
    results = yolo_model(image, conf=conf)
    
    # Process results
    detections = []
    boxes = results[0].boxes.data.cpu().numpy()
    
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = yolo_model.names[int(cls)]
        
        detections.append({
            'class': class_name,
            'confidence': float(conf),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
    
    return detections

def estimate_depth(image):
    """Estimate depth for the provided image."""
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map to 0-1 range
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    normalized_depth = (depth_map - min_val) / (max_val - min_val)
    
    # Convert to base64 for API response
    depth_colored = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    _, buffer = cv2.imencode('.jpg', depth_colored)
    depth_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Also return some statistics about the depth
    depth_stats = {
        'min_depth': float(min_val),
        'max_depth': float(max_val),
        'mean_depth': float(np.mean(depth_map)),
        'median_depth': float(np.median(depth_map))
    }
    
    return depth_base64, depth_stats

@app.route('/detect', methods=['POST'])
def detect_api():
    """API endpoint for object detection."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get confidence threshold from request or use default
    confidence = request.form.get('confidence', default=0.45, type=float)
    
    # Perform detection
    try:
        detections = detect_objects(image, conf=confidence)
        return jsonify({
            'status': 'success',
            'objects_detected': len(detections),
            'detections': detections
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/depth', methods=['POST'])
def depth_api():
    """API endpoint for depth estimation."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform depth estimation
    try:
        depth_base64, depth_stats = estimate_depth(image)
        return jsonify({
            'status': 'success',
            'depth_image': depth_base64,
            'depth_stats': depth_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_api():
    """API endpoint that performs both object detection and depth estimation."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get confidence threshold from request or use default
    confidence = request.form.get('confidence', default=0.45, type=float)
    
    # Perform analysis
    try:
        # Object Detection
        detections = detect_objects(image, conf=confidence)
        
        # Depth Estimation
        depth_base64, depth_stats = estimate_depth(image)
        
        return jsonify({
            'status': 'success',
            'objects_detected': len(detections),
            'detections': detections,
            'depth_image': depth_base64,
            'depth_stats': depth_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'object_detection': 'YOLOv8s',
            'depth_estimation': 'MiDaS_small'
        },
        'device': str(device)
    })

if __name__ == '__main__':
    # Use PORT environment variable for compatibility with cloud services
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 