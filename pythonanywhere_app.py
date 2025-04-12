from flask import Flask, request, jsonify
import os
import sys
import base64
import io
import numpy as np
from PIL import Image
import cv2
import time

app = Flask(__name__)

# Flag to determine if we're running on PythonAnywhere
ON_PYTHONANYWHERE = 'PYTHONANYWHERE_SITE' in os.environ

print(f"Running on PythonAnywhere: {ON_PYTHONANYWHERE}")

# Try to import the required packages
try:
    import torch
    has_torch = True
    print(f"PyTorch available: {torch.__version__}")
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Try to import YOLO
    try:
        from ultralytics import YOLO
        has_yolo = True
        print("Ultralytics YOLO available")
        
        # Try to load YOLO model
        try:
            yolo_model = YOLO("yolov8s.pt").to(device)
            print("YOLOv8s model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            yolo_model = None
    except ImportError as e:
        print(f"YOLO import error: {e}")
        has_yolo = False
        yolo_model = None
    
    # Try to import and load MiDaS
    try:
        # Load MiDaS (only if required packages are available)
        print("Attempting to load MiDaS...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.eval()
        midas = midas.to(device)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        print("MiDaS Small loaded successfully!")
        has_midas = True
    except Exception as e:
        print(f"MiDaS error: {e}")
        has_midas = False
        midas = None
        transform = None
    
except ImportError as e:
    print(f"Import error: {e}")
    has_torch = False
    has_yolo = False
    has_midas = False
    device = "cpu (torch not available)"
    yolo_model = None
    midas = None
    transform = None

# Reuse your original functions with error handling

def detect_objects(image, conf=0.45):
    """Perform object detection on the provided image."""
    if not has_yolo or yolo_model is None:
        raise ValueError("YOLO model not available")
    
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
    if not has_midas or midas is None or transform is None:
        raise ValueError("MiDaS model not available")
        
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

@app.route('/')
def index():
    """API Status endpoint"""
    status = {
        "status": "running",
        "environment": "PythonAnywhere" if ON_PYTHONANYWHERE else "Other",
        "capabilities": {
            "pytorch": has_torch,
            "object_detection": has_yolo and yolo_model is not None,
            "depth_estimation": has_midas and midas is not None
        },
        "device": str(device)
    }
    return jsonify(status)

@app.route('/detect', methods=['POST'])
def detect_api():
    """API endpoint for object detection."""
    if not has_yolo or yolo_model is None:
        return jsonify({"error": "Object detection is not available"}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get confidence threshold from request or use default
        confidence = request.form.get('confidence', default=0.45, type=float)
        
        # Perform detection
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
    if not has_midas or midas is None:
        return jsonify({"error": "Depth estimation is not available"}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Perform depth estimation
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
    
    response = {'status': 'partial_success'}
    
    # Object Detection
    if has_yolo and yolo_model is not None:
        try:
            confidence = request.form.get('confidence', default=0.45, type=float)
            detections = detect_objects(image, conf=confidence)
            response['objects_detected'] = len(detections)
            response['detections'] = detections
        except Exception as e:
            response['object_detection_error'] = str(e)
    else:
        response['object_detection_error'] = "Object detection not available"
    
    # Depth Estimation
    if has_midas and midas is not None:
        try:
            depth_base64, depth_stats = estimate_depth(image)
            response['depth_image'] = depth_base64
            response['depth_stats'] = depth_stats
        except Exception as e:
            response['depth_estimation_error'] = str(e)
    else:
        response['depth_estimation_error'] = "Depth estimation not available"
    
    return jsonify(response)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'models': {
            'object_detection': 'YOLOv8s' if has_yolo and yolo_model is not None else 'Not available',
            'depth_estimation': 'MiDaS_small' if has_midas and midas is not None else 'Not available'
        },
        'device': str(device)
    })

if __name__ == '__main__':
    # Only for local development
    app.run(debug=True) 