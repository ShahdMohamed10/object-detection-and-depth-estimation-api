from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import time
import logging
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define paths for pickle files
PICKLE_DIR = "pickled_models"
YOLO_PICKLE_PATH = os.path.join(PICKLE_DIR, "yolo_model.pkl")
MIDAS_PICKLE_PATH = os.path.join(PICKLE_DIR, "midas_model.pkl")
MIDAS_TRANSFORM_PICKLE_PATH = os.path.join(PICKLE_DIR, "midas_transform.pkl")

# Initialize variables
yolo_model = None
midas = None
transform = None
device = None

# Create directory for pickled models if it doesn't exist
os.makedirs(PICKLE_DIR, exist_ok=True)

# Initialize models with error handling
try:
    import torch
    from ultralytics import YOLO
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # YOLO Model Loading
    try:
        # Try to load from pickle first
        if os.path.exists(YOLO_PICKLE_PATH):
            logger.info(f"Loading YOLO model from pickle file: {YOLO_PICKLE_PATH}")
            start_time = time.time()
            with open(YOLO_PICKLE_PATH, 'rb') as f:
                yolo_model = pickle.load(f)
            logger.info(f"YOLO model loaded from pickle in {time.time() - start_time:.2f} seconds")
        else:
            # Load model normally and then pickle it
            logger.info("Loading YOLO model from weights file")
            start_time = time.time()
            yolo_model_path = os.environ.get('YOLO_MODEL_PATH', 'yolov8s.pt')
            yolo_model = YOLO(yolo_model_path).to(device)
            logger.info(f"YOLO model loaded in {time.time() - start_time:.2f} seconds")
            
            # Save model to pickle for faster loading next time
            logger.info(f"Saving YOLO model to pickle file: {YOLO_PICKLE_PATH}")
            with open(YOLO_PICKLE_PATH, 'wb') as f:
                pickle.dump(yolo_model, f)
            logger.info("YOLO model saved to pickle file")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        yolo_model = None
    
    # MiDaS Model Loading
    try:
        # First try to load MiDaS model from pickle
        if os.path.exists(MIDAS_PICKLE_PATH) and os.path.exists(MIDAS_TRANSFORM_PICKLE_PATH):
            logger.info("Loading MiDaS model and transform from pickle files")
            start_time = time.time()
            with open(MIDAS_PICKLE_PATH, 'rb') as f:
                midas = pickle.load(f)
            with open(MIDAS_TRANSFORM_PICKLE_PATH, 'rb') as f:
                transform = pickle.load(f)
            logger.info(f"MiDaS model loaded from pickle in {time.time() - start_time:.2f} seconds")
        else:
            # Load MiDaS normally and then pickle it
            logger.info("Loading MiDaS model from torch hub")
            start_time = time.time()
            midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            midas.eval()
            midas = midas.to(device)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.small_transform
            logger.info(f"MiDaS model loaded in {time.time() - start_time:.2f} seconds")
            
            # Save to pickle for faster loading next time
            logger.info("Saving MiDaS model and transform to pickle files")
            with open(MIDAS_PICKLE_PATH, 'wb') as f:
                pickle.dump(midas, f)
            with open(MIDAS_TRANSFORM_PICKLE_PATH, 'wb') as f:
                pickle.dump(transform, f)
            logger.info("MiDaS model and transform saved to pickle files")
    except Exception as e:
        logger.error(f"Error loading MiDaS model: {e}")
        midas = None
        transform = None
        
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.warning("Running with limited functionality due to missing packages.")

def detect_objects(image, conf=0.45):
    """Perform object detection on the provided image."""
    if yolo_model is None:
        raise ValueError("YOLO model not available")
    
    # Record start time for performance tracking
    start_time = time.time()
    
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
    
    processing_time = time.time() - start_time
    logger.info(f"YOLO inference completed in {processing_time:.2f} seconds")
    
    return detections, processing_time

def estimate_depth(image):
    """Estimate depth for the provided image."""
    if midas is None or transform is None:
        raise ValueError("MiDaS model not available")
    
    # Record start time for performance tracking
    start_time = time.time()
    
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
    
    processing_time = time.time() - start_time
    logger.info(f"MiDaS inference completed in {processing_time:.2f} seconds")
    
    return depth_base64, depth_stats, processing_time

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        'name': 'Object Detection and Depth Estimation API (Pickle Optimized)',
        'version': '1.0',
        'endpoints': {
            'health': '/health',
            'object_detection': '/detect',
            'depth_estimation': '/depth',
            'combined_analysis': '/analyze'
        },
        'documentation': 'https://github.com/ShahdMohamed10/object-detection-and-depth-estimation-api'
    })

@app.route('/detect', methods=['POST'])
def detect_api():
    """API endpoint for object detection."""
    if yolo_model is None:
        return jsonify({"error": "Object detection model not available"}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the image
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get confidence threshold from request or use default
        confidence = request.form.get('confidence', default=0.45, type=float)
        
        # Perform detection
        detections, inference_time = detect_objects(image, conf=confidence)
        
        return jsonify({
            'status': 'success',
            'inference_time': f"{inference_time:.2f}s",
            'objects_detected': len(detections),
            'detections': detections
        })
    except Exception as e:
        logger.error(f"Error in object detection endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/depth', methods=['POST'])
def depth_api():
    """API endpoint for depth estimation."""
    if midas is None or transform is None:
        return jsonify({"error": "Depth estimation model not available"}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the image
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Perform depth estimation
        depth_base64, depth_stats, inference_time = estimate_depth(image)
        
        return jsonify({
            'status': 'success',
            'inference_time': f"{inference_time:.2f}s",
            'depth_image': depth_base64,
            'depth_stats': depth_stats
        })
    except Exception as e:
        logger.error(f"Error in depth estimation endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_api():
    """API endpoint that performs both object detection and depth estimation."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Initialize response
    response = {'status': 'partial_success'}
    
    try:
        # Get the image
        file = request.files['image']
        img_bytes = file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Get confidence threshold from request or use default
        confidence = request.form.get('confidence', default=0.45, type=float)
        
        # Perform object detection (if available)
        if yolo_model is not None:
            try:
                detections, detection_time = detect_objects(image, conf=confidence)
                response['objects_detected'] = len(detections)
                response['detections'] = detections
                response['detection_time'] = f"{detection_time:.2f}s"
            except Exception as e:
                logger.error(f"Error in object detection: {str(e)}")
                response['object_detection_error'] = str(e)
        else:
            response['object_detection_error'] = "Object detection model not available"
        
        # Perform depth estimation (if available)
        if midas is not None and transform is not None:
            try:
                depth_base64, depth_stats, depth_time = estimate_depth(image)
                response['depth_image'] = depth_base64
                response['depth_stats'] = depth_stats
                response['depth_time'] = f"{depth_time:.2f}s"
            except Exception as e:
                logger.error(f"Error in depth estimation: {str(e)}")
                response['depth_estimation_error'] = str(e)
        else:
            response['depth_estimation_error'] = "Depth estimation model not available"
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'models': {
            'object_detection': {
                'status': 'Available' if yolo_model is not None else 'Not available',
                'type': 'YOLOv8s',
                'pickle_file': YOLO_PICKLE_PATH,
                'pickle_exists': os.path.exists(YOLO_PICKLE_PATH)
            },
            'depth_estimation': {
                'status': 'Available' if midas is not None and transform is not None else 'Not available',
                'type': 'MiDaS_small',
                'pickle_file': MIDAS_PICKLE_PATH,
                'pickle_exists': os.path.exists(MIDAS_PICKLE_PATH)
            }
        },
        'device': str(device) if device is not None else 'No device (CPU)',
        'pickle_directory': os.path.abspath(PICKLE_DIR)
    })

@app.route('/clear-pickle', methods=['POST'])
def clear_pickle():
    """Endpoint to clear pickle files and reload models"""
    try:
        if os.path.exists(YOLO_PICKLE_PATH):
            os.remove(YOLO_PICKLE_PATH)
            logger.info(f"Removed YOLO pickle file: {YOLO_PICKLE_PATH}")
        
        if os.path.exists(MIDAS_PICKLE_PATH):
            os.remove(MIDAS_PICKLE_PATH)
            logger.info(f"Removed MiDaS pickle file: {MIDAS_PICKLE_PATH}")
        
        if os.path.exists(MIDAS_TRANSFORM_PICKLE_PATH):
            os.remove(MIDAS_TRANSFORM_PICKLE_PATH)
            logger.info(f"Removed MiDaS transform pickle file: {MIDAS_TRANSFORM_PICKLE_PATH}")
        
        # Refresh page to reload models
        return jsonify({
            'status': 'success',
            'message': 'Pickle files cleared successfully. Models will be reloaded on next use.'
        })
    except Exception as e:
        logger.error(f"Error clearing pickle files: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use PORT environment variable for compatibility with cloud services
    port = int(os.environ.get('PORT', 5000))
    # In production, debug should be False
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting server on port {port}, debug mode: {debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode) 