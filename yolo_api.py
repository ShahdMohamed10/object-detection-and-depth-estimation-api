from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import torch
import os

app = Flask(__name__)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLO model
try:
    # Load YOLOv8s
    yolo_model = YOLO("yolov8s.pt").to(device)
    print("YOLOv8s model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
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
        
        # Create a visualized result image
        result_image = image.copy()
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class']
            conf = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(result_image, label, (bbox[0], bbox[1]-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'objects_detected': len(detections),
            'detections': detections,
            'result_image': result_base64
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'model': 'YOLOv8s',
        'device': str(device)
    })

if __name__ == '__main__':
    # Use PORT environment variable for compatibility with cloud services
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False) 