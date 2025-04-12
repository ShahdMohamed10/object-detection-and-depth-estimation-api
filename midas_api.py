from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import torch
import os

app = Flask(__name__)

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MiDaS model
try:
    print("Loading MiDaS Small...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    midas = midas.to(device)
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    print("MiDaS Small loaded successfully!")
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    raise

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
    
    # Create colored depth map
    depth_colored = cv2.applyColorMap((normalized_depth * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    
    # Convert to base64 for API response
    _, buffer = cv2.imencode('.jpg', depth_colored)
    depth_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Also return some statistics about the depth
    depth_stats = {
        'min_depth': float(min_val),
        'max_depth': float(max_val),
        'mean_depth': float(np.mean(depth_map)),
        'median_depth': float(np.median(depth_map))
    }
    
    return depth_base64, depth_stats, normalized_depth

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
        depth_base64, depth_stats, _ = estimate_depth(image)
        return jsonify({
            'status': 'success',
            'depth_image': depth_base64,
            'depth_stats': depth_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/depth_with_distances', methods=['POST'])
def depth_with_points_api():
    """API endpoint for depth estimation with distance measurements at specified points."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    
    # Convert to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get points from the request (format: [x1,y1,x2,y2,...])
    points_str = request.form.get('points', '')
    points = []
    if points_str:
        try:
            # Parse comma-separated points
            points_list = [int(p) for p in points_str.split(',')]
            # Convert flat list to pairs
            points = [(points_list[i], points_list[i+1]) for i in range(0, len(points_list), 2)]
        except:
            return jsonify({'error': 'Invalid points format. Use comma-separated x,y values.'}), 400
    
    # Perform depth estimation
    try:
        depth_base64, depth_stats, normalized_depth = estimate_depth(image)
        
        # If points were provided, calculate depths at those points
        point_depths = []
        if points:
            for x, y in points:
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    # Get relative depth value at this point (0-1 range)
                    relative_depth = float(normalized_depth[y, x])
                    
                    # Scale to the actual depth range
                    scaled_depth = depth_stats['min_depth'] + relative_depth * (depth_stats['max_depth'] - depth_stats['min_depth'])
                    
                    point_depths.append({
                        'position': [x, y],
                        'relative_depth': relative_depth,
                        'scaled_depth': float(scaled_depth)
                    })
        
        return jsonify({
            'status': 'success',
            'depth_image': depth_base64,
            'depth_stats': depth_stats,
            'point_depths': point_depths
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is running."""
    return jsonify({
        'status': 'healthy',
        'model': 'MiDaS_small',
        'device': str(device)
    })

if __name__ == '__main__':
    # Use PORT environment variable for compatibility with cloud services
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False) 