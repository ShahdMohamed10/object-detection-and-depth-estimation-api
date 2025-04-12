import requests
import json
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# The API URL (change this to your deployed API URL)
API_URL = "http://localhost:5002"

def call_depth_api(image_path):
    """Call the basic depth estimation API with an image file."""
    endpoint = f"{API_URL}/depth"
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        
        response = requests.post(endpoint, files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def call_depth_with_points_api(image_path, points=None):
    """Call the depth estimation API with distance measurements at specific points."""
    endpoint = f"{API_URL}/depth_with_distances"
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {}
        
        if points:
            # Convert points to flat list and then to comma-separated string
            flat_points = [str(coord) for point in points for coord in point]
            data['points'] = ','.join(flat_points)
        
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def display_results(image_path, api_response):
    """Display the results from the API."""
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Display original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    # Display depth image
    if 'depth_image' in api_response:
        # Decode base64 depth image
        depth_bytes = base64.b64decode(api_response['depth_image'])
        depth_image = Image.open(io.BytesIO(depth_bytes))
        depth_image = np.array(depth_image)
        
        plt.subplot(1, 2, 2)
        plt.title('Depth Map')
        plt.imshow(depth_image)
        plt.axis('off')
        
        # If there are point measurements, show them
        if 'point_depths' in api_response and api_response['point_depths']:
            # Create a copy of the depth image to plot points on
            depth_with_points = depth_image.copy()
            
            for point_info in api_response['point_depths']:
                x, y = point_info['position']
                depth = point_info['scaled_depth']
                
                # Draw a circle at the point
                cv2.circle(depth_with_points, (x, y), 5, (0, 255, 0), -1)
                
                # Add text with depth value
                cv2.putText(depth_with_points, f"{depth:.2f}", (x+10, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            plt.subplot(1, 2, 2)
            plt.imshow(depth_with_points)
    
    plt.tight_layout()
    plt.show()
    
    # Print depth statistics
    if 'depth_stats' in api_response:
        print("\nDepth Statistics:")
        for key, value in api_response['depth_stats'].items():
            print(f"  {key}: {value}")
    
    # Print point depth measurements
    if 'point_depths' in api_response and api_response['point_depths']:
        print("\nDepth at specified points:")
        for i, point_info in enumerate(api_response['point_depths']):
            x, y = point_info['position']
            rel_depth = point_info['relative_depth']
            scaled_depth = point_info['scaled_depth']
            print(f"  Point {i+1} at ({x}, {y}): {scaled_depth:.2f} (relative: {rel_depth:.2f})")

def check_api_health():
    """Check if the API is healthy and running."""
    endpoint = f"{API_URL}/health"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            print("MiDaS API is healthy!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"API health check failed with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to API at {API_URL}")
        return False

def main():
    # First, check if the API is running
    if not check_api_health():
        print("Please make sure the MiDaS API is running before proceeding.")
        return
    
    # Example usage
    print("\nMiDaS Depth Estimation API Client Example")
    print("========================================")
    
    # Ask for image path
    image_path = input("Enter path to image file: ")
    
    # Choose API endpoint
    print("\nChoose API endpoint:")
    print("1. Basic depth estimation")
    print("2. Depth estimation with point measurements")
    
    choice = input("\nEnter your choice (1-2): ")
    
    if choice == '1':
        # Call the basic depth API
        response = call_depth_api(image_path)
        
        if response:
            print("\nDepth estimation completed successfully")
            display_results(image_path, response)
    
    elif choice == '2':
        # Ask for points
        use_points = input("\nWould you like to specify points to measure depths? (y/n): ").lower() == 'y'
        
        points = []
        if use_points:
            print("\nEnter points as x,y coordinates (e.g., '100,200'). Type 'done' when finished.")
            while True:
                point_input = input("Enter point (or 'done'): ")
                if point_input.lower() == 'done':
                    break
                try:
                    x, y = map(int, point_input.split(','))
                    points.append((x, y))
                    print(f"Added point ({x}, {y})")
                except:
                    print("Invalid format. Use 'x,y' (e.g., '100,200')")
        
        # Call the depth with points API
        response = call_depth_with_points_api(image_path, points)
        
        if response:
            print("\nDepth estimation with points completed successfully")
            display_results(image_path, response)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 