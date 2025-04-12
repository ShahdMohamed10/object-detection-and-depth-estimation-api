import requests
import json
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# The API URL (change this to your deployed API URL)
API_URL = "http://localhost:5000"

def call_detect_api(image_path, confidence=0.45):
    """Call the object detection API with an image file."""
    endpoint = f"{API_URL}/detect"
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'confidence': confidence}
        
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def call_depth_api(image_path):
    """Call the depth estimation API with an image file."""
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

def call_analyze_api(image_path, confidence=0.45):
    """Call the analyze API (both detection and depth) with an image file."""
    endpoint = f"{API_URL}/analyze"
    
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        data = {'confidence': confidence}
        
        response = requests.post(endpoint, files=files, data=data)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def display_results(image_path, api_response):
    """Display the results from the API."""
    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw bounding boxes if detections are present
    if 'detections' in api_response:
        for detection in api_response['detections']:
            bbox = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(image, label, (bbox[0], bbox[1]-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Show original image with detections
    plt.subplot(1, 2, 1)
    plt.title('Object Detection')
    plt.imshow(image)
    plt.axis('off')
    
    # Show depth map if available
    if 'depth_image' in api_response:
        # Decode base64 depth image
        depth_bytes = base64.b64decode(api_response['depth_image'])
        depth_image = Image.open(io.BytesIO(depth_bytes))
        depth_image = np.array(depth_image)
        
        # Display depth image
        plt.subplot(1, 2, 2)
        plt.title('Depth Map')
        plt.imshow(depth_image)
        plt.axis('off')
        
        # If depth stats are available, print them
        if 'depth_stats' in api_response:
            stats = api_response['depth_stats']
            print(f"Depth Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
    
    plt.tight_layout()
    plt.show()

def check_api_health():
    """Check if the API is healthy and running."""
    endpoint = f"{API_URL}/health"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            print("API is healthy!")
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
        print("Please make sure the API is running before proceeding.")
        return
    
    # Example usage
    print("\nAPI Client Example")
    print("=================")
    
    # Ask for image path
    image_path = input("Enter path to image file: ")
    
    # Choose which API to call
    print("\nWhich API would you like to call?")
    print("1. Object Detection Only")
    print("2. Depth Estimation Only")
    print("3. Both (Object Detection + Depth Estimation)")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        confidence = input("Enter confidence threshold (0.1-1.0, default 0.45): ") or 0.45
        response = call_detect_api(image_path, float(confidence))
        if response:
            print(f"\nDetected {response.get('objects_detected', 0)} objects")
            display_results(image_path, response)
    
    elif choice == '2':
        response = call_depth_api(image_path)
        if response:
            print("\nDepth estimation completed successfully")
            display_results(image_path, response)
    
    elif choice == '3':
        confidence = input("Enter confidence threshold (0.1-1.0, default 0.45): ") or 0.45
        response = call_analyze_api(image_path, float(confidence))
        if response:
            print(f"\nDetected {response.get('objects_detected', 0)} objects")
            print("Depth estimation completed successfully")
            display_results(image_path, response)
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 