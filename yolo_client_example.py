import requests
import json
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# The API URL (change this to your deployed API URL)
API_URL = "http://localhost:5001"

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

def display_results(image_path, api_response):
    """Display the results from the API."""
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Display original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    
    # Display result image with detections
    if 'result_image' in api_response:
        # Decode base64 result image
        result_bytes = base64.b64decode(api_response['result_image'])
        result_image = Image.open(io.BytesIO(result_bytes))
        result_image = np.array(result_image)
        
        plt.subplot(1, 2, 2)
        plt.title('Detected Objects')
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detection details
    if 'detections' in api_response:
        print(f"\nDetected {len(api_response['detections'])} objects:")
        for i, detection in enumerate(api_response['detections']):
            print(f"{i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")
            print(f"   Bounding box: {detection['bbox']}")

def check_api_health():
    """Check if the API is healthy and running."""
    endpoint = f"{API_URL}/health"
    
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            print("YOLO API is healthy!")
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
        print("Please make sure the YOLO API is running before proceeding.")
        return
    
    # Example usage
    print("\nYOLO Object Detection API Client Example")
    print("=======================================")
    
    # Ask for image path
    image_path = input("Enter path to image file: ")
    
    # Get confidence threshold
    confidence = input("Enter confidence threshold (0.1-1.0, default 0.45): ") or 0.45
    
    # Call API
    response = call_detect_api(image_path, float(confidence))
    
    if response:
        print(f"\nDetected {response.get('objects_detected', 0)} objects")
        display_results(image_path, response)

if __name__ == "__main__":
    main() 