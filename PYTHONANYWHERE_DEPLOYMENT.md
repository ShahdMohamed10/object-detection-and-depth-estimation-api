# Deploying to PythonAnywhere

This document provides step-by-step instructions for deploying the Object Detection and Depth Estimation API to PythonAnywhere.

## Deployment Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ShahdMohamed10/object-detection-and-depth-estimation-api.git
cd object-detection-and-depth-estimation-api
```

### 2. Set Up the Environment

Due to PythonAnywhere space constraints, install packages in stages:

```bash
# Install basic requirements first
pip install flask pillow numpy opencv-python-headless

# Try to install PyTorch (might exceed quota)
pip install torch torchvision --no-cache-dir

# Try to install Ultralytics (might exceed quota)
pip install ultralytics
```

If you encounter disk quota errors, consider using PythonAnywhere's system packages or upgrading to a paid account.

### 3. Configure the Web App

1. In PythonAnywhere, go to the "Web" tab
2. Click "Add a new web app"
3. Choose your domain (e.g., yourusername.pythonanywhere.com)
4. Select "Manual Configuration" and Python 3.8+
5. In the "Code" section, set:
   - Source code: `/home/yourusername/object-detection-and-depth-estimation-api`
   - Working directory: `/home/yourusername/object-detection-and-depth-estimation-api`

### 4. Configure the WSGI File

Edit the WSGI configuration file:

```python
import sys
import os

# Add your project directory to the sys.path
path = '/home/yourusername/object-detection-and-depth-estimation-api'
if path not in sys.path:
    sys.path.append(path)

# Import your Flask application
from flask_app import application
```

### 5. Testing Your Deployment

Test if your API is working by visiting:

```
https://yourusername.pythonanywhere.com/health
```

## Fallback Strategy

If you encounter disk quota issues with model dependencies, consider these options:

1. Start with a minimal deployment that only checks health but doesn't load models
2. Upgrade to a paid PythonAnywhere account with more storage
3. Use a different deployment platform with more resources

## Testing API Endpoints

Once deployed, you can test the API with:

```python
import requests

# Test object detection
response = requests.post(
    "https://yourusername.pythonanywhere.com/detect",
    files={"image": open("test_image.jpg", "rb")}
)
print(response.json())
``` 