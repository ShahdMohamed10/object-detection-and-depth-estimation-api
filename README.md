# Object Detection and Depth Estimation API

A collection of APIs for computer vision tasks, providing functionality for real-time object detection and depth estimation.

## Overview

This project consists of two separate APIs:

1. **YOLO Object Detection API** - Detects objects in images using YOLOv8s
2. **MiDaS Depth Estimation API** - Estimates depth maps from images using MiDaS

Each API can be deployed and used independently.

## Features

### YOLO Object Detection API
- Real-time object detection using YOLOv8s
- Configurable confidence thresholds
- Returns both structured data (class, bounding box, confidence) and visualization
- Simple REST API interface

### MiDaS Depth Estimation API
- Depth map estimation using MiDaS small model
- Optional point-specific depth measurements
- Returns depth visualization and statistics
- Simple REST API interface

## Setup

### YOLO API

```bash
# Install requirements
pip install -r yolo_requirements.txt

# Run the API
python yolo_api.py
```

### MiDaS API

```bash
# Install requirements
pip install -r midas_requirements.txt

# Run the API
python midas_api.py
```

## Usage

Both APIs accept HTTP POST requests with image data in multipart form format and return JSON responses.

### Example Client Code

Client example scripts are provided to demonstrate usage:
- `yolo_client_example.py` for object detection
- `midas_client_example.py` for depth estimation

## Deployment

These APIs can be deployed to cloud services like Render, Heroku, Railway, or any other platform that supports Python web applications.

See `API_README.md` for detailed deployment instructions.

## License

This project is available under the MIT License.

## Note

The large model files (.pt) are not included in this repository due to size constraints. Download the YOLOv8s model directly from Ultralytics before running the APIs.

