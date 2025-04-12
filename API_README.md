# Object Detection and Depth Estimation APIs

This project provides two separate APIs for computer vision tasks:
1. **YOLO Object Detection API** - Detects objects in images
2. **MiDaS Depth Estimation API** - Estimates depth maps from images

## Setup and Installation

Each API can be installed and run separately.

### YOLO Object Detection API

1. Install the requirements:
```bash
pip install -r yolo_requirements.txt
```

2. Run the API:
```bash
python yolo_api.py
```

The YOLO API will be available at http://localhost:5001

### MiDaS Depth Estimation API

1. Install the requirements:
```bash
pip install -r midas_requirements.txt
```

2. Run the API:
```bash
python midas_api.py
```

The MiDaS API will be available at http://localhost:5002

## API Endpoints

### YOLO Object Detection API

1. **Health Check**
   - `GET /health`
   - Returns the status of the API and model information

2. **Object Detection**
   - `POST /detect`
   - Parameters:
     - `image` (file): The image to analyze
     - `confidence` (optional, float): Confidence threshold (default: 0.45)
   - Returns:
     - JSON with detected objects, their bounding boxes, and confidence scores
     - A base64-encoded image with detection visualizations

### MiDaS Depth Estimation API

1. **Health Check**
   - `GET /health`
   - Returns the status of the API and model information

2. **Basic Depth Estimation**
   - `POST /depth`
   - Parameters:
     - `image` (file): The image to analyze
   - Returns:
     - A base64-encoded depth map visualization
     - Depth statistics (min, max, mean, median)

3. **Depth Estimation with Point Measurements**
   - `POST /depth_with_distances`
   - Parameters:
     - `image` (file): The image to analyze
     - `points` (optional, string): Comma-separated list of x,y coordinates
   - Returns:
     - A base64-encoded depth map visualization
     - Depth statistics
     - Depth measurements at the specified points

## Client Examples

Example client scripts are provided to demonstrate how to use these APIs:

- `yolo_client_example.py` - Shows how to call the YOLO object detection API
- `midas_client_example.py` - Shows how to call the MiDaS depth estimation API

To run the examples:
```bash
python yolo_client_example.py
```
or
```bash
python midas_client_example.py
```

## Deployment Options

These APIs can be deployed to various free platforms:

1. **Render**: https://render.com/
   - Free web services for small APIs
   - Easy deployment from GitHub repositories

2. **Heroku**: https://www.heroku.com/
   - Free tier with limited hours per month
   - Good for occasional or demo usage

3. **Railway**: https://railway.app/
   - Free starter tier
   - Simple deployment from GitHub

4. **PythonAnywhere**: https://www.pythonanywhere.com/
   - Free tier available
   - Good for Python applications

5. **Google Cloud Run**: https://cloud.google.com/run
   - Free tier with generous limits
   - Good for containerized applications

6. **Fly.io**: https://fly.io/
   - Free tier available
   - Global deployment options

## Deployment Instructions

1. Choose a hosting platform from the options above.
2. Create an account and set up a new project/application.
3. Install their CLI tool if applicable.
4. Configure the API port using environment variables (each platform may have different methods).
5. Deploy the API code.

For most platforms, you'll want to create a `Procfile` (for Heroku) or similar that specifies how to run the API:

```
web: gunicorn yolo_api:app  # For YOLO API
```
or
```
web: gunicorn midas_api:app  # For MiDaS API
```

The requirements files for each API already include `gunicorn` for deployment.

## Notes for Backend Developers

When integrating these APIs with your backend:

1. The input for both APIs is an image file sent as multipart/form-data.
2. The response is JSON with base64-encoded images where applicable.
3. For large-scale deployment, consider:
   - Increasing the server timeout for processing large images
   - Setting up proper error handling
   - Adding authentication to protect the APIs
   - Implementing rate limiting for public-facing endpoints 