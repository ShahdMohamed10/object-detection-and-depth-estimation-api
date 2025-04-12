# Deploying to Render.com

This document provides step-by-step instructions for deploying the Object Detection and Depth Estimation API to Render.com.

## Why Render?

- More generous free tier compared to PythonAnywhere
- Better support for machine learning applications
- No hard disk quota limits on the free tier
- Direct deployment from GitHub

## Deployment Steps

### 1. Sign up for Render

Go to [Render.com](https://render.com/) and sign up for a free account.

### 2. Connect Your GitHub Repository

1. In the Render dashboard, click on "New +" and select "Web Service"
2. Connect your GitHub account if you haven't already
3. Select your repository: `object-detection-and-depth-estimation-api`

### 3. Configure the Web Service

Fill in the following details:
- **Name**: `object-detection-depth-api` (or choose your own)
- **Environment**: Python
- **Region**: Choose the one closest to your users
- **Branch**: `master` (or your main branch)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn api:app`
- **Plan**: Free

### 4. Advanced Settings (Optional)

You can click on "Advanced" to configure:
- Environment variables (if needed)
- Auto-Deploy settings
- Health Check Path: `/health`

### 5. Create Web Service

Click on "Create Web Service" to begin the deployment process.

### 6. Monitor the Deployment

Render will automatically:
1. Clone your repository
2. Install dependencies from requirements.txt
3. Start your application

The deployment might take 5-10 minutes initially because it needs to install large dependencies like PyTorch and Ultralytics.

### 7. Accessing Your API

Once deployed, your API will be available at:
```
https://object-detection-depth-api.onrender.com
```

Or whatever URL Render assigns to your service.

## Testing Your Deployment

You can test your API with:

```python
import requests

# Test health endpoint
response = requests.get("https://your-app-name.onrender.com/health")
print(response.json())

# Test object detection
response = requests.post(
    "https://your-app-name.onrender.com/detect",
    files={"image": open("test_image.jpg", "rb")}
)
print(response.json())

# Test depth estimation
response = requests.post(
    "https://your-app-name.onrender.com/depth",
    files={"image": open("test_image.jpg", "rb")}
)
print(response.json())
```

## Free Tier Limitations

While Render's free tier is more generous than PythonAnywhere, be aware of:
- Services on the free plan will spin down after 15 minutes of inactivity
- They will spin back up when a new request comes in (may take 30s on first request)
- Free tier provides 750 hours per month

## Troubleshooting

If you encounter any issues:
1. Check the logs in the Render dashboard
2. Make sure your `requirements.txt` includes all necessary dependencies
3. Verify that the start command correctly points to your Flask application 