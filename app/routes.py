from flask import render_template, Response, jsonify
from app import app
from app.camera import Camera

camera = Camera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream')
def start_stream():
    return camera.start_stream()

@app.route('/stop_stream')
def stop_stream():
    return camera.stop_stream()