import os
import traceback
import json
import numpy as np
import time
from pathlib import Path
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from collections import defaultdict

# Set your actual email credentials here before imports that need them
os.environ['ALERT_EMAIL'] = "your_actual_email@gmail.com"
os.environ['ALERT_PASSWORD'] = "your_actual_app_password"
os.environ['ALERT_RECIPIENT'] = "recipient_email@example.com"

from flask import Flask, request, jsonify, send_from_directory, abort, render_template, url_for
from werkzeug.utils import secure_filename

try:
    from flask_sock import Sock
except ImportError:
    print("Please install flask-sock: pip install flask-sock")
    raise

from accident_utils.video_processor import process_video, get_speed_data
from accident_utils.crowd_detector import process_crowd_video
from accident_utils.crowd_analytics import process_crowd_video

app = Flask(__name__, static_url_path='/static', static_folder='static')
sock = Sock(app)

# Define base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Add global video tracking
active_videos = {
    'cam1': None,
    'cam2': None,
    'cam3': None,
    'cam4': None
}

# Add these global variables for tracking statistics
accident_stats = defaultdict(int)
crowd_stats = defaultdict(int)
hourly_stats = {
    'accidents': [0] * 24,
    'crowdDensity': [0] * 24
}

def get_current_hour():
    return datetime.now().hour

# Home route - serve upload UI
@app.route('/')
def serve_index():
    return render_template('index.html')  # Renders templates/index.html

# Upload route - handles video upload and processing
@app.route('/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file part in the request"}), 400

        video = request.files['video']
        if video.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Check file extension
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return jsonify({"error": "Unsupported file format. Please upload MP4, AVI, or MOV files"}), 400

        # Save uploaded video
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        try:
            processed_path, report = process_video(video_path)
            processed_filename = os.path.basename(processed_path)
            
            return jsonify({
                "processed_video_url": f"/processed/{processed_filename}",
                "analysis_report": report
            })
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            print(traceback.format_exc())  # Print full stack trace
            return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": "Server error during upload"}), 500

# Serve processed video file
@app.route('/processed/<filename>')
def serve_processed_video(filename):
    try:
        # Set the correct MIME type for MP4 videos
        return send_from_directory(PROCESSED_FOLDER, filename, mimetype='video/mp4')
    except FileNotFoundError:
        abort(404)

# Serve speed data for chart rendering
@app.route('/speed_data')
def speed_data():
    try:
        data = get_speed_data()  # Should return dict: {timestamps, objects:[{id, speeds}]}
        return jsonify(data)
    except Exception as e:
        print(f"Speed data error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Crowd detection route
@app.route('/crowd')
def crowd_detection():
    return render_template('crowd.html')

@app.route('/analyze_crowd', methods=['POST'])
def analyze_crowd():
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file uploaded"}), 400
            
        video = request.files['video']
        if video.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        # Ensure proper filename handling
        safe_filename = secure_filename(video.filename)
        base_filename = Path(safe_filename).stem
        
        # Always use .mp4 extension for output
        video_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        output_path = os.path.join(PROCESSED_FOLDER, f"crowd_{base_filename}.mp4")
        
        # Ensure directories exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        
        # Save uploaded video
        video.save(video_path)
        
        try:
            # Process video for crowd detection
            stats_history = process_crowd_video(video_path, output_path)
            
            # Verify the output file exists and has size
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise ValueError("Failed to generate valid output video")
            
            return jsonify({
                "video_url": f"/processed/crowd_{base_filename}.mp4",
                "stats_history": stats_history
            })
            
        finally:
            # Clean up input video
            if os.path.exists(video_path):
                os.remove(video_path)
            
    except Exception as e:
        print(f"Crowd analysis error: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/register_video', methods=['POST'])
def register_video():
    try:
        data = request.get_json()
        camera_id = data.get('camera_id')
        video_url = data.get('video_url')
        
        if camera_id and video_url:
            active_videos[camera_id] = video_url
            return jsonify({"status": "success"})
        return jsonify({"error": "Invalid parameters"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_active_videos')
def get_active_videos():
    return jsonify(active_videos)

# Websocket route for real-time updates
@sock.route('/ws')
def websocket(ws):
    while True:
        # Handle real-time updates
        try:
            # Send updates about accidents
            ws.send(json.dumps({
                'type': 'accident',
                'camera': 'cam1',
                'message': 'Accident detected!'
            }))
        except Exception as e:
            break

@sock.route('/crowd_updates')
def crowd_updates(ws):
    """Send real-time crowd updates via WebSocket"""
    try:
        while True:
            # Simulate real-time updates for demo
            stats = {
                "current_count": np.random.randint(10, 40),
                "peak_count": 45,
                "average_count": 25.5,
                "density_level": "Moderate"
            }
            ws.send(json.dumps(stats))
            time.sleep(1)
    except Exception:
        pass

@sock.route('/ws')
async def websocket_endpoint():
    async with websockets.connect(this_websocket) as websocket:
        try:
            while True:
                # Send periodic statistics updates
                current_hour = get_current_hour()
                stats_data = {
                    'type': 'stats',
                    'accidents': hourly_stats['accidents'],
                    'crowdDensity': hourly_stats['crowdDensity']
                }
                await websocket.send(json.dumps(stats_data))
                
                # Handle incoming data
                data = await websocket.receive_text()
                event = json.loads(data)
                
                if event['type'] == 'accident':
                    camera_id = event['camera']
                    accident_stats[camera_id] += 1
                    hourly_stats['accidents'][current_hour] += 1
                    
                    await websocket.send(json.dumps({
                        'type': 'accident',
                        'camera': camera_id,
                        'message': f'Accident detected in {camera_id}!',
                        'count': accident_stats[camera_id]
                    }))
                
                elif event['type'] == 'crowd':
                    camera_id = event['camera']
                    density = event['density']
                    crowd_stats[camera_id] = density
                    hourly_stats['crowdDensity'][current_hour] = max(
                        hourly_stats['crowdDensity'][current_hour],
                        density
                    )
                    
                    await websocket.send(json.dumps({
                        'type': 'crowd',
                        'camera': camera_id,
                        'density': density
                    }))
                
                # Reset stats at midnight
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    accident_stats.clear()
                    crowd_stats.clear()
                    hourly_stats = {
                        'accidents': [0] * 24,
                        'crowdDensity': [0] * 24
                    }
                
                await asyncio.sleep(1)  # Prevent excessive CPU usage
                
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
