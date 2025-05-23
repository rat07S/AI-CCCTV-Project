import os
import cv2
import time
from .accident_predictor import detect_accidents
from .speed_estimator import estimate_vehicle_speeds
from accident_utils.send_mail import send_email_alert  # Make sure this module exists and is configured

PROCESSED_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed"))

# Global speed log for charting
speed_log = []

def process_video(video_path):
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError("Video file not found")

        video_filename = os.path.basename(video_path)
        output_path = os.path.join(PROCESSED_FOLDER, f"processed_{video_filename}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0 or height == 0:
            raise ValueError("Invalid video dimensions")

        # Try different codecs if default fails
        codecs = [
            ('H264', 'H264'),
            ('MP4V', 'mp4v'),
            ('XVID', 'XVID'),
            ('MJPG', 'MJPG')
        ]
        
        out = None
        for codec_name, codec_code in codecs:
            try:
                out = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*codec_code),
                    fps,
                    (width, height)
                )
                if out.isOpened():
                    print(f"Using {codec_name} codec")
                    break
            except Exception as e:
                print(f"Codec {codec_name} failed: {str(e)}")
                continue
        
        if out is None or not out.isOpened():
            raise ValueError("Could not initialize video writer")

        vehicle_data = {}
        accident_detected = False
        accident_frame = None
        frame_idx = 0
        accident_confidence = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            timestamp = round(frame_idx / fps, 2)

            # Process frame for accident detection
            frame, frame_info, is_accident = detect_accidents(frame, timestamp)
            
            # Update accident status
            if is_accident and not accident_detected:
                accident_detected = True
                accident_frame = frame.copy()
                accident_confidence = frame_info[0]['confidence']
                send_email_alert(
                    subject="🚨 ACCIDENT DETECTED!",
                    content=f"Accident detected at timestamp: {timestamp:.2f}s\nConfidence: {accident_confidence:.2f}\nVideo: {video_filename}"
                )

            # Track vehicles
            for info in frame_info:
                vid = info['id']
                if vid not in vehicle_data:
                    vehicle_data[vid] = []
                vehicle_data[vid].append({
                    'center': info['center'],
                    'timestamp': timestamp,
                    'confidence': info['confidence']
                })

            out.write(frame)

        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Generate report
        speed_data = estimate_vehicle_speeds(vehicle_data)
        report = {
            "accident_detected": accident_detected,
            "accident_confidence": accident_confidence if accident_detected else 0,
            "accident_vehicles": list(vehicle_data.keys()),
            "timestamp": timestamp if accident_detected else None,
            "speed_data": speed_data
        }

        return output_path, report

    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

def get_speed_data():
    """Return speed data for chart rendering"""
    try:
        # Create sample data structure for speed chart
        data = {
            "timestamps": [],
            "objects": []
        }
        
        # If we have logged speed data, format it for the chart
        for entry in speed_log:
            data["timestamps"].append(entry["timestamp"])
            for vehicle in entry["vehicles"]:
                # Find or create vehicle entry
                vehicle_data = next(
                    (obj for obj in data["objects"] if obj["id"] == vehicle["id"]),
                    None
                )
                
                if vehicle_data is None:
                    vehicle_data = {
                        "id": vehicle["id"],
                        "speeds": [None] * len(data["timestamps"])
                    }
                    data["objects"].append(vehicle_data)
                
                # Add speed data point
                vehicle_data["speeds"][-1] = vehicle["speed"]
        
        return data
        
    except Exception as e:
        print(f"Error getting speed data: {str(e)}")
        return {"timestamps": [], "objects": []}
