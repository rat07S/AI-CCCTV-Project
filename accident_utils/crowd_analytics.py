import cv2
import torch
import numpy as np
from pathlib import Path

# Load YOLOv5 model for person detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0]  # Person class only
model.conf = 0.5     # Confidence threshold

def process_crowd_video(video_path, output_path):
    """Process video file for crowd detection"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Try different codecs for web compatibility
        codecs = [
            ('avc1', 'avc1'),  # H.264 codec
            ('mp4v', 'mp4v'),  # MP4V codec
            ('XVID', 'XVID'),  # XVID codec
            ('MJPG', 'MJPG')   # Motion JPEG
        ]

        out = None
        for codec_name, codec_code in codecs:
            try:
                # Ensure output path ends with .mp4
                output_path = str(Path(output_path).with_suffix('.mp4'))
                
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

        stats_history = []
        total_count = 0
        peak_count = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect people in frame
            results = model(frame)
            people = results.xyxy[0].cpu().numpy()
            
            # Count people
            current_count = len(people)
            total_count += current_count
            peak_count = max(peak_count, current_count)
            
            # Draw boxes around detected people
            for person in people:
                box = person[:4].astype(int)
                conf = person[4]
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (box[0], box[1]), 
                            (box[2], box[3]), 
                            (0, 255, 0), 2)
                
                # Add confidence score
                cv2.putText(frame, 
                           f'{conf:.2f}', 
                           (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, 
                           (0, 255, 0), 
                           2)

            # Add count overlay
            cv2.putText(frame,
                       f'People Count: {current_count}',
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2)

            # Store frame statistics
            avg_count = total_count / (frame_count + 1)
            stats = {
                "current_count": current_count,
                "peak_count": peak_count,
                "average_count": round(avg_count, 1),
                "density_level": get_density_level(current_count)
            }
            stats_history.append(stats)
            
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return stats_history

    except Exception as e:
        print(f"Error processing crowd video: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()
        raise

def get_density_level(count):
    if count > 50:
        return "Very High"
    elif count > 30:
        return "High"
    elif count > 15:
        return "Moderate"
    else:
        return "Normal"
