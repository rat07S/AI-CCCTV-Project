import cv2
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms

# Load YOLOv5 model with better configuration
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)  # Using larger model
model.conf = 0.3  # Lower confidence threshold for better detection
model.iou = 0.4   # Lower IOU threshold
model.classes = [0]  # Person class only
model.max_det = 100  # Increased maximum detections
model.agnostic = True  # Better handling of overlapping people

class CrowdAnalyzer:
    def __init__(self):
        self.total_counts = []
        self.peak_count = 0
        self.moving_average_window = 3
        self.min_detection_area = 500  # Reduced minimum area for detection
        self.detection_history = []
        
    def filter_detections(self, detections, frame_height):
        """Filter detections with improved criteria"""
        filtered = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            area = (x2 - x1) * (y2 - y1)
            height = y2 - y1
            
            # Accept detection if:
            # 1. Area is reasonable
            # 2. Height is proportional to frame
            # 3. Confidence is good
            if (area >= self.min_detection_area and 
                height >= frame_height * 0.05 and 
                conf >= model.conf):
                filtered.append(det)
        return filtered
        
    def analyze_frame(self, frame):
        """Analyze frame with improved detection"""
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height = frame.shape[0]
        
        # Run detection with test-time augmentation
        results = model(frame_rgb, augment=True)
        detections = results.xyxy[0].cpu().numpy()
        
        # Filter and process detections
        filtered_detections = self.filter_detections(detections, height)
        current_count = len(filtered_detections)
        
        # Update counts with smoothing
        self.total_counts.append(current_count)
        if len(self.total_counts) > self.moving_average_window:
            self.total_counts.pop(0)
        
        # Calculate smoothed count using weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.total_counts)))
        weights /= weights.sum()
        smoothed_count = int(np.average(self.total_counts, weights=weights))
        
        # Update peak count
        self.peak_count = max(self.peak_count, smoothed_count)
        
        # Calculate robust average
        if len(self.total_counts) >= 3:
            sorted_counts = sorted(self.total_counts)
            avg_count = np.mean(sorted_counts[1:-1])  # Exclude extremes
        else:
            avg_count = np.mean(self.total_counts)
        
        # Determine density level
        if smoothed_count > 40:
            density = "Very High"
            color = (0, 0, 255)
        elif smoothed_count > 25:
            density = "High"
            color = (0, 165, 255)
        elif smoothed_count > 10:
            density = "Moderate"
            color = (0, 255, 255)
        else:
            density = "Normal"
            color = (0, 255, 0)
            
        # Draw detections with enhanced visibility
        for person in filtered_detections:
            box = person[:4].astype(int)
            conf = person[4]
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (box[0], box[1]), 
                         (box[2], box[3]), 
                         color, 
                         2)
            
            # Draw confidence
            label = f"{conf:.2f}"
            cv2.putText(frame,
                       label,
                       (box[0], box[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       color,
                       2)
        
        # Add enhanced overlay
        overlay_text = f"People Count: {smoothed_count} | Density: {density}"
        cv2.putText(frame,
                   overlay_text,
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1,
                   color,
                   2)
        
        # Draw density meter
        self.draw_density_meter(frame, smoothed_count)
        
        stats = {
            "current_count": smoothed_count,
            "peak_count": self.peak_count,
            "average_count": round(avg_count, 1),
            "density_level": density,
            "raw_count": current_count
        }
        
        return frame, stats
    
    def draw_density_meter(self, frame, count):
        """Draw a visual density meter"""
        height, width = frame.shape[:2]
        meter_width = 200
        meter_height = 20
        x = width - meter_width - 20
        y = 20
        
        # Draw background
        cv2.rectangle(frame, 
                     (x, y), 
                     (x + meter_width, y + meter_height),
                     (0, 0, 0),
                     -1)
        
        # Calculate fill width
        fill_width = int((min(count, 50) / 50) * meter_width)
        
        # Draw fill
        if count <= 10:
            color = (0, 255, 0)
        elif count <= 25:
            color = (0, 255, 255)
        elif count <= 40:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)
            
        cv2.rectangle(frame,
                     (x, y),
                     (x + fill_width, y + meter_height),
                     color,
                     -1)
        
        # Draw border
        cv2.rectangle(frame,
                     (x, y),
                     (x + meter_width, y + meter_height),
                     (255, 255, 255),
                     1)

def process_crowd_video(video_path, output_path):
    """Process video with improved accuracy"""
    cap = cv2.VideoCapture(video_path)
    analyzer = CrowdAnalyzer()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer with H264 codec
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )
    
    if not out.isOpened():
        # Fallback to MP4V codec
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
    
    stats_history = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame, stats = analyzer.analyze_frame(frame)
        
        # Add frame number to stats
        stats['frame_number'] = frame_count
        stats_history.append(stats)
        
        out.write(processed_frame)
        frame_count += 1
        
        # Print progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    
    # Ensure we have some stats
    if not stats_history:
        stats_history = [{
            "current_count": 0,
            "peak_count": 0,
            "average_count": 0,
            "density_level": "Normal",
            "frame_number": 0
        }]
    
    print(f"Processing complete. Generated {len(stats_history)} stats entries")
    return stats_history
