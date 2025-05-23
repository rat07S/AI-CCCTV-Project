import cv2
import torch
import numpy as np
from pathlib import Path

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.25  # Confidence threshold
model.classes = [2, 3, 5, 7]  # Only detect cars, motorcycles, bus, trucks

def detect_accidents(frame, timestamp):
    """
    Detect potential accidents by analyzing vehicle interactions
    """
    # Convert frame to RGB for YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = model(frame_rgb)
    
    # Get detections
    vehicles = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, conf, class
    
    # Initialize variables
    is_accident = False
    accident_confidence = 0
    frame_info = []
    
    # Check for vehicle interactions/collisions
    for i, v1 in enumerate(vehicles):
        box1 = v1[:4]
        vid = i + 1
        
        # Check intersection with other vehicles
        for j, v2 in enumerate(vehicles[i+1:]):
            box2 = v2[:4]
            
            # Calculate IoU (Intersection over Union)
            iou = calculate_iou(box1, box2)
            
            # If vehicles are too close or overlapping
            if iou > 0.2:  # Threshold for accident detection
                is_accident = True
                accident_confidence = max(accident_confidence, iou)
        
        # Draw bounding box
        cv2.rectangle(frame, 
                     (int(box1[0]), int(box1[1])), 
                     (int(box1[2]), int(box1[3])), 
                     (0, 0, 255) if is_accident else (0, 255, 0), 
                     2)
        
        # Calculate center point
        center = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
        
        # Add vehicle info
        frame_info.append({
            'id': vid,
            'in_accident': is_accident,
            'center': center,
            'confidence': float(v1[4])
        })
    
    # Add timestamp and status
    cv2.putText(frame, 
                f"Time: {timestamp:.2f}s {'ACCIDENT!' if is_accident else ''}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255) if is_accident else (0, 255, 0), 
                2)
    
    return frame, frame_info, is_accident

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
