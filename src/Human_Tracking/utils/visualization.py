# import cv2

# def draw_boxes_with_confidence(frame, results, color, label):
#     """
#     Draw bounding boxes with confidence scores on the frame.
#     """
#     for result in results:
#         for box, confidence, class_id in zip(
#             result.boxes.xyxy.cpu().numpy(), 
#             result.boxes.conf.cpu().numpy(), 
#             result.boxes.cls.cpu().numpy()
#         ):
#             # Only process humans for "Human" label
#             if label == "Human" and class_id != 0:
#                 continue
                
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 0)
#             text = f"{label} {confidence:.2f}"
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 0)
    
#     return frame

# src/Human_Tracking/utils/visualization.py
import cv2
import numpy as np

def draw_boxes(frame, detections, color=(0, 255, 0), label="Face"):
    """
    Draw bounding boxes on frame for SCRFD detections
    
    Args:
        frame: Input frame
        detections: List of detections in format [x1, y1, x2, y2, confidence]
        color: Box color
        label: Label text
        
    Returns:
        Frame with drawn boxes
    """
    for detection in detections:
        x1, y1, x2, y2, confidence = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return frame