import cv2
import os

SAVE_DIR = "data/processed/faces"

def save_face(frame, box, timestamp, confidence):
    """
    Saves the best quality face image to the designated directory.
    
    Args:
        frame (numpy array): Original frame.
        box (tuple): (x1, y1, x2, y2) coordinates of the face.
        timestamp (float): Current timestamp.
        confidence (float): Confidence score of detection.
    
    Returns:
        str: Saved image path.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    x1, y1, x2, y2 = map(int, box)
    face_crop = frame[y1:y2, x1:x2]

    # Save the image
    filename = f"{SAVE_DIR}/face_{int(timestamp)}_{int(confidence * 100)}.png"
    cv2.imwrite(filename, face_crop)

    return filename


# import cv2
# import os
# import time
# from collections import defaultdict

# class FaceSaver:
#     def __init__(self, save_dir="data/processed/faces", max_faces_per_track=5):
#         self.save_dir = save_dir
#         self.max_faces_per_track = max_faces_per_track
#         self.track_faces = defaultdict(list)
#         os.makedirs(save_dir, exist_ok=True)
        
#     def save_face(self, frame, box, track_id=None, timestamp=None, confidence=None):
#         try:
#             x1, y1, x2, y2 = map(int, box)
            
#             # Validate coordinates
#             h, w = frame.shape[:2]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w-1, x2), min(h-1, y2)
            
#             if x2 <= x1 or y2 <= y1:
#                 return None
                
#             face_crop = frame[y1:y2, x1:x2]
            
#             if face_crop.size == 0:
#                 return None
                
#             # Save with track ID if available
#             if track_id is not None:
#                 filename = f"{self.save_dir}/track_{track_id}_{int(time.time())}.png"
#             else:
#                 filename = f"{self.save_dir}/face_{int(time.time())}.png"
                
#             cv2.imwrite(filename, face_crop)
#             return filename
            
#         except Exception as e:
#             print(f"Error saving face: {str(e)}")
#             return None