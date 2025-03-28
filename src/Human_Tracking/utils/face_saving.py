# import cv2
# import os

# SAVE_DIR = "data/processed/faces"

# def save_face(frame, box, timestamp, confidence):
#     """
#     Saves the best quality face image to the designated directory.
    
#     Args:
#         frame (numpy array): Original frame.
#         box (tuple): (x1, y1, x2, y2) coordinates of the face.
#         timestamp (float): Current timestamp.
#         confidence (float): Confidence score of detection.
    
#     Returns:
#         str: Saved image path.
#     """
#     os.makedirs(SAVE_DIR, exist_ok=True)

#     x1, y1, x2, y2 = map(int, box)
#     face_crop = frame[y1:y2, x1:x2]

#     # Save the image
#     filename = f"{SAVE_DIR}/face_{int(timestamp)}_{int(confidence * 100)}.png"
#     cv2.imwrite(filename, face_crop)

#     return filename

#*****************************arcface
import time
import cv2
import os

SAVE_DIR = "data/processed/faces"

def save_face(frame, box):
    """
    Saves a face image after ensuring directory exists.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    x1, y1, x2, y2 = map(int, box)
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    filename = f"{SAVE_DIR}/face_{int(time.time())}.png"
    cv2.imwrite(filename, face_crop)

    return filename


