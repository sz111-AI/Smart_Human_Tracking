import cv2
import os
import numpy as np

SAVE_DIR = "data/processed/faces"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_face(frame, box, timestamp, confidence, target_width=300, target_height=350):
    """
    Save a 300x350 upper-body image centered around the detected face.

    Args:
        frame (np.array): The video frame.
        box (list): [x1, y1, x2, y2] bounding box of the detected face.
        timestamp (float): Time when the face was detected.
        confidence (float): Detection confidence score.
        target_width (int): Target width for saved image.
        target_height (int): Target height for saved image.

    Returns:
        str: Path of the saved image.
    """
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)

    # Compute face center
    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2

    # Define upper-body region centered on face
    y1_upper = max(0, face_center_y - target_height // 3)  # Shift slightly upwards
    y2_upper = min(h, y1_upper + target_height)

    x1_upper = max(0, face_center_x - target_width // 2)
    x2_upper = min(w, x1_upper + target_width)

    # Ensure fixed 300x350 size by cropping correctly
    upper_body_crop = frame[y1_upper:y2_upper, x1_upper:x2_upper]

    # Save the image
    save_path = os.path.join(SAVE_DIR, f"upper_body_{timestamp:.0f}_{confidence:.2f}.jpg")
    cv2.imwrite(save_path, upper_body_crop)

    return save_path
