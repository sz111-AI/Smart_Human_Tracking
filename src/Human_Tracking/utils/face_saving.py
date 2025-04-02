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
import cv2
import os
import time

SAVE_DIR = "data/processed/faces"

def save_face(face_image, box, timestamp, confidence):
    """
    Save face image with quality preservation
    Args:
        face_image: Can be either:
                   - Full frame (if box is provided)
                   - Pre-cropped face image (if box is None)
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    # If box is provided, extract from frame
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        face_image = face_image[y1:y2, x1:x2]

    if face_image.size == 0:
        return None

    # Generate filename with size info
    filename = f"{SAVE_DIR}/face_{int(timestamp)}_{face_image.shape[1]}x{face_image.shape[0]}_{int(confidence*100)}.jpg"
    
    # Save with high quality JPEG compression
    cv2.imwrite(filename, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return filename
