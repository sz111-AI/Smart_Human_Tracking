import cv2
import numpy as np

def is_high_quality(frame, box, brightness_thresh=50, sharpness_thresh=100):
    """
    Checks if the detected face is of high quality based on brightness and sharpness.
    
    Args:
        frame (numpy array): The original frame.
        box (tuple): (x1, y1, x2, y2) coordinates of the face.
        brightness_thresh (int): Minimum brightness level.
        sharpness_thresh (int): Minimum sharpness level.

    Returns:
        bool: True if the image is high quality, False otherwise.
    """
    x1, y1, x2, y2 = map(int, box)
    face_crop = frame[y1:y2, x1:x2]

    # Convert to grayscale for analysis
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # Calculate brightness (mean pixel value)
    brightness = np.mean(gray)

    # Calculate sharpness using Laplacian variance
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return brightness > brightness_thresh and sharpness > sharpness_thresh
