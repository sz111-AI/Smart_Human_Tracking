# import cv2
# import numpy as np

# def is_high_quality(frame, box, brightness_thresh=50, sharpness_thresh=100):
#     """
#     Checks if the detected face is of high quality based on brightness and sharpness.
    
#     Args:
#         frame (numpy array): The original frame.
#         box (tuple): (x1, y1, x2, y2) coordinates of the face.
#         brightness_thresh (int): Minimum brightness level.
#         sharpness_thresh (int): Minimum sharpness level.

#     Returns:
#         bool: True if the image is high quality, False otherwise.
#     """
#     x1, y1, x2, y2 = map(int, box)
#     face_crop = frame[y1:y2, x1:x2]

#     # Convert to grayscale for analysis
#     gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

#     # Calculate brightness (mean pixel value)
#     brightness = np.mean(gray)

#     # Calculate sharpness using Laplacian variance
#     sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

#     return brightness > brightness_thresh and sharpness > sharpness_thresh


import cv2
import numpy as np

def calculate_quality_score(face_image, box=None):
    """
    Calculate quality score - can accept either cropped face or full frame + box
    """
    # If box is provided, extract face from frame first
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        face_image = face_image[y1:y2, x1:x2]
    
    if face_image.size == 0 or face_image.shape[0] < 10 or face_image.shape[1] < 10:
        return 0
        
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    
    # Sharpness (Laplacian variance)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = np.clip(sharpness / 300, 0, 1)
    
    # Brightness (avoid too dark/bright)
    brightness = np.mean(gray)
    brightness_score = 1 - abs(brightness - 127) / 127
    
    # Contrast score
    contrast_score = np.std(gray) / 100
    
    # Combined score
    quality_score = (sharpness_score * 0.5 + 
                    brightness_score * 0.3 + 
                    contrast_score * 0.2)
    
    return np.clip(quality_score, 0, 1)

def is_high_quality(frame, box, min_score=0.68):
    """Check if face meets quality threshold"""
    quality_score = calculate_quality_score(frame, box)
    return quality_score >= min_score