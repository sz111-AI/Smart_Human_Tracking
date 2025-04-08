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

def is_high_quality(frame, box, min_score):
    """Check if face meets quality threshold"""
    quality_score = calculate_quality_score(frame, box)
    return quality_score >= min_score


#***************************************scrafd

# import cv2
# import numpy as np

# def calculate_quality_score(face_image, box=None):
#     """Calculate quality score with debug info"""
#     if box is not None:
#         x1, y1, x2, y2 = map(int, box)
#         face_image = face_image[y1:y2, x1:x2]
    
#     if face_image.size == 0 or face_image.shape[0] < 10 or face_image.shape[1] < 10:
#         print("⚠️ Empty or too small face image")
#         return 0
        
#     try:
#         gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#     except:
#         print("⚠️ Failed to convert to grayscale")
#         return 0
    
#     # Sharpness
#     sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#     sharpness_score = np.clip(sharpness / 300, 0, 1)
    
#     # Brightness
#     brightness = np.mean(gray)
#     brightness_score = 1 - abs(brightness - 127) / 127
    
#     # Contrast
#     contrast_score = np.std(gray) / 100
    
#     quality_score = (sharpness_score * 0.5 + 
#                    brightness_score * 0.3 + 
#                    contrast_score * 0.2)
    
#     print(f"🔧 Quality components - Sharp: {sharpness_score:.2f}, Bright: {brightness_score:.2f}, Contrast: {contrast_score:.2f}")
    
#     return np.clip(quality_score, 0, 1)