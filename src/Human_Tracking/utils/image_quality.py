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


#***************************************

# import cv2
# import numpy as np

# def calculate_quality_score(face_image, box=None):
#     """
#     Improved quality score calculation with better error handling
#     """
#     try:
#         # Extract face region if box is provided
#         if box is not None:
#             x1, y1, x2, y2 = map(int, box)
#             face_image = face_image[y1:y2, x1:x2]
        
#         # Basic validation
#         if face_image.size == 0 or face_image.shape[0] < 20 or face_image.shape[1] < 20:
#             return 0.0
        
#         # Convert to grayscale if needed
#         if len(face_image.shape) == 3:
#             gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = face_image
        
#         # Calculate quality metrics with error handling
#         try:
#             # Sharpness (Laplacian variance)
#             sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
#             sharpness_score = np.clip(sharpness / 100, 0, 1)  # Lower threshold
            
#             # Brightness (avoid too dark/bright)
#             brightness = np.mean(gray)
#             brightness_score = 1 - abs(brightness - 127) / 127
            
#             # Contrast score
#             contrast = np.std(gray)
#             contrast_score = np.clip(contrast / 50, 0, 1)  # Lower threshold
            
#             # Face detection confidence
#             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#             face_score = 1.0 if len(faces) > 0 else 0.3
            
#             # Combined score with weights
#             quality_score = (
#                 0.4 * sharpness_score + 
#                 0.3 * brightness_score + 
#                 0.2 * contrast_score + 
#                 0.1 * face_score
#             )
            
#             return np.clip(quality_score, 0, 1)
            
#         except Exception as e:
#             print(f"Quality calculation error: {e}")
#             return 0.3  # Default score if metrics fail
            
#     except Exception as e:
#         print(f"Face quality assessment failed: {e}")
#         return 0.0