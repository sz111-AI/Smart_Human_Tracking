import cv2
import os

SAVE_DIR = "saved_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_face(frame, box, timestamp, confidence, expand_ratio=0.8):
    """
    Save an expanded region around the detected face for better identification.

    Args:
        frame (np.array): The video frame.
        box (list): [x1, y1, x2, y2] bounding box of the detected face.
        timestamp (float): Time when the face was detected.
        confidence (float): Detection confidence score.
        expand_ratio (float): How much to expand for upper-body region.

    Returns:
        str: Path of the saved image.
    """
    x1, y1, x2, y2 = map(int, box)

    # Expand the bounding box to include upper body
    h, w, _ = frame.shape
    expand_h = int((y2 - y1) * expand_ratio)

    y1_exp = max(0, y1 - expand_h)  # Expand upwards
    x1_exp = max(0, x1 - expand_h // 2)
    x2_exp = min(w, x2 + expand_h // 2)

    # Crop the expanded region
    upper_body_crop = frame[y1_exp:y2, x1_exp:x2_exp]

    # Save the image
    save_path = os.path.join(SAVE_DIR, f"upper_body_{timestamp:.0f}_{confidence:.2f}.jpg")
    cv2.imwrite(save_path, upper_body_crop)
    
    return save_path
