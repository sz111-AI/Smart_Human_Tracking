# import cv2
# import time
# from src.Human_Tracking.models.yolo_face import FaceDetector
# from src.Human_Tracking.utils.video_processing import resize_frame
# from src.Human_Tracking.utils.visualization import draw_boxes_with_confidence
# from src.Human_Tracking.utils.face_saving import save_face
# from src.Human_Tracking.utils.image_quality import calculate_quality_score 


# def process_video(camera_ip, min_quality_score=0.68, save_interval=2, expansion_factor=0.8):
#     """
#     Process video stream to detect and save high-quality face images
    
#     Args:
#         camera_ip (str): Camera URL or device index
#         min_quality_score (float): Minimum quality threshold (0-1)
#         save_interval (int): Minimum seconds between saves for same face
#         expansion_factor (float): How much to expand bounding boxes (0-1)
#     """
#     cap = cv2.VideoCapture(camera_ip)
#     if not cap.isOpened():
#         print(f"❌ Error opening video source: {camera_ip}")
#         return

#     face_detector = FaceDetector()
#     previous_faces = {}  # Store face timestamps and quality info

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame read error or stream ended")
#             break

#         frame = resize_frame(frame, scale_percent=60)  # Resize for faster processing

#         # Detect Faces
#         results_face = face_detector.detect(frame)
#         frame = draw_boxes_with_confidence(frame, results_face, color=(0, 0, 255), label="Face")

#         # Process and save high-quality faces
#         timestamp = time.time()
#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
#                 x1, y1, x2, y2 = map(int, box)
#                 face_key = tuple(map(int, box))  # Unique face identifier
                
#                 # Check if we should process this face
#                 should_process = (
#                     face_key not in previous_faces or 
#                     timestamp - previous_faces[face_key].get('last_saved', 0) > save_interval
#                 )
                
#                 if should_process:
#                     # Calculate quality score directly
#                     quality_score = calculate_quality_score(frame, box)
                    
#                     # Update face tracking info
#                     face_info = {
#                         'last_detected': timestamp,
#                         'quality_score': quality_score,
#                         'count': previous_faces.get(face_key, {}).get('count', 0) + 1
#                     }
                    
#                     if quality_score >= min_quality_score:  # Direct quality check
#                         # Expand bounding box
#                         box_width = x2 - x1
#                         box_height = y2 - y1
#                         expand_x = int(box_width * expansion_factor)
#                         expand_y = int(box_height * expansion_factor)

#                         # Update coordinates with boundary checks
#                         x1 = max(0, x1 - expand_x)
#                         y1 = max(0, y1 - expand_y)
#                         x2 = min(frame.shape[1], x2 + expand_x)
#                         y2 = min(frame.shape[0], y2 + expand_y)

#                         expanded_box = (x1, y1, x2, y2)
                        
#                         # Save the high-quality face
#                         saved_path = save_face(frame, expanded_box, timestamp, confidence)
                        
                    #     if saved_path:
                    #         face_info['last_saved'] = timestamp
                    #         print(f"✅ Saved face (Q:{quality_score:.2f} C:{confidence:.2f}): {saved_path}")
                    #     else:
                    #         print(f"⚠️ Failed to save face (Q:{quality_score:.2f})")
                    # else:
                    #     print(f"⏭️ Skipping (Q:{quality_score:.2f} Dets:{face_info['count']})")
                    
                    # previous_faces[face_key] = face_info

#         # Show detection results
#         cv2.imshow("Face Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("🛑 Processing stopped")

##****************good

# import cv2
# import time
# from src.Human_Tracking.models.yolo_face import FaceDetector
# from src.Human_Tracking.utils.video_processing import resize_frame
# from src.Human_Tracking.utils.face_saving import save_face
# from src.Human_Tracking.utils.image_quality import calculate_quality_score

# def process_video(camera_ip, min_quality_score=0.70, quality_improvement_threshold=0.10, expansion_factor=0.8):
#     """
#     Process video stream to detect and save high-quality face images
    
#     Args:
#         camera_ip (str): Camera URL or device index
#         min_quality_score (float): Minimum quality threshold (0-1)
#         quality_improvement_threshold (float): Minimum improvement needed over previous best quality image
#         expansion_factor (float): How much to expand bounding boxes (0-1)
#     """
#     cap = cv2.VideoCapture(camera_ip)
#     if not cap.isOpened():
#         print(f"❌ Error opening video source: {camera_ip}")
#         return

#     face_detector = FaceDetector()
#     previous_faces = {}  # Stores the best quality face per unique ID

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame read error or stream ended")
#             break

#         frame = resize_frame(frame, scale_percent=80)  # Resize for faster processing

#         # Detect Faces
#         results_face = face_detector.detect(frame)

#         # Process detected faces
#         timestamp = time.time()
#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
#                 x1, y1, x2, y2 = map(int, box)
#                 face_key = tuple(map(int, box))  # Unique face identifier
                
#                 # Calculate quality score
#                 quality_score = calculate_quality_score(frame, box)

#                 # Check if this is the best quality image so far
#                 if face_key not in previous_faces:
#                     previous_faces[face_key] = {'quality_score': quality_score, 'saved_path': None}

#                 prev_quality = previous_faces[face_key]['quality_score']

#                 # Save only if quality is significantly better (improvement_threshold)
#                 if quality_score >= min_quality_score and (previous_faces[face_key]['saved_path'] is None or quality_score > previous_faces[face_key]['quality_score'] + quality_improvement_threshold):

                
#                     # Expand bounding box
#                     box_width = x2 - x1
#                     box_height = y2 - y1
#                     expand_x = int(box_width * expansion_factor)
#                     expand_y = int(box_height * expansion_factor)

#                     x1 = max(0, x1 - expand_x)
#                     y1 = max(0, y1 - expand_y)
#                     x2 = min(frame.shape[1], x2 + expand_x)
#                     y2 = min(frame.shape[0], y2 + expand_y)

#                     expanded_box = (x1, y1, x2, y2)

#                     # Save the new best-quality face
#                     saved_path = save_face(frame, expanded_box, timestamp, confidence)

#                     if saved_path:
#                         previous_faces[face_key] = {
#                             'quality_score': quality_score,  
#                             'saved_path': saved_path
#                         }
#                         print(f"✅ Saved face (Q:{quality_score:.2f} Q:{prev_quality + quality_improvement_threshold:.2f} C:{confidence:.2f}): {saved_path}")
#                     else:
#                         print(f"⚠️ Failed to save face (Q:{quality_score:.2f})")
#                 else:
#                     print(f"⏭️ Skipping face (Q:{quality_score:.2f}), not a significant improvement (prev: {prev_quality:.2f})")

#         # Show detection results
#         cv2.imshow("Face Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("🛑 Processing stopped")



## **********Implimented arcface*********
# import cv2
# import time
# import numpy as np
# import hashlib
# from src.Human_Tracking.models.yolo_face import FaceDetector
# from src.Human_Tracking.models.arcface_recognition import ArcFaceRecognizer
# from src.Human_Tracking.utils.video_processing import resize_frame
# from src.Human_Tracking.utils.face_saving import save_face
# from src.Human_Tracking.utils.image_quality import calculate_quality_score

# def generate_face_key(embedding):
#     if embedding is None:
#         print("⏭️ Skipping face: No embedding available.")
#         return None  # Avoid errors
#     try:
#         embedding_hash = hashlib.sha256(np.array(embedding).tobytes()).hexdigest()
#         return embedding_hash
#     except Exception as e:
#         print(f"⏭️ Face key generation error: {e}")
#         return None

    
# def process_video(camera_ip, min_quality=0.50, quality_threshold=0.05, expand_factor=0.8, frame_skip=15):
#     cap = cv2.VideoCapture(camera_ip)
#     if not cap.isOpened():
#         print(f"❌ Error: Cannot open video source {camera_ip}")
#         return

#     face_detector = FaceDetector()
#     recognizer = ArcFaceRecognizer()
#     stored_faces = []
#     previous_faces = {}
#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame error or stream ended")
#             break

#         frame_count += 1
#         if frame_count % frame_skip != 0:
#             continue

#         frame = resize_frame(frame, scale_percent=150)
#         results_face = face_detector.detect(frame)
#         timestamp = time.time()

#         # debug
#         results_face = face_detector.detect(frame)
#         print(f"Detected faces: {len(results_face)}")  # Add this line to log the number of detected faces


#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
#                 x1, y1, x2, y2 = map(int, box)
#                 face_crop = frame[y1:y2, x1:x2]
#                 quality_score = calculate_quality_score(frame, box)

#                 if face_crop.size == 0:
#                     print("⏭️ Invalid face crop, skipping")
#                     continue

#                 embedding = recognizer.get_embeddings(face_crop)
#                 # debug
#                 print(embedding)  # Check if it's an empty tensor

#                 if embedding is None:
#                     print("⏭️ Embedding extraction failed")
#                     continue

#                 face_key = generate_face_key(embedding)
#                 if face_key is None:
#                     print("⏭️ Face key generation failed")
#                     continue

#                 if quality_score < min_quality:
#                     print(f"⏭️ Low-quality face (Q:{quality_score:.2f}, ID:{face_key} ) skipped")
#                     continue

#                 if recognizer.is_duplicate(embedding, stored_faces):
#                     print("⏭️ Duplicate face skipped")
#                     continue

#                 if face_key not in previous_faces:
#                     previous_faces[face_key] = {'quality_score': quality_score, 'saved_path': None}

#                 prev_quality = previous_faces[face_key]['quality_score']

#                 if quality_score >= min_quality and (
#                         previous_faces[face_key]['saved_path'] is None or 
#                         quality_score > prev_quality + quality_threshold):
#                     expand_x, expand_y = int((x2 - x1) * expand_factor), int((y2 - y1) * expand_factor)
#                     x1, y1 = max(0, x1 - expand_x), max(0, y1 - expand_y)
#                     x2, y2 = min(frame.shape[1], x2 + expand_x), min(frame.shape[0], y2 + expand_y)
#                     expanded_box = (x1, y1, x2, y2)
                    
#                     saved_path = save_face(frame, expanded_box, timestamp, confidence)
#                     if saved_path:
#                         previous_faces[face_key] = {'quality_score': quality_score, 'saved_path': saved_path}
#                         stored_faces.append(embedding)
#                         print(f"✅ Face saved (Q:{quality_score:.2f},  ID:{face_key}, C:{confidence:.2f}): {saved_path}")
#                     else:
#                         print(f"⚠️ Save failed (Q:{quality_score:.2f})")
#                 else:
#                     print(f"⏭️ Face skipped (Q:{quality_score:.2f}), ID:{face_key}, minor improvement")

#         cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)  # Create a resizable window
#         cv2.resizeWindow("Face Detection", 800, 400)  # Set the window size (width=800, height=600)

#         cv2.imshow("Face Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("🛑 Processing stopped")






##****************02-ARC

import cv2
import time
import numpy as np
import hashlib
import os
from src.Human_Tracking.models.yolo_face import FaceDetector
from src.Human_Tracking.models.arcface_recognition import ArcFaceRecognizer
from src.Human_Tracking.utils.video_processing import resize_frame
from src.Human_Tracking.utils.face_saving import save_face
from src.Human_Tracking.utils.image_quality import calculate_quality_score

def generate_face_key(embedding):
    """Generate a unique key for a face embedding"""
    if embedding is None or len(embedding) == 0:
        return None
    try:
        return hashlib.sha256(np.array(embedding).tobytes()).hexdigest()
    except Exception as e:
        print(f"Face key error: {e}")
        return None

def prepare_face_for_embedding(face_img):
    """Prepare face image for ArcFace model"""
    if face_img is None or face_img.size == 0:
        return None
    
    # Convert to 3 channels if needed
    if len(face_img.shape) == 2:  # Grayscale
        face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
    elif face_img.shape[2] == 4:  # RGBA
        face_img = face_img[:, :, :3]
    elif face_img.shape[2] != 3:  # Unexpected format
        return None

    # Resize to ArcFace input size
    face_img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_CUBIC)
    return face_img.astype(np.float32) / 255.0  # Normalize to [0,1]

def process_video(
    camera_ip,
    min_quality=0.65,            # Higher quality threshold
    quality_threshold=0.1,        # Significant quality improvement needed
    expand_factor=0.3,            # Conservative expansion
    frame_skip=5,                 # Process fewer frames
    min_face_size=50,             # Minimum reasonable face size
    max_face_size=500,            # Maximum face size
    save_dir="data/processed/faces"
):
    """
    Optimized face processing pipeline that:
    - Only saves high-quality faces
    - Only updates when quality improves significantly
    - Tracks faces using ArcFace embeddings
    - No debug image saving
    """
    # Create save directory if needed
    os.makedirs(save_dir, exist_ok=True)

    # Initialize video capture
    cap = cv2.VideoCapture(camera_ip)
    if not cap.isOpened():
        print(f"Error opening video source: {camera_ip}")
        return

    # Initialize models
    face_detector = FaceDetector()
    recognizer = ArcFaceRecognizer()
    
    # Tracking variables
    stored_embeddings = []
    tracked_faces = {}  # face_key: {'best_quality', 'best_path', 'last_seen'}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Process at original resolution
        results_face = face_detector.detect(frame)
        timestamp = time.time()

        for result in results_face:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                face_width = x2 - x1
                face_height = y2 - y1

                # Skip faces that are too small or too large
                if (face_width < min_face_size or face_height < min_face_size or
                    face_width > max_face_size or face_height > max_face_size):
                    continue

                # Calculate quality score
                quality_score = calculate_quality_score(frame, box)
                if quality_score < min_quality:
                    continue

                # Expand bounding box conservatively
                expand_x = min(int(face_width * expand_factor), x1, frame.shape[1] - x2)
                expand_y = min(int(face_height * expand_factor), y1, frame.shape[0] - y2)
                x1, y1 = max(0, x1 - expand_x), max(0, y1 - expand_y)
                x2, y2 = min(frame.shape[1], x2 + expand_x), min(frame.shape[0], y2 + expand_y)

                # Extract and prepare face
                face_crop = frame[y1:y2, x1:x2]
                prepared_face = prepare_face_for_embedding(face_crop)
                if prepared_face is None:
                    continue

                # Get embedding
                embedding = recognizer.get_embeddings(prepared_face)
                if embedding is None:
                    continue

                # Generate face key
                face_key = generate_face_key(embedding)
                if face_key is None:
                    continue

                # Check if this is a new or known face
                if face_key not in tracked_faces:
                    # New face - always save first high-quality detection
                    filename = f"{save_dir}/face_{face_key[:8]}_{int(timestamp)}.jpg"
                    cv2.imwrite(filename, face_crop)
                    tracked_faces[face_key] = {
                        'best_quality': quality_score,
                        'best_path': filename,
                        'last_seen': timestamp
                    }
                    stored_embeddings.append(embedding)
                    print(f"✅ New face saved (Q:{quality_score:.2f}): {filename}")
                else:
                    # Known face - check if quality improved significantly
                    prev_quality = tracked_faces[face_key]['best_quality']
                    if quality_score > prev_quality + quality_threshold:
                        # Save new better quality face
                        filename = f"{save_dir}/face_{face_key[:8]}_{int(timestamp)}.jpg"
                        cv2.imwrite(filename, face_crop)
                        tracked_faces[face_key] = {
                            'best_quality': quality_score,
                            'best_path': filename,
                            'last_seen': timestamp
                        }
                        print(f"🔄 Better face (Q:{quality_score:.2f} vs {prev_quality:.2f}): {filename}")

        # Display results
        cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Detection", 800, 600)
        cv2.imshow("Face Detection", frame)

        # Display detection (optional)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Unique faces tracked: {len(tracked_faces)}")
    print(f"High-quality faces saved: {len([f for f in tracked_faces.values() if f['best_path'] is not None])}")