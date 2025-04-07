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
#     """Generate a unique key for a face embedding with enhanced error handling"""
#     if embedding is None or len(embedding) == 0:
#         print("⏭️ Skipping face: Empty or invalid embedding")
#         return None
        
#     try:
#         embedding_array = np.array(embedding, dtype=np.float32).flatten()
#         if embedding_array.size == 0:
#             print("⏭️ Skipping face: Empty embedding array")
#             return None
            
#         embedding_hash = hashlib.sha256(embedding_array.tobytes()).hexdigest()
#         return embedding_hash
#     except Exception as e:
#         print(f"⏭️ Face key generation error: {str(e)}")
#         return None

# def validate_and_preprocess_face(face_crop, target_size=(112, 112)):
#     """
#     Validate and preprocess face crop for embedding extraction
#     Returns: Preprocessed face or None if invalid
#     """
#     if face_crop.size == 0:
#         print("⏭️ Empty face crop")
#         return None
        
#     # Ensure 3 channels (convert grayscale to RGB if needed)
#     if len(face_crop.shape) == 2:
#         face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
#     elif face_crop.shape[2] == 4:
#         face_crop = face_crop[:, :, :112]
        
#     # Resize to target size expected by the model
#     try:
#         resized = cv2.resize(face_crop, target_size)
#         return resized
#     except Exception as e:
#         print(f"⏭️ Face resize error: {str(e)}")
#         return None

# def process_video(camera_ip, min_quality=0.30, quality_threshold=0.02, expand_factor=0.3, frame_skip=20):
#     """Enhanced video processing pipeline with fixes for embedding extraction"""
    
#     cap = cv2.VideoCapture(camera_ip)
#     cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    
#     if not cap.isOpened():
#         print(f"❌ Error: Cannot open video source {camera_ip}")
#         return

#     face_detector = FaceDetector()
#     recognizer = ArcFaceRecognizer()
#     stored_faces = []
#     previous_faces = {}
#     frame_count = 0
#     fps_counter = 0
#     fps_timer = time.time()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("⚠️ Frame error or stream ended")
#             break

#         frame_count += 1
#         fps_counter += 1
        
#         if time.time() - fps_timer >= 1.0:
#             print(f"📊 FPS: {fps_counter}")
#             fps_counter = 0
#             fps_timer = time.time()

#         if frame_count % frame_skip != 0:
#             continue

#         try:
#             frame = resize_frame(frame, scale_percent=200)
            
#             results_face = face_detector.detect(frame)
#             timestamp = time.time()
            
#             print(f"\nFrame {frame_count}: Detected {len(results_face)} faces")
            
#             for result in results_face:
#                 for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
#                     x1, y1, x2, y2 = map(int, box)
#                     print(f"🔍 Processing face at ({x1},{y1})-({x2},{y2}) with confidence {confidence:.2f}")
                    
#                     # Validate bounding box
#                     if x1 >= x2 or y1 >= y2:
#                         print(f"⏭️ Invalid bounding box: ({x1},{y1})-({x2},{y2})")
#                         continue
                        
#                     face_crop = frame[y1:y2, x1:x2]
                    
#                     # Skip if face crop is too small
#                     if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
#                         print(f"⏭️ Face crop too small (size: {face_crop.shape})")
#                         continue
                        
#                     # Preprocess face for embedding extraction
#                     preprocessed_face = validate_and_preprocess_face(face_crop)
#                     if preprocessed_face is None:
#                         print("⏭️ Face preprocessing failed")
#                         continue
                        
#                     quality_score = calculate_quality_score(frame, box)
#                     print(f"📊 Quality score: {quality_score:.2f}")
                    
#                     if quality_score < min_quality:
#                         print(f"⏭️ Low-quality face (Q:{quality_score:.2f}) skipped")
#                         continue
                        
#                     # Get embedding
#                     embedding = recognizer.get_embeddings(preprocessed_face)
#                     if embedding is None or len(embedding) == 0:
#                         print("⏭️ Embedding extraction failed")
#                         continue
                        
#                     face_key = generate_face_key(embedding)
#                     if face_key is None:
#                         print("⏭️ Face key generation failed")
#                         continue
                        
#                     if recognizer.is_duplicate(embedding, stored_faces):
#                         print(f"⏭️ Duplicate face skipped (ID: {face_key})")
#                         continue
                        
#                     if face_key not in previous_faces:
#                         previous_faces[face_key] = {
#                             'quality_score': quality_score, 
#                             'saved_path': None,
#                             'first_seen': timestamp
#                         }
                        
#                     prev_quality = previous_faces[face_key]['quality_score']
                    
#                     if (quality_score >= min_quality and 
#                         (previous_faces[face_key]['saved_path'] is None or 
#                          quality_score > prev_quality + quality_threshold)):
                        
#                         expand_x = int((x2 - x1) * expand_factor)
#                         expand_y = int((y2 - y1) * expand_factor)
#                         x1 = max(0, x1 - expand_x)
#                         y1 = max(0, y1 - expand_y)
#                         x2 = min(frame.shape[1], x2 + expand_x)
#                         y2 = min(frame.shape[0], y2 + expand_y)
                        
#                         expanded_box = (x1, y1, x2, y2)
#                         saved_path = save_face(frame, expanded_box, timestamp, confidence)
                        
#                         if saved_path:
#                             previous_faces[face_key] = {
#                                 'quality_score': quality_score,
#                                 'saved_path': saved_path,
#                                 'last_seen': timestamp
#                             }
#                             stored_faces.append(embedding)
#                             print(f"✅ Face saved (Q:{quality_score:.2f}, ID:{face_key}, C:{confidence:.2f}): {saved_path}")
#                         else:
#                             print(f"⚠️ Save failed (Q:{quality_score:.2f})")
#                     else:
#                         print(f"⏭️ Face skipped (Q:{quality_score:.2f}), minor improvement")
                        
#             cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)
#             cv2.resizeWindow("Face Detection", 800, 400)
#             cv2.imshow("Face Detection", frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#         except Exception as e:
#             print(f"⚠️ Error processing frame: {str(e)}")
#             continue

#     cap.release()
#     cv2.destroyAllWindows()
    
#     print("\n📝 Processing Summary:")
#     print(f"Total frames processed: {frame_count}")
#     print(f"Unique faces detected: {len(previous_faces)}")
#     print(f"Faces saved: {len([f for f in previous_faces.values() if f['saved_path'] is not None])}")
#     print("🛑 Processing stopped")



#*************************Arcface 02

import cv2
import time
import numpy as np
import hashlib
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
        print(f"Face key generation error: {e}")
        return None

def preprocess_face_crop(face_crop, target_size=(112, 112)):
    """
    Preprocess face crop for ArcFace model
    Returns: Preprocessed face or None if invalid
    """
    if face_crop.size == 0:
        print("Face crop is empty.")
        return None

    # Convert to 3 channels if needed (handle grayscale or BGRA)
    if len(face_crop.shape) == 2:
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
    elif face_crop.shape[2] == 4:
        face_crop = face_crop[:, :, :3]
    elif face_crop.shape[2] != 3:
        print(f"Unexpected face crop channels: {face_crop.shape[2]}")
        return None

    # Print face crop shape for debugging
    print(f"Face crop shape: {face_crop.shape}")

    # Resize to target size
    try:
        # Ensure target size is valid before resizing
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError("Invalid target size for resizing")
        
        resized = cv2.resize(face_crop, target_size)
        return resized
    except Exception as e:
        print(f"Face resize error: {e}")
        return None

def process_video(camera_ip, min_quality=0.30, quality_threshold=0.02, expand_factor=0.3, frame_skip=20):
    """Video processing pipeline with fixed face preprocessing"""
    
    cap = cv2.VideoCapture(camera_ip)
    if not cap.isOpened():
        print(f"Error: Cannot open video source {camera_ip}")
        return

    face_detector = FaceDetector()
    recognizer = ArcFaceRecognizer()
    stored_faces = []
    previous_faces = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        try:
            frame = resize_frame(frame, scale_percent=200)
            results_face = face_detector.detect(frame)
            timestamp = time.time()

            for result in results_face:
                for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Skip invalid boxes
                    if x1 >= x2 or y1 >= y2:
                        continue

                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Skip empty or too small faces
                    if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
                        continue

                    # Preprocess face for ArcFace
                    preprocessed_face = preprocess_face_crop(face_crop)
                    if preprocessed_face is None:
                        continue

                    quality_score = calculate_quality_score(frame, box)
                    if quality_score < min_quality:
                        continue

                    # Get embedding
                    embedding = recognizer.get_embeddings([preprocessed_face])  # Pass as list
                    if not embedding:
                        continue

                    face_key = generate_face_key(embedding[0])
                    if not face_key:
                        continue

                    if recognizer.is_duplicate(embedding[0], stored_faces):
                        continue

                    # Save or update face
                    if face_key not in previous_faces or \
                       quality_score > previous_faces[face_key]['quality_score'] + quality_threshold:
                        
                        expand_x = int((x2 - x1) * expand_factor)
                        expand_y = int((y2 - y1) * expand_factor)
                        x1 = max(0, x1 - expand_x)
                        y1 = max(0, y1 - expand_y)
                        x2 = min(frame.shape[1], x2 + expand_x)
                        y2 = min(frame.shape[0], y2 + expand_y)
                        
                        saved_path = save_face(frame, (x1, y1, x2, y2), timestamp, confidence)
                        if saved_path:
                            previous_faces[face_key] = {
                                'quality_score': quality_score,
                                'saved_path': saved_path
                            }
                            stored_faces.append(embedding[0])

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            continue

    cap.release()
    cv2.destroyAllWindows()
