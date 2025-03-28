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

import cv2
import time
import numpy as np
from src.Human_Tracking.models.yolo_face import FaceDetector
from src.Human_Tracking.models.arcface_recognition import ArcFaceRecognizer
from src.Human_Tracking.utils.video_processing import resize_frame
from src.Human_Tracking.utils.face_saving import save_face
from src.Human_Tracking.utils.image_quality import calculate_quality_score
from scipy.spatial.distance import cdist

def process_video(camera_ip, min_quality_score=0.5, quality_improvement_threshold=0.02, expansion_factor=0.8):
    cap = cv2.VideoCapture(camera_ip)
    if not cap.isOpened():
        print(f"❌ Error opening video source: {camera_ip}")
        return

    face_detector = FaceDetector()
    recognizer = ArcFaceRecognizer()
    stored_faces = []  # Store recognized face embeddings
    previous_faces = {}  # Track best quality per face ID
    
    frame_count = 0
    frame_skip = 3  # Process every 3rd frame (adjust as needed)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read error or stream ended")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frame processing

        frame = resize_frame(frame, scale_percent=60)

        # Detect Faces
        results_face = face_detector.detect(frame)
        timestamp = time.time()

        for result in results_face:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                face_crop = frame[y1:y2, x1:x2]

                # Calculate quality score
                quality_score = calculate_quality_score(frame, box)

                # Check if the face quality is below the threshold
                if quality_score < min_quality_score:
                    continue  # Skip low-quality faces

                # Extract ArcFace embedding
                
                embedding = recognizer.get_embeddings(face_crop)
                if embedding is None:
                    continue  # Skip if embedding extraction fails

                # Check if face is a duplicate
                if recognizer.is_duplicate(embedding, stored_faces):
                    print("⏭️ Skipping duplicate face")
                    continue

                # Ensure high-quality image
                face_key = tuple(map(int, box))
                if face_key not in previous_faces:
                    previous_faces[face_key] = {'quality_score': quality_score, 'saved_path': None}

                prev_quality = previous_faces[face_key]['quality_score']

                if quality_score >= min_quality_score and (
                        previous_faces[face_key]['saved_path'] is None or 
                        quality_score > prev_quality + quality_improvement_threshold):

                    # Expand bounding box
                    expand_x = int((x2 - x1) * expansion_factor)
                    expand_y = int((y2 - y1) * expansion_factor)

                    x1 = max(0, x1 - expand_x)
                    y1 = max(0, y1 - expand_y)
                    x2 = min(frame.shape[1], x2 + expand_x)
                    y2 = min(frame.shape[0], y2 + expand_y)

                    expanded_box = (x1, y1, x2, y2)

                    # Save face
                    saved_path = save_face(frame, expanded_box)
                    if saved_path:
                        previous_faces[face_key] = {'quality_score': quality_score, 'saved_path': saved_path}
                        stored_faces.append(embedding)  # Save embedding to FAISS
                        print(f"✅ Saved high-quality face: {saved_path}")
                    else:
                        print(f"⚠️ Failed to save face")

        # Show detection results
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Processing stopped")
