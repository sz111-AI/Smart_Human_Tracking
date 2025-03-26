import cv2
import time
# from src.Human_Tracking.models.yolo_human import HumanDetector
from src.Human_Tracking.models.yolo_face import FaceDetector
from src.Human_Tracking.utils.video_processing import resize_frame
from src.Human_Tracking.utils.visualization import draw_boxes
from src.Human_Tracking.utils.face_saving import save_face
from src.Human_Tracking.utils.image_quality import is_high_quality

def draw_boxes_with_confidence(frame, results, color, label):
    """
    Draw bounding boxes with confidence scores on the frame.
    """
    for result in results:
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().numpy(), 
            result.boxes.conf.cpu().numpy(), 
            result.boxes.cls.cpu().numpy()
        ):
            # Only process humans for "Human" label
            if label == "Human" and class_id != 0:
                continue
                
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1+5, y1+5), (x2+5, y2+5), color, 1)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def process_video(camera_ip):
    cap = cv2.VideoCapture(camera_ip)
    # human_detector = HumanDetector()
    face_detector = FaceDetector()

    previous_faces = {}  # Store face timestamps to avoid frequent saving

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame, scale_percent=100)  # Resize for faster processing

        # Detect Humans
        # results_human = human_detector.detect(frame)
        # frame = draw_boxes_with_confidence(frame, results_human, color=(0, 255, 0), label="Human")

        # Detect Faces
        results_face = face_detector.detect(frame)
        frame = draw_boxes_with_confidence(frame, results_face, color=(0, 0, 255), label="Face")

        # Save Best Face
        # Save best expanded face region
        timestamp = time.time()
        for result in results_face:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                face_key = tuple(map(int, box))  # Unique face identifier
                
                if face_key not in previous_faces or timestamp - previous_faces[face_key] > 2:
                    previous_faces[face_key] = timestamp
                    
                    # Expand bounding box around face
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Expand the bounding box to cover the upper body (30% larger)
                    expansion_factor = 0.8  # 80% expansion
                    box_width = x2 - x1
                    box_height = y2 - y1
                    expand_x = int(box_width * expansion_factor)
                    expand_y = int(box_height * expansion_factor)

                    # Update coordinates to expand box
                    x1 = max(0, x1 - expand_x)
                    y1 = max(0, y1 - expand_y)
                    x2 = min(frame.shape[1], x2 + expand_x)
                    y2 = min(frame.shape[0], y2 + expand_y)

                    expanded_box = (x1, y1, x2, y2)  # New expanded bounding box

                    saved_path = save_face(frame, expanded_box, timestamp, confidence)
                    
                    if saved_path:
                        print(f"✅ Saved upper-body image for identification: {saved_path}")

        # Show detection results
        cv2.imshow("Human & Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


###*******************************************************************

# import cv2
# import time
# # from src.Human_Tracking.models.yolo_human import HumanDetector
# from src.Human_Tracking.models.yolo_face import FaceDetector
# from src.Human_Tracking.utils.video_processing import resize_frame
# from src.Human_Tracking.utils.visualization import draw_boxes
# from src.Human_Tracking.utils.face_saving import FaceSaver
# from src.Human_Tracking.utils.image_quality import is_high_quality

# from src.Human_Tracking.utils.face_tracking import FaceTracker

# def draw_boxes_with_confidence(frame, results, color, label):
#     """
#     Draw bounding boxes with confidence scores on the frame.
#     """
#     for result in results:
#         for box, confidence, class_id in zip(
#             result.boxes.xyxy.cpu().numpy(), 
#             result.boxes.conf.cpu().numpy(), 
#             result.boxes.cls.cpu().numpy()
#         ):
#             # Only process humans for "Human" label
#             if label == "Human" and class_id != 0:
#                 continue
                
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1+5, y1+5), (x2+5, y2+5), color, 1)
#             text = f"{label} {confidence:.2f}"
#             cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
#     return frame

# def process_video(camera_ip):
#     cap = cv2.VideoCapture(camera_ip)
#     # human_detector = HumanDetector()
#     face_detector = FaceDetector()
#     face_tracker = FaceTracker() 

#     previous_faces = {}  # Store face timestamps to avoid frequent saving

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = resize_frame(frame, scale_percent=100)  # Resize for faster processing
#         results_face = face_detector.detect(frame)

#         # Convert detections to tracker format
#         detections = []
#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), 
#                                      result.boxes.conf.cpu().numpy()):
#                 detections.append({
#                     'box': box,
#                     'confidence': confidence,
#                     'class_id': 0  # Assuming 0 is face class
#                 })
        
#         # Update tracker
#         tracks = face_tracker.update(detections, frame)
        
#         # Process tracked faces
#         for track in tracks:
#             track_id = track['track_id']
#             box = track['box']
#             confidence = track['confidence']

#         # Detect Humans
#         # results_human = human_detector.detect(frame)
#         # frame = draw_boxes_with_confidence(frame, results_human, color=(0, 255, 0), label="Human")

#         # Detect Faces
#         results_face = face_detector.detect(frame)
#         frame = draw_boxes_with_confidence(frame, results_face, color=(0, 0, 255), label="Face")

#         # Save Best Face
#         # Save best expanded face region
#         timestamp = time.time()
#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
#                 face_key = tuple(map(int, box))  # Unique face identifier
                
#                 if face_key not in previous_faces or timestamp - previous_faces[face_key] > 2:
#                     previous_faces[face_key] = timestamp
                    
#                     # Expand bounding box around face
#                     x1, y1, x2, y2 = map(int, box)
                    
#                     # Expand the bounding box to cover the upper body (30% larger)
#                     expansion_factor = 0.8  # 80% expansion
#                     box_width = x2 - x1
#                     box_height = y2 - y1
#                     expand_x = int(box_width * expansion_factor)
#                     expand_y = int(box_height * expansion_factor)

#                     # Update coordinates to expand box
#                     x1 = max(0, x1 - expand_x)
#                     y1 = max(0, y1 - expand_y)
#                     x2 = min(frame.shape[1], x2 + expand_x)
#                     y2 = min(frame.shape[0], y2 + expand_y)

#                     expanded_box = (x1, y1, x2, y2)  # New expanded bounding box

#                     saved_path = FaceSaver(frame, expanded_box, timestamp, confidence)
                    
#                     if saved_path:
#                         print(f"✅ Saved upper-body image for identification: {saved_path}")

#         # Show detection results
#         cv2.imshow("Human & Face Detection", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

###*******************************************************************



# import cv2
# import time
# from collections import defaultdict
# from src.Human_Tracking.models.yolo_face import FaceDetector
# from src.Human_Tracking.utils.video_processing import resize_frame
# from src.Human_Tracking.utils.visualization import draw_boxes, draw_boxes_with_confidence
# from src.Human_Tracking.utils.face_saving import FaceSaver
# from src.Human_Tracking.utils.image_quality import calculate_quality_score
# from src.Human_Tracking.utils.face_tracking import FaceTracker

# class FaceTracker:
#     def __init__(self, max_age=30):
#         self.tracks = {}
#         self.next_id = 1
#         self.max_age = max_age
        
#     def update(self, detections):
#         # Age all tracks
#         for track_id in list(self.tracks.keys()):
#             self.tracks[track_id]['age'] += 1
#             if self.tracks[track_id]['age'] > self.max_age:
#                 del self.tracks[track_id]
        
#         # Match detections to existing tracks
#         matched_pairs = []
#         unmatched_detections = list(range(len(detections)))
#         unmatched_tracks = list(self.tracks.keys())
        
#         if self.tracks and detections:
#             # Simple center-distance based matching
#             for det_idx, det in enumerate(detections):
#                 for track_id in list(self.tracks.keys()):
#                     det_center = self._get_center(det['box'])
#                     track_center = self._get_center(self.tracks[track_id]['box'])
#                     distance = ((det_center[0]-track_center[0])**2 + 
#                               (det_center[1]-track_center[1])**2)**0.5
                    
#                     # If close enough, consider it a match
#                     if distance < 50:  # pixels
#                         matched_pairs.append((det_idx, track_id))
#                         if det_idx in unmatched_detections:
#                             unmatched_detections.remove(det_idx)
#                         if track_id in unmatched_tracks:
#                             unmatched_tracks.remove(track_id)
        
#         # Update matched tracks
#         for det_idx, track_id in matched_pairs:
#             self.tracks[track_id]['box'] = detections[det_idx]['box']
#             self.tracks[track_id]['confidence'] = detections[det_idx]['confidence']
#             self.tracks[track_id]['age'] = 0  # Reset age
        
#         # Create new tracks for unmatched detections
#         for det_idx in unmatched_detections:
#             self.tracks[self.next_id] = {
#                 'box': detections[det_idx]['box'],
#                 'confidence': detections[det_idx]['confidence'],
#                 'age': 0
#             }
#             self.next_id += 1
        
#         return self.tracks
    
#     def _get_center(self, box):
#         return ((box[0]+box[2])/2, (box[1]+box[3])/2)

# def process_video(camera_ip):
#     cap = cv2.VideoCapture(camera_ip)
#     face_detector = FaceDetector()
#     face_tracker = FaceTracker()
#     face_saver = FaceSaver()
    
#     # For visualization
#     color_map = {}
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = resize_frame(frame, scale_percent=100)
#         results_face = face_detector.detect(frame)
        
#         # Convert detections to tracker format
#         detections = []
#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), 
#                                     result.boxes.conf.cpu().numpy()):
#                 detections.append({
#                     'box': box,
#                     'confidence': confidence
#                 })
        
#         # Update tracker
#         tracks = face_tracker.update(detections)
        
#         # Process each tracked face
#         for track_id, track in tracks.items():
#             box = track['box']
#             confidence = track['confidence']
            
#             # Assign color for visualization
#             if track_id not in color_map:
#                 color_map[track_id] = (
#                     int((track_id * 50) % 255),
#                     int((track_id * 100) % 255),
#                     int((track_id * 150) % 255)
#                 )
            
#             # Draw tracking info
#             x1, y1, x2, y2 = map(int, box)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color_map[track_id], 2)
#             cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[track_id], 2)
            
#             # Expand bounding box (your original approach)
#             expansion_factor = 0.8
#             box_width = x2 - x1
#             box_height = y2 - y1
#             expand_x = int(box_width * expansion_factor)
#             expand_y = int(box_height * expansion_factor)

#             x1 = max(0, x1 - expand_x)
#             y1 = max(0, y1 - expand_y)
#             x2 = min(frame.shape[1], x2 + expand_x)
#             y2 = min(frame.shape[0], y2 + expand_y)
#             expanded_box = (x1, y1, x2, y2)

#             # Save face with quality check
#             saved_path = face_saver.save_face(
#                 frame, 
#                 expanded_box,
#                 track_id=track_id,
#                 timestamp=time.time(),
#                 confidence=confidence
#             )
            
#             if saved_path:
#                 print(f"✅ Saved face for ID {track_id}: {saved_path}")

#         cv2.imshow("Face Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



###*******************************************************************

# import cv2
# import time
# from src.Human_Tracking.models.yolo_face import FaceDetector
# from src.Human_Tracking.utils.video_processing import resize_frame
# from src.Human_Tracking.utils.face_saving import FaceSaver
# from src.Human_Tracking.utils.face_tracking import FaceTracker

# def draw_boxes_with_confidence(frame, detections, color, label, is_tracking_output=False):
#     """
#     Modified to handle both YOLO results and tracker outputs
#     """
#     for detection in detections:
#         if is_tracking_output:
#             # Handle tracker dictionary format
#             box = detection['box']
#             confidence = detection['confidence']
#             class_id = detection.get('class_id', 0)
#         else:
#             # Handle YOLO results format
#             box = detection.boxes.xyxy.cpu().numpy()[0]
#             confidence = detection.boxes.conf.cpu().numpy()[0]
#             class_id = detection.boxes.cls.cpu().numpy()[0]
        
#         if label == "Human" and class_id != 0:
#             continue
            
#         x1, y1, x2, y2 = map(int, box)
#         cv2.rectangle(frame, (x1+5, y1+5), (x2+5, y2+5), color, 1)
#         text = f"{label} {confidence:.2f}" if not is_tracking_output else f"ID:{detection.get('id', '?')} {confidence:.2f}"
#         cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
#     return frame

# def process_video(camera_ip):
#     cap = cv2.VideoCapture(camera_ip)
#     face_detector = FaceDetector()
#     tracker = FaceTracker(max_age=30)
#     face_saver = FaceSaver()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = resize_frame(frame, scale_percent=100)
        
#         # 1. Detect faces using YOLO
#         results_face = face_detector.detect(frame)
        
#         # 2. Convert to tracker format
#         detections = []
#         for result in results_face:
#             for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), 
#                                     result.boxes.conf.cpu().numpy()):
#                 detections.append({
#                     'box': box,
#                     'confidence': confidence,
#                     'class_id': 0
#                 })

#         # 3. Update tracker
#         tracks = tracker.update(detections)

#         # 4. Draw original detections (optional)
#         # frame = draw_boxes_with_confidence(frame, results_face, (0,0,255), "Face")
        
#         # 5. Draw tracked faces
#         frame = draw_boxes_with_confidence(frame, tracks, (0,255,0), "Face", is_tracking_output=True)

#         # 6. Save best faces
#         timestamp = time.time()
#         for track in tracks:
#             box = track['box']
#             confidence = track['confidence']
#             track_id = track.get('id', 0)
            
#             # Your original box expansion
#             x1, y1, x2, y2 = map(int, box)
#             expansion_factor = 0.8
#             expand_x = int((x2 - x1) * expansion_factor)
#             expand_y = int((y2 - y1) * expansion_factor)
#             expanded_box = (
#                 max(0, x1 - expand_x),
#                 max(0, y1 - expand_y),
#                 min(frame.shape[1], x2 + expand_x),
#                 min(frame.shape[0], y2 + expand_y)
#             )

#             saved_path = face_saver.save_face(
#                 frame,
#                 expanded_box,
#                 timestamp=timestamp,
#                 confidence=confidence,
#                 track_id=track_id
#             )
            
#             if saved_path:
#                 print(f"✅ Saved face for ID {track_id}: {saved_path}")

#         cv2.imshow("Face Tracking", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()