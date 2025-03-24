import cv2
import time
from src.Human_Tracking.models.yolo_human import HumanDetector
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def process_video(camera_ip):
    cap = cv2.VideoCapture(camera_ip)
    human_detector = HumanDetector()
    face_detector = FaceDetector()

    previous_faces = {}  # Store face timestamps to avoid frequent saving

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize_frame(frame, scale_percent=50)  # Resize for faster processing

        # Detect Humans
        results_human = human_detector.detect(frame)
        frame = draw_boxes_with_confidence(frame, results_human, color=(0, 255, 0), label="Human")

        # Detect Faces
        results_face = face_detector.detect(frame)
        frame = draw_boxes_with_confidence(frame, results_face, color=(0, 0, 255), label="Face")

        # Save Best Face
        # Save the best expanded face region
        timestamp = time.time()
        for result in results_face:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                face_key = tuple(map(int, box))  # Unique face identifier
                
                if face_key not in previous_faces or timestamp - previous_faces[face_key] > 2:
                    previous_faces[face_key] = timestamp
                    saved_path = save_face(frame, box, timestamp, confidence)
                    
                    if saved_path:
                        print(f"✅ Saved high-quality image for identification: {saved_path}")


        # Show detection results
        cv2.imshow("Human & Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
