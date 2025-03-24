import cv2
import os
print(os.getcwd())  # This will print the current working directory

from src.Human_Tracking.models.yolo_human import HumanDetector
from src.Human_Tracking.models.yolo_face import FaceDetector
from src.Human_Tracking.utils.video_processing import resize_frame
from src.Human_Tracking.utils.visualization import draw_boxes

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
            # For human detection, filter class_id to 0 (person)
            if label == "Human" and class_id != 0:  # Ensure only humans are detected
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to speed up processing
        frame = resize_frame(frame, scale_percent=50)

        # Human Detection
        results_human = human_detector.detect(frame)
        frame = draw_boxes_with_confidence(frame, results_human, color=(0, 255, 0), label="Human")

        # Face Detection
        results_face = face_detector.detect(frame)
        frame = draw_boxes_with_confidence(frame, results_face, color=(0, 0, 255), label="Face")

        # Show the frame with bounding boxes
        cv2.imshow("Human & Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
