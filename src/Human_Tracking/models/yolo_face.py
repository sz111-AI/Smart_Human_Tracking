import torch
from ultralytics import YOLO

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

class FaceDetector:
    def __init__(self, model_path="checkpoints/face_yolov8n.pt"):
        self.model = YOLO(model_path).to(device)

    def detect(self, frame):
        results = self.model(frame, device=device, imgsz=640, batch=4)
        return results
