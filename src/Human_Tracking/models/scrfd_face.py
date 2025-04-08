# src/Human_Tracking/models/scrfd.py
import cv2
import numpy as np
import onnxruntime as ort

class SCRFDFaceDetector:
    def __init__(self, model_path="checkpoints/scrfd_10g_bnkps.onnx", conf_thresh=0.5, nms_thresh=0.5):
        """
        Initialize SCRFD face detector
        
        Args:
            model_path (str): Path to ONNX model
            conf_thresh (float): Confidence threshold
            nms_thresh (float): NMS threshold
        """
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        
        # Initialize ONNX runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # Fixed input size that matches the model's expected scale
        self.input_height = 640
        self.input_width = 640
        
        print(f"Model loaded. Using fixed input size: {self.input_width}x{self.input_height}")

    def detect(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame (np.ndarray): Input image (BGR format)
            
        Returns:
            list: List of detected faces in format [x1, y1, x2, y2, confidence]
        """
        # Preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        img_resized = img_resized.transpose(2, 0, 1)  # HWC to CHW
        img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)
        img_resized /= 255.0  # Normalize to [0,1]
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img_resized})
        
        # Process outputs (3 detection layers: stride 8, 16, 32)
        detections = []
        for i in range(3):  # For each detection layer
            scores = outputs[i][:, 0]
            bboxes = outputs[i+3]
            
            # Filter by confidence and valid boxes
            for j in range(len(scores)):
                if scores[j] > self.conf_thresh:
                    x1, y1, x2, y2 = bboxes[j]
                    
                    # Validate box coordinates
                    if x2 > x1 and y2 > y1:  # Only keep valid boxes
                        detections.append([x1, y1, x2, y2, float(scores[j])])
        
        # Scale to original image size
        scale_x = frame.shape[1] / self.input_width
        scale_y = frame.shape[0] / self.input_height
        
        scaled_detections = []
        for det in detections:
            x1, y1, x2, y2, score = det
            scaled_detections.append([
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
                score
            ])
        
        # Apply NMS
        if len(scaled_detections) > 0:
            scaled_detections = self._nms(scaled_detections)
            
        return scaled_detections

    def _nms(self, detections):
        """
        Apply Non-Maximum Suppression
        
        Args:
            detections: List of detections [x1, y1, x2, y2, score]
            
        Returns:
            list: Filtered detections after NMS
        """
        if len(detections) == 0:
            return []
            
        boxes = np.array([d[:4] for d in detections])
        scores = np.array([d[4] for d in detections])
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            # Filter based on NMS threshold
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]