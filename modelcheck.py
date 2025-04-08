import onnxruntime as ort
# model_path = "checkpoints/scrfd_10g_bnkps.onnx"
# session = ort.InferenceSession(model_path)
# print("Input details:", session.get_inputs()[0])
# print("Output details:", session.get_outputs())




# import onnxruntime as ort
# model = ort.InferenceSession("checkpoints/scrfd_10g_bnkps.onnx")
# print("Input shape:", model.get_inputs()[0].shape)
# print("Output shapes:", [output.shape for output in model.get_outputs()])




# import cv2
# from src.Human_Tracking.models.scrfd_face import SCRFDFaceDetector 
# detector = SCRFDFaceDetector()
# test_img = cv2.imread("/mnt/aa8c671d-b865-4400-8afd-a85038c29903/HTracking/2185_171_457.png")  # Use any test image
# if test_img is not None:
#     detections = detector.detect(test_img)
#     print("Detections:", detections)
# else:
#     print("Failed to load test image")



# test_detection.py
import cv2
from src.Human_Tracking.models.scrfd_face import SCRFDFaceDetector

detector = SCRFDFaceDetector()
test_img = cv2.imread("/mnt/aa8c671d-b865-4400-8afd-a85038c29903/HTracking/face_1743569678_71.png")  # Use a clear face image
detections = detector.detect(test_img)

print("Detections:", detections)
for det in detections:
    x1, y1, x2, y2, conf = det
    cv2.rectangle(test_img, (x1,y1), (x2,y2), (0,255,0), 2)
    
cv2.imshow("Test", test_img)
cv2.waitKey(0)