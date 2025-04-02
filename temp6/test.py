import cv2
from src.Human_Tracking.pipeline.process_video import generate_face_key

image = cv2.imread("/mnt/aa8c671d-b865-4400-8afd-a85038c29903/HTracking/data/processed/faces/face_1743569678_71.png")
embedding = generate_face_key(image)  # Call the function you use in process_video.py

if embedding is None:
    print("❌ Embedding failed for test image")
else:
    print("✅ Embedding successful:", embedding[:5])  # Print first 5 values
