import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist
import cv2

class ArcFaceRecognizer:
    def __init__(self, model_name="buffalo_l", use_gpu=True):
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(
            name=model_name, 
            providers=providers,
            allowed_modules=['detection', 'recognition']  # Both modules required
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1)
        self.required_size = (112, 112)

    def get_embeddings(self, face_images):
        """
        Get face embeddings from preprocessed face images
        Args:
            face_images: Single image or list of images (RGB, 112x112, normalized)
        Returns:
            List of embeddings or None if extraction fails
        """
        if not isinstance(face_images, (list, np.ndarray)):
            face_images = [face_images]
        
        # Convert to numpy array if needed
        if isinstance(face_images, list):
            face_images = np.array(face_images)
        
        if face_images.size == 0:
            return None
            
        try:
            # Get embeddings - input must be BGR format
            if face_images.shape[-1] == 3:  # Assume RGB if 3 channels
                face_images = face_images[..., ::-1]  # Convert RGB to BGR
            
            faces = self.app.get(face_images)
            return [face.embedding for face in faces if hasattr(face, 'embedding')]
        except Exception as e:
            print(f"Embedding extraction failed: {str(e)}")
            return None

    def is_duplicate(self, new_embedding, stored_embeddings, threshold=0.5):
        """Check if embedding matches any stored embeddings"""
        if not stored_embeddings or new_embedding is None or len(new_embedding) == 0:
            return False
            
        new_embedding = np.array(new_embedding).reshape(1, -1)
        stored_embeddings = np.array(stored_embeddings)
        
        if stored_embeddings.ndim == 1:
            stored_embeddings = stored_embeddings.reshape(1, -1)
            
        similarity = 1 - cdist(new_embedding, stored_embeddings, 'cosine')
        return np.any(similarity > threshold)