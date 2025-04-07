import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist
import cv2

class ArcFaceRecognizer:
    def __init__(self, model_name="buffalo_l", use_gpu=False):
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(
            name=model_name, 
            providers=providers,
            allowed_modules=['detection', 'recognition']  # Ensure only needed modules are loaded
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1)

    def get_embeddings(self, face_images):
        # Convert single image to list if needed
        if not isinstance(face_images, (list, np.ndarray)):
            face_images = [face_images]
        
        # Validate and preprocess images
        valid_images = []
        for img in face_images:
            if isinstance(img, np.ndarray) and img.size > 0:
                # Resize if too small
                if img.shape[0] < 20 or img.shape[1] < 20:
                    img = cv2.resize(img, (112, 112))
                valid_images.append(img)
        
        if not valid_images:
            print("⚠️ No valid faces provided for embedding")
            return None
            
        try:
            # Convert to numpy array if it's a list
            if isinstance(valid_images, list):
                valid_images = np.array(valid_images)
            
            # Get embeddings
            faces = self.app.get(valid_images)
            
            if not faces:
                print("⚠️ No faces found in provided images")
                return None
                
            # Return embeddings
            return [face.embedding for face in faces if hasattr(face, 'embedding')]
            
        except Exception as e:
            print(f"Embedding extraction error: {str(e)}")
            return None

    def is_duplicate(self, new_embedding, stored_embeddings, threshold=0.5):
        """
        Check if the new embedding is similar to stored embeddings using cosine similarity.
        
        Args:
            new_embedding (np.array or list): Embedding of the new face.
            stored_embeddings (list of np.array): Stored face embeddings.
            threshold (float): Similarity threshold for duplicates.
        
        Returns:
            bool: True if duplicate, False otherwise.
        """
        if isinstance(new_embedding, list):
            # Convert the list to a numpy array if it's a list
            new_embedding = np.array(new_embedding)
            
        if new_embedding.size == 0:
            print("Warning: Empty embedding detected!")
            return False  # Or handle it in some other way

        # If no stored embeddings, it's not a duplicate
        if not stored_embeddings:
            return False

        # Ensure that stored_embeddings is a 2D array
        stored_embeddings = np.array(stored_embeddings)
        if stored_embeddings.ndim == 1:
            stored_embeddings = np.expand_dims(stored_embeddings, axis=0)

        # Convert new_embedding to numpy array if it is not already
        new_embedding = np.array(new_embedding)

        #debug
        # Assuming new_embedding is a list
        new_embedding = np.array(new_embedding)

        # Now you can access the shape
        print("Embedding Shape:", new_embedding.shape)

        print("Embedding Shape:", new_embedding.shape)  # Log the shape of the embedding
        print("Embedding:", new_embedding)  # Log the embedding itself


        # Ensure new_embedding is a 2D array
        if new_embedding.ndim == 1:
            new_embedding = np.expand_dims(new_embedding, axis=0)
        
        # Check if the feature dimensions match (i.e., columns)
        if new_embedding.shape[1] != stored_embeddings.shape[1]:
            raise ValueError(f"Embedding dimensions do not match. "
                            f"New embedding has {new_embedding.shape[1]} features, "
                            f"but stored embeddings have {stored_embeddings.shape[1]} features.")
        
        # Compute cosine similarity efficiently
        similarity = 1 - cdist(new_embedding, stored_embeddings, metric='cosine')
        
        # If any match exceeds the threshold, it's a duplicate
        return np.any(similarity > threshold)