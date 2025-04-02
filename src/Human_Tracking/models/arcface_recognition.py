import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist

# class ArcFaceRecognizer:
#     def __init__(self, model_name="buffalo_l", use_gpu=False):
#         """
#         Initialize ArcFace model.
        
#         Args:
#             model_name (str): Name of the model to load.
#             use_gpu (bool): Whether to use GPU acceleration (requires CUDA).
#         """
#         providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
#         self.app = FaceAnalysis(name=model_name, providers=providers)
#         self.app.prepare(ctx_id=0 if use_gpu else -1)  # Use GPU if available

#     def get_embeddings(self, face_images):
#         # Check if face_images is empty
#         if len(face_images) == 0:
#             return None  # Return None if no faces are provided

#         embeddings = self.app.get(np.array(face_images))  # Batch process all images
#         return [face.embedding for face in embeddings if face is not None]



#     def is_duplicate(self, new_embedding, stored_embeddings, threshold=0.5):
#         """
#         Check if the new embedding is similar to stored embeddings using cosine similarity.
        
#         Args:
#             new_embedding (np.array or list): Embedding of the new face.
#             stored_embeddings (list of np.array): Stored face embeddings.
#             threshold (float): Similarity threshold for duplicates.
        
#         Returns:
#             bool: True if duplicate, False otherwise.
#         """
#         if isinstance(new_embedding, list):
#             # Convert the list to a numpy array if it's a list
#             new_embedding = np.array(new_embedding)
            
#         if new_embedding.size == 0:
#             print("Warning: Empty embedding detected!")
#             return False  # Or handle it in some other way

#         # If no stored embeddings, it's not a duplicate
#         if not stored_embeddings:
#             return False

#         # Ensure that stored_embeddings is a 2D array
#         stored_embeddings = np.array(stored_embeddings)
#         if stored_embeddings.ndim == 1:
#             stored_embeddings = np.expand_dims(stored_embeddings, axis=0)

#         # Convert new_embedding to numpy array if it is not already
#         new_embedding = np.array(new_embedding)

#         #debug
#         # Assuming new_embedding is a list
#         new_embedding = np.array(new_embedding)

#         # Now you can access the shape
#         print("Embedding Shape:", new_embedding.shape)

#         print("Embedding Shape:", new_embedding.shape)  # Log the shape of the embedding
#         print("Embedding:", new_embedding)  # Log the embedding itself


#         # Ensure new_embedding is a 2D array
#         if new_embedding.ndim == 1:
#             new_embedding = np.expand_dims(new_embedding, axis=0)
        
#         # Check if the feature dimensions match (i.e., columns)
#         if new_embedding.shape[1] != stored_embeddings.shape[1]:
#             raise ValueError(f"Embedding dimensions do not match. "
#                             f"New embedding has {new_embedding.shape[1]} features, "
#                             f"but stored embeddings have {stored_embeddings.shape[1]} features.")
        
#         # Compute cosine similarity efficiently
#         similarity = 1 - cdist(new_embedding, stored_embeddings, metric='cosine')
        
#         # If any match exceeds the threshold, it's a duplicate
#         return np.any(similarity > threshold)



class ArcFaceRecognizer:
    def __init__(self, model_name="buffalo_l", use_gpu=False):
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(
            name=model_name, 
            providers=providers,
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1)

    def get_embeddings(self, face_images):
        """
        Get embeddings for one or more face images
        Args:
            face_images: Can be:
                - Single image (numpy array)
                - List of images
                - Batch of images (numpy array)
        Returns:
            List of embeddings or single embedding
        """
        if not isinstance(face_images, (list, np.ndarray)):
            face_images = [face_images]
            
        # Convert all images to numpy arrays if needed
        face_images = [img if isinstance(img, np.ndarray) else np.array(img) for img in face_images]
        
        # Remove empty images
        face_images = [img for img in face_images if img.size > 0 and len(img.shape) == 3]
        
        if not face_images:
            print("⚠️ No valid faces provided for embedding")
            return None
            
        try:
            # Convert single image to batch format if needed
            if isinstance(face_images, list):
                face_images = np.array(face_images)
            
            # Get embeddings
            faces = self.app.get(face_images)
            
            if not faces:
                print("⚠️ No faces found in provided images")
                return None
                
            # Extract valid embeddings
            embeddings = []
            for face in faces:
                if hasattr(face, 'embedding') and face.embedding is not None:
                    embeddings.append(face.embedding)
                else:
                    print("⚠️ Face missing embedding")
            
            return embeddings[0] if len(embeddings) == 1 else embeddings
            
        except Exception as e:
            print(f"Embedding extraction error: {str(e)}")
            return None