import numpy as np
import insightface
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cdist

from scipy.spatial.distance import cdist
import numpy as np

from scipy.spatial.distance import cdist
import numpy as np

    
class ArcFaceRecognizer:
    def __init__(self, model_name="buffalo_l", use_gpu=False):
        """
        Initialize ArcFace model.
        
        Args:
            model_name (str): Name of the model to load.
            use_gpu (bool): Whether to use GPU acceleration (requires CUDA).
        """
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1)  # Use GPU if available

    def get_embeddings(self, face_images):
        # Check if face_images is empty
        if len(face_images) == 0:
            return None  # Return None if no faces are provided

        embeddings = self.app.get(np.array(face_images))  # Batch process all images
        return [face.embedding for face in embeddings if face is not None]


    from scipy.spatial.distance import cdist
    import numpy as np

    def is_duplicate(self, new_embedding, stored_embeddings, threshold=0.5):
        """
        Check if the new embedding is similar to stored embeddings using cosine similarity.
        
        Args:
            new_embedding (np.array): Embedding of the new face.
            stored_embeddings (list of np.array): Stored face embeddings.
            threshold (float): Similarity threshold for duplicates.
        
        Returns:
            bool: True if duplicate, False otherwise.
        """
        if not stored_embeddings:
            return False  # No stored embeddings, not a duplicate

        # Ensure that stored_embeddings is a 2D array
        stored_embeddings = np.array(stored_embeddings)
        if stored_embeddings.ndim == 1:
            stored_embeddings = np.expand_dims(stored_embeddings, axis=0)
        
        # Convert new_embedding to numpy array if it is not already
        new_embedding = np.array(new_embedding)
        
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


