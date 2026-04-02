import numpy as np
from typing import Optional, List, Dict

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("WARNING: deepface library not installed. Tensorflow may not be available on this Python distribution.")

class BaselineEngine:
    """
    Standard Industry Baseline using Deep Convolutional Neural Networks.
    Provides ArcFace / FaceNet embeddings to compare against our 3D Mesh BioHashing pipeline.
    """
    def __init__(self, model_name: str = "ArcFace"):
        """
        Available generic deepface models: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib
        """
        self.model_name = model_name

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Pushes a raw RGB frame through the CNN baseline to extract the 512D or 128D mathematical embedding.
        Returns the embedding array, or None if the CNN pipeline crashes.
        """
        if not DEEPFACE_AVAILABLE:
            # Safe Fallback for pipeline implementation testing without TF compilation
            return np.random.rand(512).astype(np.float32)
            
        try:
            # enforce_detection=False guarantees the CNN actually attempts to run on the heavily 
            # degraded CCTV variants (blur/noise) instead of just rejecting processing immediately.
            embeddings = DeepFace.represent(
                img_path=image, 
                model_name=self.model_name, 
                enforce_detection=False,
                align=True
            )
            return np.array(embeddings[0]["embedding"], dtype=np.float32)
        except Exception as e:
            # If the baseline completely mathematically collapses during detection
            return None

    def compute_cosine_similarity(self, embed_a: np.ndarray, embed_b: np.ndarray) -> float:
        """
        Computes the standard Angular/Cosine distance margin between two CNN continuous vectors.
        Output scaled smoothly from [-1, 1] into [0, 1] for direct benchmarking comparison against Hamming scores.
        """
        if embed_a is None or embed_b is None:
            return 0.0
            
        dot = np.dot(embed_a, embed_b)
        norm_a = np.linalg.norm(embed_a)
        norm_b = np.linalg.norm(embed_b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
            
        cosine_sim = dot / (norm_a * norm_b)
        
        # Normalize into [0, 1] probability bound
        score = (cosine_sim + 1.0) / 2.0
        return float(score)

    def search_database(self, query_embedding: np.ndarray, db_embeddings: Dict[str, np.ndarray]) -> Dict:
        """
        Iterates over a dictionary of pre-computed stored embeddings.
        Returns the best matched ID and the calculated mathematical score.
        """
        best_id = None
        best_sim = -1.0
        
        if query_embedding is None:
            return {"match_id": None, "similarity": 0.0}
            
        for db_id, stored_emb in db_embeddings.items():
            sim = self.compute_cosine_similarity(query_embedding, stored_emb)
            if sim > best_sim:
                best_sim = sim
                best_id = db_id
                
        # Typical generic CNN bounding thresholds rest between 0.60 and 0.68 depending on exact model
        return {
            "match_id": best_id if best_sim > 0.68 else None,
            "similarity": best_sim
        }
