import numpy as np
from collections import deque
from typing import Dict, List, Optional, Any

class TemporalBuffer:
    """
    Maintains a sliding window of recent tracking data to model Identity over time.
    Calculates spatial and temporal stability to reject spoofing, occlusion, and jitter.
    """
    def __init__(self, max_frames: int = 15, lambda_param: float = 100.0, sim_threshold: float = 0.7):
        self.max_frames = max_frames
        self.lambda_param = lambda_param
        self.sim_threshold = sim_threshold
        
        # Buffer structure: track_id -> deque(maxlen=K)
        # deque automatically removes oldest elements when maxlen is exceeded
        self.buffers: Dict[int, deque] = {}

    def add_frame(self, track_id: int, mesh: np.ndarray, hash_dict: dict, similarity: Optional[float] = None) -> None:
        """
        Stores the normalized 3D mesh, the generated region hashes, and similarity metric.
        Automatically handles eviction of items older than max_frames.
        """
        if track_id not in self.buffers:
            self.buffers[track_id] = deque(maxlen=self.max_frames)
            
        frame_data = {
            'mesh': mesh,
            'hash': hash_dict,
            'similarity': similarity if similarity is not None else 0.0
        }
        self.buffers[track_id].append(frame_data)

    def get_buffer(self, track_id: int) -> List[Dict[str, Any]]:
        """Retrieve the sliding window buffer for a specific track ID."""
        if track_id in self.buffers:
            return list(self.buffers[track_id])
        return []

    def compute_temporal_variance(self, track_id: int) -> float:
        """
        Computes the statistical variance of the similarity scores over the sliding window.
        Returns 0.0 if not enough data is available.
        """
        if track_id not in self.buffers or len(self.buffers[track_id]) < 2:
            return 0.0
            
        similarities = [frame['similarity'] for frame in self.buffers[track_id]]
        return float(np.var(similarities))

    def compute_landmark_stability(self, track_id: int) -> np.ndarray:
        """
        Computes stability metric for each of the 468 landmarks: s_i = exp(-lambda * Var(p_i)).
        Returns a numpy array of shape (468,) with values bounded between [0, 1].
        """
        if track_id not in self.buffers or len(self.buffers[track_id]) < 2:
            # Maximum stability assumed if there is no history to prove otherwise
            return np.ones(468, dtype=np.float32)
            
        # Extract purely the spatial 3D coordinates.
        # Shape: (num_frames, 468, 4) -> slice to (num_frames, 468, 3)
        meshes = np.stack([f['mesh'][:, :3] for f in self.buffers[track_id]])
        
        # Calculate variance along the temporal axis (axis=0)
        # Result shape: (468, 3)
        coordinate_variances = np.var(meshes, axis=0)
        
        # Sum variances across the 3 dimensions (X, Y, Z) to get scalar variance per landmark
        # Result shape: (468,)
        point_variances = np.sum(coordinate_variances, axis=1)
        
        # Apply exponential decay formulation
        # s_i approaches 0 for highly unstable landmarks, and 1 for perfectly static ones
        stability = np.exp(-self.lambda_param * point_variances)
        return stability.astype(np.float32)

    def compute_consistency(self, track_id: int) -> int:
        """
        Calculates C_k: Count of frames in the active buffer where the similarity breached threshold.
        """
        if track_id not in self.buffers:
            return 0
            
        consistency_score = sum(
            1 for frame in self.buffers[track_id] 
            if frame['similarity'] >= self.sim_threshold
        )
        return consistency_score
