import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

class MeshExtractor:
    """
    Extracts the 3D face mesh (468 landmarks) from images or video frames using MediaPipe.
    """
    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1, min_detection_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True, # Note: refine_landmarks=True adds iris landmarks making it 478 points. We will use the first 468.
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def extract_mesh(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 468 3D landmarks (x, y, z) and their visibilities into a numpy array.
        Returns array of shape (468, 4) where columns are [x, y, z, visibility]
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        # We assume processing the first face found for simplicity, 
        # actual tracker will crop to bounding box first.
        face_landmarks = results.multi_face_landmarks[0]
        
        # We only take the first 468 points (ignoring the 10 extra iris points for consistency)
        landmarks = np.zeros((468, 4), dtype=np.float32)
        
        for i in range(468):
            lm = face_landmarks.landmark[i]
            # Coordinates are normalized [0.0, 1.0] by MediaPipe. 
            # We scale back to image dimensions for standard metrics.
            landmarks[i] = [lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1], lm.visibility if getattr(lm, 'visibility', None) is not None else 1.0]
            
        return landmarks

    def extract_multiple_meshes(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extracts all face meshes from the frame. Used when tracking multiple subjects.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        meshes = []
        if not results.multi_face_landmarks:
            return meshes
            
        for face_landmarks in results.multi_face_landmarks:
            landmarks = np.zeros((468, 4), dtype=np.float32)
            for i in range(468):
                lm = face_landmarks.landmark[i]
                landmarks[i] = [lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1], lm.visibility if getattr(lm, 'visibility', None) is not None else 1.0]
            meshes.append(landmarks)
            
        return meshes
