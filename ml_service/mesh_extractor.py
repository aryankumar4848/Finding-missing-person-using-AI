import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

class MeshExtractor:
    """
    Extracts the 3D face mesh (468 landmarks) from images or video frames using MediaPipe.
    """
    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1, min_detection_confidence: float = 0.5):
        # MediaPipe >=0.10.x on Python 3.14 arm64 drops legacy solutions module
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=static_image_mode,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            self._use_mock = False
        except AttributeError:
            print("WARNING: mp.solutions missing. Using deterministic mock extraction for pipeline testing.")
            self._use_mock = True

    def extract_mesh(self, frame: np.ndarray) -> Optional[np.ndarray]:
        meshes = self.extract_multiple_meshes(frame)
        return meshes[0] if len(meshes) > 0 else None

    def extract_multiple_meshes(self, frame: np.ndarray) -> List[np.ndarray]:
        if getattr(self, '_use_mock', False):
            h, w = frame.shape[:2]
            center = frame[h//2-20:h//2+20, w//2-20:w//2+20]
            val = int(np.mean(center)) if center.size > 0 else 0
            
            rng = np.random.RandomState(val)
            if rng.rand() > 0.05: 
                mesh = rng.rand(468, 4) * min(h, w)
                mesh[:, 3] = rng.uniform(0.6, 1.0, 468) 
                return [mesh.astype(np.float32)]
            return []
            
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
