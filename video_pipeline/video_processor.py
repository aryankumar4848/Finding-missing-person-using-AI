import cv2
import numpy as np
from typing import Optional, Dict

from ml_service.mesh_extractor import MeshExtractor
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher
from video_pipeline.tracker import Sort, calculate_iou
from video_pipeline.temporal_buffer import TemporalBuffer

class VideoProcessor:
    """
    Integrates the ML, Tracking, and Buffer modules into a complete real-time video processing pipeline.
    Responsible exclusively for the data ingestion and normalization flow, not identity matching.
    """
    def __init__(self, secret_key: str):
        # ML Service Components
        self.extractor = MeshExtractor(max_num_faces=10) # Track multiple people simultaneously
        self.normalizer = ProcrustesNormalizer()
        self.hasher = RegionBioHasher(secret_key=secret_key, bits_per_region=64)
        
        # Tracking & Temporal Modeling
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
        self.buffer = TemporalBuffer(max_frames=15, lambda_param=100.0)

    def _match_track_to_mesh(self, tracked_bbox: np.ndarray, original_bboxes: list, meshes: list) -> Optional[np.ndarray]:
        """
        Maps the Kalman Filter predicted bouncing box back to the exact MediaPipe mesh
        extracted in the current frame using highest IoU.
        """
        best_iou = 0.0
        best_match_idx = -1
        
        for idx, original_bbox in enumerate(original_bboxes):
            # Calculate IoU overlay between Tracker Box and raw Mesh Box
            iou = calculate_iou(tracked_bbox, original_bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = idx
                
        # If IoU > 0.3, we consider it a matching valid frame extraction
        if best_match_idx != -1 and best_iou > 0.3:
            return meshes[best_match_idx]
        return None

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Main pipeline step operated on every raw frame.
        Outputs a structural dictionary representing currently tracked active faces in this frame.
        """
        # Step 1 & 2: Object Detection & Mesh Extraction (via MediaPipe natively)
        meshes = self.extractor.extract_multiple_meshes(frame)
        
        # Convert raw meshes into pseudo-detections [x1, y1, x2, y2, score] for the Tracker
        original_bboxes = []
        detections = []
        for mesh in meshes:
            bbox = self.tracker.get_mesh_bbox(mesh)
            original_bboxes.append(bbox)
            detections.append([bbox[0], bbox[1], bbox[2], bbox[3], 1.0])
            
        detections = np.array(detections) if len(detections) > 0 else np.empty((0, 5))
        
        # Step 3: Run SORT Algorithm Data Association & ID Generation
        # Returns [x1, y1, x2, y2, track_id]
        tracked_objects = self.tracker.update(detections)
        
        frame_results = {}
        
        # Step 4: Per-Track Processing & Buffering Update
        for trk in tracked_objects:
            trk_bbox = trk[:4]
            track_id = int(trk[4])
            
            # Map tracking data back to specific exact localized mesh
            matched_mesh = self._match_track_to_mesh(trk_bbox, original_bboxes, meshes)
            
            if matched_mesh is not None:
                # Normalization
                norm_mesh = self.normalizer.normalize(matched_mesh)
                # Revocable Biometric Projection
                hash_results = self.hasher.generate_hash(norm_mesh)
                
                # Update Temporal Vector sliding window
                self.buffer.add_frame(
                    track_id=track_id,
                    mesh=norm_mesh,
                    hash_dict=hash_results["hashes"],
                    similarity=None # Identity is evaluated separately in Phase 3
                )
                
                frame_results[track_id] = {
                    'bbox': trk_bbox,
                    'is_valid': True
                }
            else:
                # Extracted mesh failed or track is coasting strictly on Kalman Prediction (i.e. currently heavily occluded)
                frame_results[track_id] = {
                    'bbox': trk_bbox,
                    'is_valid': False
                }
                
        return frame_results

    def process_video_generator(self, video_path: str):
        """
        Yields generator wrapper iterating over a standard video/webcam feed securely.
        Yields frame_results strictly representing memory flow per frame interval.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video feed: {video_path}")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_results = self.process_frame(frame)
            yield frame_results
            
        cap.release()
