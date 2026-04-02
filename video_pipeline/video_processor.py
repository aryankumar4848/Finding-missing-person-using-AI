import cv2
import numpy as np
from typing import Optional, Dict

from ml_service.mesh_extractor import MeshExtractor
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher
from video_pipeline.tracker import Sort, calculate_iou
from video_pipeline.temporal_buffer import TemporalBuffer
from video_pipeline.face_detector import FaceDetector

class VideoProcessor:
    """
    Integrates the ML, Tracking, and Buffer modules into a complete real-time video processing pipeline.
    Responsible exclusively for the data ingestion and normalization flow, not identity matching.
    """
    def __init__(self, secret_key: str):
        # ML Service Components
        self.detector = FaceDetector()
        self.extractor = MeshExtractor(max_num_faces=1) # Extraction now independently handles single crops reliably
        self.normalizer = ProcrustesNormalizer()
        self.hasher = RegionBioHasher(secret_key=secret_key, bits_per_region=64)
        
        # Tracking & Temporal Modeling
        self.tracker = Sort(max_age=15, min_hits=3, iou_threshold=0.3)
        self.buffer = TemporalBuffer(max_frames=15, lambda_param=100.0)

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Main pipeline step operated on every raw frame.
        Outputs a structural dictionary representing currently tracked active faces in this frame.
        """
        # Step 1: Object Detection (via YOLOv8)
        bboxes = self.detector.detect_faces(frame)
        self.last_mesh_count = len(bboxes)
        
        original_bboxes = [b[:4] for b in bboxes]
        detections = np.array(bboxes) if len(bboxes) > 0 else np.empty((0, 5))
        
        # Step 2: Run SORT Algorithm Data Association & ID Generation
        tracked_objects = self.tracker.update(detections)
        
        frame_results = {}
        
        # Step 3: Per-Track Processing & Buffering Update
        for trk in tracked_objects:
            trk_bbox = trk[:4]
            track_id = int(trk[4])
            
            # Map tracking data back to original YOLO bbox via highest IoU
            best_iou = 0.0
            matched_bbox = None
            
            for ob in original_bboxes:
                iou = calculate_iou(trk_bbox, ob)
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox = ob
                    
            if matched_bbox is not None and best_iou > 0.3:
                x1, y1, x2, y2 = map(int, matched_bbox)
                h_frame, w_frame = frame.shape[:2]
                
                # Expand box slightly to give MediaPipe adequate facial context
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w_frame, x2 + pad_x)
                cy2 = min(h_frame, y2 + pad_y)
                
                area = (cx2 - cx1) * (cy2 - cy1)
                
                # Multi-Scale Handling: Detect resolution viability
                if area < 3600: # Below an effective 60x60 pixel region
                    frame_results[track_id] = {
                        'bbox': trk_bbox,
                        'is_valid': False,
                        'reason': 'low_res_face'
                    }
                    continue
                    
                crop_img = frame[cy1:cy2, cx1:cx2]
                
                # Mesh local extraction mapped back to Global projection points
                crop_mesh = self.extractor.extract_mesh(crop_img)
                
                if crop_mesh is not None:
                    # MediaPipe outputs coordinates mapped identically to pixel width bounds natively inside crop
                    global_mesh = crop_mesh.copy()
                    global_mesh[:, 0] += cx1 # Shift X logic back globally
                    global_mesh[:, 1] += cy1 # Shift Y logic back globally
                    
                    # Normalization uses normalized mathematical logic, inherently robust and scale-invariant
                    norm_mesh = self.normalizer.normalize(global_mesh)
                    hash_results = self.hasher.generate_hash(norm_mesh)
                    
                    self.buffer.add_frame(
                        track_id=track_id, 
                        mesh=norm_mesh, 
                        hash_dict=hash_results["hashes"],
                        similarity=None
                    )
                    
                    frame_results[track_id] = {
                        'bbox': trk_bbox,
                        'is_valid': True
                    }
                else:
                    frame_results[track_id] = {
                        'bbox': trk_bbox,
                        'is_valid': False,
                        'reason': 'mesh_extraction_failed'
                    }
            else:
                # Extracted tracking failed or track is coasting strictly on Kalman Prediction (occluded)
                frame_results[track_id] = {
                    'bbox': trk_bbox,
                    'is_valid': False,
                    'reason': 'coast'
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

    def draw_debug_overlay(self, frame: np.ndarray, matched_entities: list) -> np.ndarray:
        """
        Applies mathematical bounding boxes and strictly displays metrics.
        matched_entities expects a list of dictionaries with matching ML states.
        """
        out_frame = frame.copy()
        
        for entity in matched_entities:
            bbox = entity.get("bbox")
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = map(int, bbox)
            status = entity.get("status", "rejected")
            track_id = entity.get("track_id", "?")
            identity = entity.get("identity", "Unknown")
            sim = entity.get("similarity", 0.0)
            unc = entity.get("uncertainty", 0.0)
            cons = entity.get("consistency", 0)
            
            # Color formulation
            status_text = "UNKNOWN"
            if status == "accepted":
                color = (0, 255, 0) # Green (BGR OpenCV layout)
                status_text = "MATCH"
            elif status == "warming_up":
                color = (0, 165, 255) # Orange 
                status_text = "BUFFERING"
            elif status == "rejected_high_uncertainty":
                color = (0, 165, 255) # Orange/Redish
                identity = "Unknown"
                status_text = "REJECTED (HIGH UNCERTAINTY)"
            elif status == "low_res_face":
                color = (0, 255, 255) # Yellow format
                identity = "Unknown"
                status_text = "LOW RES FACE"
            else:
                color = (0, 0, 255) # Red (rejected / unknown bounds)
                identity = "Unknown"
                status_text = "UNKNOWN"
                
            # 1. Bounding Box
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            
            # Typography config
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # 2. Track ID banner above box
            title = f"ID: {track_id} | {identity} | {status_text}"
            cv2.putText(out_frame, title, (x1, max(y1 - 10, 20)), font, 0.6, color, 2)
            
            # 3. Floating Math Metrics mapping underneath the bounding box structurally
            metrics_y = y2 + 20
            if status != "warming_up":
                cv2.putText(out_frame, f"S: {sim:.2f}", (x1, metrics_y), font, 0.5, (255, 255, 255), 1)
                cv2.putText(out_frame, f"U: {unc:.3f}", (x1, metrics_y + 20), font, 0.5, (255, 255, 255), 1)
                cv2.putText(out_frame, f"C: {cons}/15", (x1, metrics_y + 40), font, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(out_frame, "C: Warming Up Buffer...", (x1, metrics_y), font, 0.5, (0, 255, 255), 1)

        return out_frame
