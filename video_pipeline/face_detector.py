import os
import urllib.request
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    """
    Robust Multi-Scale Dense Face tracking using YOLOv8-Face weights.
    Downloads the model dynamically if missing.
    """
    def __init__(self, model_path="face_yolov8n.pt"):
        if not os.path.exists(model_path):
            print(f"Downloading YOLOv8 face weights to {model_path}...")
            try:
                from huggingface_hub import hf_hub_download
                import shutil
                downloaded_path = hf_hub_download(repo_id='Bingsu/adetailer', filename='face_yolov8n.pt')
                shutil.copy(downloaded_path, model_path)
            except Exception as e:
                print(f"Failed to automatically download YOLOv8 bounds model! Error: {str(e)}")
                
        self.model = YOLO(model_path)
        
    def detect_faces(self, frame: np.ndarray, conf: float = 0.5):
        """
        Processes frame natively via YOLOv8 inferencing graph.
        Returns: list of [x1, y1, x2, y2, score] arrays.
        """
        # verbose=False mutes per-frame printing
        results = self.model(frame, verbose=False, conf=conf)
        bboxes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Use cpu().numpy() for secure backend processing unreliant on metal/cuda wrappers
                coords = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                bboxes.append([coords[0], coords[1], coords[2], coords[3], float(score)])
        return bboxes
