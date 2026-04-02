import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml_service.mesh_extractor import MeshExtractor
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher
from ml_service.uncertainty_estimator import UncertaintyEstimator
from matching_engine.matcher import Matcher
from experiments.baseline_integration import BaselineEngine
from video_pipeline.temporal_buffer import TemporalBuffer

class ExperimentRunner:
    """
    Executes the analytical comparisons between the Proposed 3D Mesh Pipeline 
    and the ArcFace Deep Learning Baseline across multiple mathematical stress tests.
    """
    def __init__(self, data_dir: str = "experiments/dataset"):
        self.data_dir = data_dir
        self.result_csv_path = "experiments/results.csv"
        self.results_data = []

        # System Pipeline Modules
        self.extractor = MeshExtractor(max_num_faces=1)
        self.normalizer = ProcrustesNormalizer()
        self.hasher = RegionBioHasher(secret_key="ExpSecKey2026")
        self.matcher = Matcher(sim_threshold=0.75, unc_threshold=0.08, k_thresh=1) # K=1 for default single frame
        self.uncertainty_estimator = UncertaintyEstimator(num_perturbations=10, noise_std=0.03)
        
        # Baseline
        self.baseline = BaselineEngine(model_name="ArcFace")
        
        # In-Memory DB Simulation
        self.mesh_db = []
        self.baseline_db = {}
        
        self.init_databases()

    def _extract_identity_from_filename(self, filename: str) -> str:
        # e.g., "Ariel_Sharon_0001.jpg" -> "Ariel_Sharon"
        return "_".join(filename.split("_")[:-1])

    def init_databases(self):
        """Enroll original standard images into both systems databases."""
        print("Enrolling Identities into Backend Database...")
        orig_path = os.path.join(self.data_dir, "original")
        if not os.path.exists(orig_path):
            print("Dataset not found. Please run dataset_prep.py first.")
            return

        for filename in os.listdir(orig_path):
            if not filename.endswith(".jpg"): continue
                
            img_path = os.path.join(orig_path, filename)
            identity = self._extract_identity_from_filename(filename)
            frame = cv2.imread(img_path)
            
            # --- Baseline Enrollment ---
            baseline_emb = self.baseline.get_embedding(frame)
            if baseline_emb is not None:
                # Naive mean pooling if we enroll multiple per person
                self.baseline_db[identity] = baseline_emb
                
            # --- Proposed System Enrollment ---
            meshes = self.extractor.extract_multiple_meshes(frame)
            if len(meshes) > 0:
                norm_mesh = self.normalizer.normalize(meshes[0])
                hash_results = self.hasher.generate_hash(norm_mesh)
                # Ensure no exact duplicate identities exist to simplify testing code
                if not any(r['user_id'] == identity for r in self.mesh_db):
                    self.mesh_db.append({"user_id": identity, "hashes": hash_results["hashes"]})

    def run_single_frame_experiment(self, noise_type: str = "blurred"):
        """
        Experiment 1: Stress tests single mathematical instantaneous extraction without tracking.
        """
        print(f"\nRunning Single-Frame Experiment [{noise_type}]...")
        test_path = os.path.join(self.data_dir, noise_type)
        if not os.path.exists(test_path): return
        
        y_true, y_pred_mesh, y_pred_base = [], [], []

        for filename in os.listdir(test_path):
            if not filename.endswith(".jpg"): continue
            
            true_id = self._extract_identity_from_filename(filename)
            img_path = os.path.join(test_path, filename)
            frame = cv2.imread(img_path)
            
            # Baseline
            emb = self.baseline.get_embedding(frame)
            base_res = self.baseline.search_database(emb, self.baseline_db)
            y_pred_base.append(base_res["match_id"] if base_res["match_id"] else "Unknown")
            
            # Mesh Pipeline
            meshes = self.extractor.extract_multiple_meshes(frame)
            if len(meshes) == 0:
                y_pred_mesh.append("Unknown")
            else:
                norm_mesh = self.normalizer.normalize(meshes[0])
                query_hash = self.hasher.generate_hash(norm_mesh)
                weights = {k: 1.0 for k in query_hash["hashes"].keys()}
                best_id, sim, _ = self.matcher.find_best_match(query_hash["hashes"], weights, self.mesh_db)
                
                # Single frame decision (ignoring Temporal/Uncertainty for isolation testing)
                y_pred_mesh.append(best_id if sim > 0.70 else "Unknown")
                
            y_true.append(true_id)
            
        self.compute_and_log_metrics(f"Single_Frame_{noise_type}_Mesh", y_true, y_pred_mesh)
        self.compute_and_log_metrics(f"Single_Frame_{noise_type}_ArcFace", y_true, y_pred_base)


    def run_temporal_experiment(self, noise_type: str = "noisy"):
        """
        Experiment 2: Connects the TemporalBuffer to simulate continuous tracking identity resolution.
        """
        print(f"\nRunning Temporal Tracking Experiment [{noise_type}]...")
        # (Simulating sequence behavior using repeated buffered ingestion)
        # Will compute consistency and temporal smoothing
        # Mocking temporal implementation for metric collection
        pass

    def run_uncertainty_experiment(self):
        """
        Experiment 3: With vs Without Uncertainty bounds matching constraint.
        """
        print("\nRunning Uncertainty Filtering Bounds Experiment...")
        pass

    def run_partial_face_experiment(self):
        """
        Experiment 4: Evaluates topological mapping against Baseline occlusion.
        """
        print("\nRunning Partial (Occluded) Evaluation...")
        self.run_single_frame_experiment("occluded")

    def compute_and_log_metrics(self, experiment_name: str, y_true: list, y_pred: list):
        """
        Calculates scikit-learn statistical analysis and writes formatting.
        """
        # Exclude 'Unknown' from true metrics map, treating them as structural False Negatives
        y_true_binary = [1 if t != "Unknown" else 0 for t in y_true]
        y_pred_binary = [1 if p == t else 0 for p, t in zip(y_pred, y_true)]
        y_false_positives = [1 if (p != "Unknown" and p != t) else 0 for p, t in zip(y_pred, y_true)]
        
        acc = np.mean(y_pred_binary) if len(y_pred_binary) else 0
        tp = sum(y_pred_binary)
        fp = sum(y_false_positives)
        fn = sum([1 for p, t in zip(y_pred, y_true) if p == "Unknown"])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / len(y_pred_binary) if len(y_pred_binary) else 0
        
        self.results_data.append({
            "Experiment": experiment_name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "FPR": fpr
        })
        
    def export_results(self):
        """Saves internal dict results to CSV and prints cleanly to Console."""
        if len(self.results_data) == 0:
            return
            
        df = pd.DataFrame(self.results_data)
        df.to_csv(self.result_csv_path, index=False)
        print("\n=== SYSTEM METRICS EVALUATION ===")
        print(df.to_string(index=False))
