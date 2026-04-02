import os
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict

from ml_service.mesh_extractor import MeshExtractor
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher
from ml_service.uncertainty_estimator import UncertaintyEstimator
from matching_engine.matcher import Matcher
from experiments.baseline_integration import BaselineEngine

class MultiSeverityExperimentRunner:
    """
    Executes multiple runs per severity level evaluating both the structural 
    Mesh logic and CNN Baselines seamlessly.
    """
    def __init__(self, data_dir: str = "experiments/dataset"):
        self.data_dir = data_dir
        self.result_csv_path = "experiments/multilevel_results.csv"
        self.results_df = []

        self.extractor = MeshExtractor(max_num_faces=1)
        self.normalizer = ProcrustesNormalizer()
        self.hasher = RegionBioHasher(secret_key="ProductionKey")
        self.matcher = Matcher(sim_threshold=0.75, unc_threshold=0.08, k_thresh=1)
        self.uncertainty_estimator = UncertaintyEstimator(num_perturbations=5, noise_std=0.05)
        
        self.baseline = BaselineEngine(model_name="ArcFace")
        
        self.mesh_db = []
        self.baseline_db = {}
        
        self.severities = {
            'blur': ['5', '11', '21'],
            'noise': ['10', '25', '40'],
            'low_light': ['0.6', '0.4', '0.2'],
            'occlusion': ['0.2', '0.4', '0.6']
        }
        
    def _extract_identity(self, filename: str) -> str:
        return "_".join(filename.split("_")[:-1])

    def setup_database(self):
        """Enrolls the 'original' clean images as ground-truth representations."""
        orig_path = os.path.join(self.data_dir, "original")
        if not os.path.exists(orig_path):
            print("Run dataset_prep.py first.")
            return

        for filename in os.listdir(orig_path):
            if not filename.endswith(".jpg"): continue
            
            img_path = os.path.join(orig_path, filename)
            identity = self._extract_identity(filename)
            frame = cv2.imread(img_path)
            
            emb = self.baseline.get_embedding(frame)
            if emb is not None:
                self.baseline_db[identity] = emb
                
            meshes = self.extractor.extract_multiple_meshes(frame)
            if len(meshes) > 0:
                norm_mesh = self.normalizer.normalize(meshes[0])
                query_hash = self.hasher.generate_hash(norm_mesh)
                if not any(r['user_id'] == identity for r in self.mesh_db):
                    self.mesh_db.append({"user_id": identity, "hashes": query_hash["hashes"]})

    def eval_pipeline(self, frame: np.ndarray, is_mesh: bool) -> str:
        if not is_mesh:
            emb = self.baseline.get_embedding(frame)
            if emb is None: return "Unknown"
            res = self.baseline.search_database(emb, self.baseline_db)
            return res["match_id"] if res["match_id"] else "Unknown"

        meshes = self.extractor.extract_multiple_meshes(frame)
        if len(meshes) == 0: return "Unknown"
        
        norm_mesh = self.normalizer.normalize(meshes[0])
        query_hash = self.hasher.generate_hash(norm_mesh)
        weights = {k: 1.0 for k in query_hash["hashes"].keys()}
        
        best_id, sim, db_hash = self.matcher.find_best_match(query_hash["hashes"], weights, self.mesh_db)
        if best_id is None: return "Unknown"
        
        var_pert = self.uncertainty_estimator.compute_perturbation_variance(norm_mesh, self.hasher, db_hash, weights)
        decision = self.matcher.evaluate_identity(best_id, sim, 0.0, var_pert, 1)
        
        return decision['match_id'] if decision['match_id'] else "Unknown"

    def run_tests_on_folder(self, folder_path: str, method_is_mesh: bool):
        y_true, y_pred = [], []
        if not os.path.exists(folder_path):
            return [], []
            
        for filename in os.listdir(folder_path):
            if not filename.endswith(".jpg"): continue
            
            true_id = self._extract_identity(filename)
            img_path = os.path.join(folder_path, filename)
            frame = cv2.imread(img_path)
            
            pred = self.eval_pipeline(frame, method_is_mesh)
            
            y_pred.append(pred)
            y_true.append(true_id)
            
        return y_true, y_pred

    def compute_metrics(self, y_true: list, y_pred: list) -> dict:
        total = len(y_true)
        if total == 0: return {'acc':0, 'prec':0, 'rec':0, 'f1':0, 'fpr':0}
        
        tp = sum([1 for p, t in zip(y_pred, y_true) if p == t and p != "Unknown"])
        fp = sum([1 for p, t in zip(y_pred, y_true) if p != t and p != "Unknown"])
        fn = sum([1 for p, t in zip(y_pred, y_true) if p == "Unknown"])
        
        accuracy = sum([1 for p, t in zip(y_pred, y_true) if p == t]) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / total
        
        return {'acc': accuracy, 'prec': precision, 'rec': recall, 'f1': f1, 'fpr': fpr}

    def _execute_run(self, degradation: str, severity: str, method: str, is_mesh: bool):
        folder_name = "original" if degradation == "clean" else f"{degradation}_{severity}"
        target_dir = os.path.join(self.data_dir, folder_name)
        
        # Multiple runs for statistical stability mapping (3 times)
        metrics_collections = defaultdict(list)
        for _ in range(3):
            y_true, y_pred = self.run_tests_on_folder(target_dir, is_mesh)
            mets = self.compute_metrics(y_true, y_pred)
            for k, v in mets.items():
                metrics_collections[k].append(v)
                
        # Average
        avg_metrics = {k: np.mean(v) for k, v in metrics_collections.items()}
        
        self.results_df.append({
            'method': method,
            'degradation_type': degradation,
            'severity_level': severity if degradation != "clean" else "None",
            'accuracy': avg_metrics['acc'],
            'precision': avg_metrics['prec'],
            'recall': avg_metrics['rec'],
            'f1': avg_metrics['f1'],
            'fpr': avg_metrics['fpr']
        })

    def run_all(self):
        self.setup_database()
        
        # 1. Clean Baseline (Original non-degraded)
        print("Evaluating Clean Dataset...")
        self._execute_run("clean", "None", "Mesh", True)
        self._execute_run("clean", "None", "ArcFace", False)
        
        # 2. Multi-Severity Degradations
        for deg, levels in self.severities.items():
            for lvl in levels:
                print(f"Evaluating {deg.upper()} [Level {lvl}]...")
                self._execute_run(deg, lvl, "Mesh", True)
                self._execute_run(deg, lvl, "ArcFace", False)
                
        # Export
        df = pd.DataFrame(self.results_df)
        df.to_csv(self.result_csv_path, index=False)
        print("\n=== SYSTEM METRICS EVALUATION ===")
        print(df.to_string(index=False))

if __name__ == "__main__":
    runner = MultiSeverityExperimentRunner()
    runner.run_all()
