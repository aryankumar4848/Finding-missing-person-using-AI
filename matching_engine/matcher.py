import numpy as np
from typing import Dict, List, Tuple, Any

class Matcher:
    """
    Core matching logic for the Temporal Facial Identification System.
    Combines Weighted Hamming Similarity, Uncertainty logic, and Temporal Consistency
    to make extremely robust accept/reject evaluations.
    """
    def __init__(self, sim_threshold: float = 0.75, unc_threshold: float = 0.05, 
                 k_thresh: int = 10, alpha: float = 0.5, beta: float = 0.5):
        self.sim_threshold = sim_threshold
        self.unc_threshold = unc_threshold
        self.k_thresh = k_thresh
        self.alpha = alpha
        self.beta = beta

    def compute_similarity(self, query_hash: Dict[str, np.ndarray], db_hash: Dict[str, np.ndarray], weights: Dict[str, float]) -> float:
        """
        Computes the Region-Wise Weighted Hamming Similarity.
        
        sim_p = (matching_bits) / (total_bits)
        S = sum(W_p * sim_p) / sum(W_p)
        """
        total_weight = 0.0
        weighted_sim_sum = 0.0
        
        for region in query_hash.keys():
            if region in db_hash:
                q_bits = query_hash[region]
                d_bits = db_hash[region]
                
                # Handling lists or numpy arrays
                if not isinstance(q_bits, np.ndarray):
                    q_bits = np.array(q_bits)
                if not isinstance(d_bits, np.ndarray):
                    d_bits = np.array(d_bits)
                    
                total_bits = len(q_bits)
                if total_bits == 0:
                    continue
                    
                # Calculate raw hamming similarity (0 to 1)
                matching_bits = np.sum(q_bits == d_bits)
                sim_p = float(matching_bits) / total_bits
                
                # Fetch dynamically computed visibility/stability weight (default 1.0)
                w_p = weights.get(region, 1.0)
                
                weighted_sim_sum += (w_p * sim_p)
                total_weight += w_p
                
        if total_weight > 1e-8:
            return weighted_sim_sum / total_weight
        return 0.0

    def find_best_match(self, query_hash: Dict[str, np.ndarray], weights: Dict[str, float], database_records: List[Dict]) -> Tuple[Optional[Any], float, Optional[Dict]]:
        """
        Retrieves the top candidate from the database using Brute Force (simulating FAISS for now).
        database_records format: [{'user_id': 123, 'hashes': {'left_eye': ..., ...}}, ...]
        
        Returns:
            best_id: The identified user ID
            best_sim: Highest similarity score
            best_db_hash: The actual stored hash dictionary for the matched record
        """
        best_sim = -1.0
        best_id = None
        best_db_hash = None
        
        for record in database_records:
            sim = self.compute_similarity(query_hash, record['hashes'], weights)
            if sim > best_sim:
                best_sim = sim
                best_id = record['user_id']
                best_db_hash = record['hashes']
                
        return best_id, best_sim, best_db_hash

    def compute_uncertainty(self, var_temporal: float, var_perturbation: float) -> float:
        """
        U = alpha * Var_temporal + beta * Var_perturbation
        """
        return (self.alpha * var_temporal) + (self.beta * var_perturbation)

    def apply_decision_rules(self, similarity: float, uncertainty: float, consistency: int) -> Dict[str, Any]:
        """
        Accept match only if ALL strict mathematical bounds are satisfied:
        1. S > threshold_sim
        2. U < threshold_unc
        3. C_k > K_thresh
        """
        reasons = []
        
        passed_sim = similarity > self.sim_threshold
        if not passed_sim:
            reasons.append(f"Similarity {similarity:.2f} below threshold {self.sim_threshold}")
            
        passed_unc = uncertainty < self.unc_threshold
        if not passed_unc:
            reasons.append(f"Uncertainty {uncertainty:.4f} exceeds strict boundary {self.unc_threshold}")
            
        passed_cons = consistency >= self.k_thresh
        if not passed_cons:
            reasons.append(f"Consistency {consistency} insufficient (needs {self.k_thresh})")
            
        is_accepted = passed_sim and passed_unc and passed_cons
        
        return {
            'accepted': is_accepted,
            'reason': "Success" if is_accepted else " | ".join(reasons)
        }

    def evaluate_identity(self, best_id: Any, similarity: float, var_temporal: float, var_perturbation: float, consistency_score: int) -> Dict[str, Any]:
        """
        Wraps the mathematical checks into a final execution package output.
        """
        unc = self.compute_uncertainty(var_temporal, var_perturbation)
        decision = self.apply_decision_rules(similarity, unc, consistency_score)
        
        return {
            'match_id': best_id if decision['accepted'] else None,
            'similarity': similarity,
            'uncertainty': unc,
            'decision': 'accepted' if decision['accepted'] else 'rejected',
            'details': decision['reason']
        }
