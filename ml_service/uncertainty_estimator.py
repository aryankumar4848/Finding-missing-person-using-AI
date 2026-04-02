import numpy as np
from .biohasher import RegionBioHasher

class UncertaintyEstimator:
    """
    Computes uncertainty for matches using Perturbation Variance and Temporal Variance.
    """
    def __init__(self, num_perturbations: int = 10, noise_std: float = 0.01):
        self.num_perturbations = num_perturbations
        self.noise_std = noise_std
        
    def _hamming_similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Returns similarity [0, 1] where 1 means identical hashes."""
        match_count = np.sum(hash1 == hash2)
        total = len(hash1)
        return float(match_count) / total if total > 0 else 0.0
        
    def compute_weighted_similarity(self, query_hash_dict: dict, db_hash_dict: dict, region_weights: dict) -> float:
        """
        Computes region-weighted hamming distance (converted to similarity).
        Used by the matcher and the perturbation loop.
        """
        total_weight = 0.0
        weighted_sim_sum = 0.0
        
        for p in query_hash_dict.keys():
            if p in db_hash_dict:
                w_p = region_weights.get(p, 1.0)
                sim_p = self._hamming_similarity(query_hash_dict[p], db_hash_dict[p])
                weighted_sim_sum += w_p * sim_p
                total_weight += w_p
                
        if total_weight > 1e-6:
            return weighted_sim_sum / total_weight
        return 0.0

    def compute_perturbation_variance(
        self, 
        normalized_mesh: np.ndarray, 
        biohasher: RegionBioHasher, 
        db_hashes: dict, 
        region_weights: dict
    ) -> float:
        """
        Calculates Var_perturbation by injecting Gaussian noise to the mesh 
        and observing how much the similarity score to db_hashes fluctuates.
        """
        similarities = []
        
        for _ in range(self.num_perturbations):
            # Inject noise only to the x,y,z coordinates, leave visibilities alone (last column)
            noise = np.random.normal(0, self.noise_std, (468, 3))
            
            perturbed_mesh = np.copy(normalized_mesh)
            perturbed_mesh[:, :3] += noise
            
            # Re-hash the perturbed mesh
            perturbed_results = biohasher.generate_hash(perturbed_mesh)
            perturbed_hashes = perturbed_results["hashes"]
            
            # Calculate new similarity against the static DB candidate
            sim = self.compute_weighted_similarity(perturbed_hashes, db_hashes, region_weights)
            similarities.append(sim)
            
        # Return the variance of the similarities
        return float(np.var(similarities))
        
    def compute_combined_uncertainty(self, var_temporal: float, var_perturbation: float, alpha: float = 0.5, beta: float = 0.5) -> float:
        """
        U_t = alpha * Var_temporal + beta * Var_perturbation
        """
        return alpha * var_temporal + beta * var_perturbation
