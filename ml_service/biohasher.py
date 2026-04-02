import numpy as np
import mediapipe as mp
import hashlib

class RegionBioHasher:
    """
    Implements Region-Based BioHashing for revocable facial biometrics.
    Uses independent random projections per semantic region.
    """
    def __init__(self, secret_key: str, bits_per_region: int = 64):
        self.secret_key = secret_key
        self.bits_per_region = bits_per_region
        self.regions = self._define_regions()
        
        # Total bits = 5 regions * bits_per_region (e.g. 5 * 64 = 320 bits)
        self.num_regions = len(self.regions)
        self.total_bits = self.num_regions * self.bits_per_region
        
        # Deterministically generate R_p and b_p for each region based on the secret key
        self.R = {}
        self.b = {}
        self._initialize_projections()

    def _define_regions(self) -> dict:
        """
        Partitions the 468 landmarks into semantic regions.
        Hardcoded approximate block allocations to remove strict dependency on MediaPipe framework constants.
        """
        # Standard contiguous index blocks (approximation for FaceMesh topolgy)
        left_eye_pts = set(range(33, 133))
        right_eye_pts = set(range(263, 363))
        mouth_pts = set(range(0, 33)) | set(range(133, 163))
        nose_pts = set([4, 5, 195, 197, 275, 45, 220, 274, 238])
        
        # Ensure disjoint sets for semantic regions
        left_eye_pts = left_eye_pts - nose_pts
        right_eye_pts = right_eye_pts - nose_pts
        mouth_pts = mouth_pts - nose_pts - left_eye_pts - right_eye_pts
        
        all_semantic = left_eye_pts | right_eye_pts | mouth_pts | nose_pts
        other_pts = set(range(468)) - all_semantic
        
        return {
            'left_eye': list(left_eye_pts),
            'right_eye': list(right_eye_pts),
            'mouth': list(mouth_pts),
            'nose': list(nose_pts),
            'other': list(other_pts)
        }

    def _initialize_projections(self):
        """
        Generates R_p (Random Projection Matrix) and b_p (Bias) for each region.
        The random seed is derived from the secret key to ensure reproducible yet non-invertible hashes.
        """
        for region_name, indices in self.regions.items():
            # Create a unique but deterministic seed for this region using the key
            region_seed_str = f"{self.secret_key}_{region_name}"
            hasher = hashlib.sha256(region_seed_str.encode())
            seed = int(hasher.hexdigest()[:8], 16)
            
            rng = np.random.RandomState(seed)
            
            # Input dimension is number of points in region * 3 (x,y,z coordinates)
            input_dim = len(indices) * 3
            
            # Generate random projection matrix from normal distribution
            # For BioHashing, orthonormalization (e.g. Gram-Schmidt) of R is often used, 
            # but a standard Gaussian matrix is sufficient for high dimensions (Johnson-Lindenstrauss).
            R_p = rng.randn(self.bits_per_region, input_dim)
            
            # Orthonormalizing R_p via QR decomposition for better information packing
            if input_dim >= self.bits_per_region:
                q, r = np.linalg.qr(R_p.T)
                R_p = q.T
            else:
                # If region is very small, we just use the scaled random matrix
                R_p = R_p / np.sqrt(input_dim)
                
            b_p = rng.uniform(-0.1, 0.1, self.bits_per_region) # Small random thresholds
            
            self.R[region_name] = R_p
            self.b[region_name] = b_p

    def _normalize_input(self, x: np.ndarray) -> np.ndarray:
        """
        Normalizes input x before projection: x' = (x - mean) / ||x - mean||
        Ensures mean zero and unit variance per input vector to stabilize signs.
        """
        mean = np.mean(x)
        centered = x - mean
        norm = np.linalg.norm(centered)
        if norm > 1e-8:
            return centered / norm
        return centered

    def generate_hash(self, normalized_mesh: np.ndarray) -> dict:
        """
        Takes a (468, 4) mesh (3D points + visibilities).
        Returns a dictionary of binary bit vectors (hashes) per region, 
        and the average visibility for each region (used for weighting later).
        """
        region_hashes = {}
        region_visibilities = {}
        
        for region_name, indices in self.regions.items():
            # Get the (x,y,z) points for the region
            pts = normalized_mesh[indices, :3].flatten()
            
            # Apply formulation rule: normalize before projection
            x_p = self._normalize_input(pts)
            
            # y_p = sign(R_p * x_p + b_p)
            projection = np.dot(self.R[region_name], x_p) + self.b[region_name]
            # Convert to binary {0, 1} where True means projection >= 0
            y_p = (projection >= 0).astype(np.int8) 
            
            # Calculate region visibility (average of points)
            vis = np.mean(normalized_mesh[indices, 3])
            
            region_hashes[region_name] = y_p
            region_visibilities[region_name] = float(vis)
            
        return {
            "hashes": region_hashes,
            "visibilities": region_visibilities
        }
