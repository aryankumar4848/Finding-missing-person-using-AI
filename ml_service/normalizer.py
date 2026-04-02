import numpy as np

class ProcrustesNormalizer:
    """
    Normalizes 3D face meshes to be translation, scale, and (optionally) rotation invariant.
    """
    def __init__(self, reference_mesh: np.ndarray = None):
        """
        Args:
            reference_mesh: Optional (468, 3) reference mesh to align to. 
                            If None, only performs translation and scaling.
        """
        self.reference_mesh = reference_mesh
        if self.reference_mesh is not None:
            self.reference_mesh = self._center_and_scale(self.reference_mesh)

    def _center_and_scale(self, mesh3d: np.ndarray) -> np.ndarray:
        """Translates the center of mass to origin and scales by root mean square distance."""
        # Mean centering
        mean = np.mean(mesh3d, axis=0)
        centered = mesh3d - mean
        
        # Scaling by root mean square norm
        scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        
        # Avoid division by zero
        if scale > 1e-6:
            scaled = centered / scale
        else:
            scaled = centered
            
        return scaled

    def _procrustes_align(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Aligns source to target using SVD to find optimal rotation matrix.
        Assumes both source and target are already centered and scaled.
        """
        # Calculate covariance matrix
        H = source.T @ target
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R = Vt.T @ U.T
        
        # Handling reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # Apply rotation
        aligned = source @ R.T
        return aligned

    def normalize(self, landmarks_with_vis: np.ndarray) -> np.ndarray:
        """
        Takes (468, 4) array from extractor [x,y,z,v].
        Returns (468, 4) array where [x,y,z] are normalized and [v] is unchanged.
        """
        points_3d = landmarks_with_vis[:, :3]
        visibilities = landmarks_with_vis[:, 3:]
        
        normalized_3d = self._center_and_scale(points_3d)
        
        if self.reference_mesh is not None:
            normalized_3d = self._procrustes_align(normalized_3d, self.reference_mesh)
            
        # Re-attach visibilities
        return np.hstack([normalized_3d, visibilities])
