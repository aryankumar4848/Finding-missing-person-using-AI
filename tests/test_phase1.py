import numpy as np
import pytest
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher
from ml_service.uncertainty_estimator import UncertaintyEstimator

def generate_dummy_mesh(offset=0.0, scale=1.0, noise_std=0.0):
    """Generates a dummy 468x4 mesh (X, Y, Z, Visibility)"""
    mesh = np.random.randn(468, 4)
    # Ensure visibilities are [0, 1]
    mesh[:, 3] = np.clip(mesh[:, 3] + 0.5, 0.0, 1.0)
    
    # Apply transformation to XYZ
    mesh[:, :3] = (mesh[:, :3] * scale) + offset
    if noise_std > 0:
        mesh[:, :3] += np.random.normal(0, noise_std, (468, 3))
        
    return mesh

def test_procrustes_normalizer():
    # Base mesh
    base_mesh = generate_dummy_mesh()
    
    # Create transformed versions
    translated_mesh = np.copy(base_mesh)
    translated_mesh[:, :3] += np.array([100.0, -50.0, 20.0])
    
    scaled_mesh = np.copy(base_mesh)
    scaled_mesh[:, :3] *= 5.0
    
    normalizer = ProcrustesNormalizer()
    
    norm_base = normalizer.normalize(base_mesh)
    norm_trans = normalizer.normalize(translated_mesh)
    norm_scaled = normalizer.normalize(scaled_mesh)
    
    # 1. Test Translation & Scaling Invariance
    assert np.allclose(norm_base[:, :3], norm_trans[:, :3], atol=1e-5), "Failed translation invariance"
    assert np.allclose(norm_base[:, :3], norm_scaled[:, :3], atol=1e-5), "Failed scale invariance"
    
    # 2. Test Degenerate mesh handling
    zero_mesh = np.zeros((468, 4))
    zero_mesh[:, 3] = 1.0 # full vis
    norm_zero = normalizer.normalize(zero_mesh)
    assert not np.isnan(norm_zero).any(), "NaN in degenerate mesh"
    assert np.allclose(norm_zero[:, :3], 0.0), "Zero mesh should remain zero"

def test_region_biohasher():
    mesh1 = generate_dummy_mesh()
    mesh2 = np.copy(mesh1) # Identical mesh
    mesh3 = generate_dummy_mesh() # Different mesh
    
    normalizer = ProcrustesNormalizer()
    n_mesh1 = normalizer.normalize(mesh1)
    n_mesh2 = normalizer.normalize(mesh2)
    n_mesh3 = normalizer.normalize(mesh3)
    
    # Key 1
    hasher1 = RegionBioHasher(secret_key="UserSecret123", bits_per_region=64)
    res1_key1 = hasher1.generate_hash(n_mesh1)
    res2_key1 = hasher1.generate_hash(n_mesh2)
    res3_key1 = hasher1.generate_hash(n_mesh3)
    
    # Key 2 (Revocability test)
    hasher2 = RegionBioHasher(secret_key="RevokedKey999", bits_per_region=64)
    res1_key2 = hasher2.generate_hash(n_mesh1)
    
    # A. Same input + same key -> identical hash
    for region in res1_key1["hashes"]:
        assert np.array_equal(res1_key1["hashes"][region], res2_key1["hashes"][region]), f"Hashes differ for identical input on region {region}"
        
    # B. Same input + different key -> different hash (Revocability)
    identical_count = 0
    total_regions = 0
    for region in res1_key1["hashes"]:
        total_regions += 1
        if np.array_equal(res1_key1["hashes"][region], res1_key2["hashes"][region]):
            identical_count += 1
    # Very unlikely to be identical across keys for any region due to Random Projection
    assert identical_count < total_regions, "Revoked key produced identical hash!"
    
    # C. Consistency checks
    # Total landmarks should be 468 exactly across regions
    total_landmarks_mapped = sum([len(indices) for indices in hasher1.regions.values()])
    assert total_landmarks_mapped == 468, f"Region mapping incomplete. Total mapped: {total_landmarks_mapped}/468"

def test_uncertainty_estimator():
    mesh = generate_dummy_mesh()
    
    normalizer = ProcrustesNormalizer()
    n_mesh = normalizer.normalize(mesh)
    
    hasher = RegionBioHasher(secret_key="TestKey", bits_per_region=64)
    baseline_res = hasher.generate_hash(n_mesh)
    db_hashes = baseline_res["hashes"]
    visibilities = baseline_res["visibilities"]
    
    # Stable input estimation
    estimator = UncertaintyEstimator(num_perturbations=10, noise_std=1e-5) # Very small noise
    stable_var = estimator.compute_perturbation_variance(n_mesh, hasher, db_hashes, visibilities)
    
    # Noisy input estimation
    estimator_noisy = UncertaintyEstimator(num_perturbations=10, noise_std=0.5) # Large noise
    noisy_var = estimator_noisy.compute_perturbation_variance(n_mesh, hasher, db_hashes, visibilities)
    
    # A. Stable input -> low variance
    assert stable_var < 1e-4, f"Stable variance too high: {stable_var}"
    
    # B. Noisy input -> higher variance
    assert noisy_var > stable_var, f"Noisy variance {noisy_var} should be > stable variance {stable_var}"

def test_final_integration_pipeline():
    """
    Simulates a mini pipeline:
    Frame (represented as mesh here directly since CV2 image needs MediaPipe file deps) 
    -> Normalizer -> BioHasher -> UncertaintyEstimator
    """
    print("\n--- FINAL INTEGRATION TEST ---")
    
    # 1. Extractor mock (we skip CV2 frame processing to keep unit test self-contained, 
    # assume MeshExtractor returned a (468, 4) numpy array)
    raw_mesh = generate_dummy_mesh(offset=150, scale=0.8, noise_std=0.01)
    
    # 2. Normalizer
    normalizer = ProcrustesNormalizer()
    norm_mesh = normalizer.normalize(raw_mesh)
    
    # 3. BioHasher
    hasher = RegionBioHasher(secret_key="ProdKey123", bits_per_region=64)
    hash_results = hasher.generate_hash(norm_mesh)
    db_hashes = hash_results["hashes"]
    visibilities = hash_results["visibilities"]
    
    # 4. Uncertainty
    estimator = UncertaintyEstimator(num_perturbations=5, noise_std=0.05)
    pert_var = estimator.compute_perturbation_variance(norm_mesh, hasher, db_hashes, visibilities)
    
    # Prints for visibility
    print("Hash Output Sizes per Region:")
    for region, h in db_hashes.items():
        print(f"  {region}: {len(h)} bits, Sample: {str(h[:5])}...")
        
    print("\nVisibility Weights per Region:")
    for region, v in visibilities.items():
        print(f"  {region}: {v:.4f}")
        
    print(f"\nPerturbation Variance: {pert_var:.8f}")
    
    assert pert_var >= 0.0, "Variance cannot be negative"
    assert len(db_hashes) == 5, "Should have 5 regions"
