import cv2
import numpy as np
import time
from matching_engine.database import DatabaseManager, BiometricHash
from matching_engine.matcher import Matcher
from ml_service.uncertainty_estimator import UncertaintyEstimator
from ml_service.mesh_extractor import MeshExtractor
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher

def main():
    print("Initializing components for test scenarios...")
    db_manager = DatabaseManager()
    db_manager.init_db()
    
    extractor = MeshExtractor(max_num_faces=1)
    normalizer = ProcrustesNormalizer()
    hasher = RegionBioHasher(secret_key="ExpSecKey2026", bits_per_region=64)
    matcher = Matcher(sim_threshold=0.75, unc_threshold=0.08, k_thresh=1) # Low k_thresh for instantaneous matching testing
    uncertainty = UncertaintyEstimator()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found locally. Skipping register integration test.")
        return

    print("=== REGISTRATION PHASE ===")
    print("Look at the camera and press 'r' to register your face, or 'q' to abort.")
    registered_hash = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Registration - Press 'r' to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            meshes = extractor.extract_multiple_meshes(frame)
            if len(meshes) > 0:
                mesh = meshes[0]
                norm_mesh = normalizer.normalize(mesh)
                hash_dict = hasher.generate_hash(norm_mesh)["hashes"]
                registered_hash = hash_dict
                
                print("Face successfully registered!")
                
                # Setup Database explicitly
                session = db_manager.get_session()
                # Clear for local dev testing
                session.query(BiometricHash).delete()
                
                # Numpy to lists explicit translation
                db_hash_dict = {k: v.tolist() for k, v in hash_dict.items()}
                new_record = BiometricHash(user_id="local_test_user", region_hashes=db_hash_dict)
                session.add(new_record)
                session.commit()
                session.close()
                break
            else:
                print("No face detected, try again.")
        elif key == ord('q'):
            break

    cv2.destroyWindow("Registration - Press 'r' to capture")

    if not registered_hash:
        print("Registration aborted.")
        cap.release()
        return

    print("\n=== MATCHING PHASE ===")
    print("Looking at the camera... Press 'q' to quit.")
    
    # Reload local cache
    session = db_manager.get_session()
    records = session.query(BiometricHash).all()
    db_records = [{"user_id": r.user_id, "hashes": {k: np.array(v) for k, v in r.region_hashes.items()}} for r in records]
    session.close()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = frame.copy()
        meshes = extractor.extract_multiple_meshes(frame)
        
        if len(meshes) > 0:
            mesh = meshes[0]
            norm_mesh = normalizer.normalize(mesh)
            hash_dict = hasher.generate_hash(norm_mesh)["hashes"]
            
            # Setup base visibility weighting for real-time emulation
            weights = {k: 1.0 for k in hash_dict.keys()}
            
            best_id, sim, db_hash = matcher.find_best_match(hash_dict, weights, db_records)
            
            if best_id is not None:
                # Simulating temporal metrics locally
                var_pert = uncertainty.compute_perturbation_variance(norm_mesh, hasher, db_hash, weights)
                decision = matcher.evaluate_identity(best_id, sim, 0.0, var_pert, 1) # temporal var is 0.0
                
                status = "MATCH" if decision['match_id'] else "NO MATCH"
                details = decision['details']
                reason_str = f"Reasons: {details}" if not decision['match_id'] else "Reasons: Success"
                
                print(f"{status} | ID: {best_id} | S: {sim:.4f} | U: {decision['uncertainty']:.4f} | {reason_str}")
                cv2.putText(out_frame, f"{status} (S: {sim:.2f})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 255, 0) if status == "MATCH" else (0, 0, 255), 2)
            else:
                print("NO MATCH | S: 0 | User not found in database array limits")
                cv2.putText(out_frame, "NO MATCH", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
             print("No Track")
             cv2.putText(out_frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
             
        cv2.imshow("Matching - Press 'q' to quit", out_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
