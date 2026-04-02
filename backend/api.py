import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from matching_engine.database import DatabaseManager, User, BiometricHash
from matching_engine.matcher import Matcher
from ml_service.mesh_extractor import MeshExtractor
from ml_service.normalizer import ProcrustesNormalizer
from ml_service.biohasher import RegionBioHasher
from video_pipeline.video_processor import VideoProcessor
from ml_service.uncertainty_estimator import UncertaintyEstimator

app = FastAPI(title="Privacy-Preserving Facial Identification API")

# Setup Keys and Services
SECRET_KEY = os.getenv("BIO_SECRET_KEY", "DefaultSecureKey2026")

# Database
db_manager = DatabaseManager()
db_manager.init_db()

# Components for stateless Registration (no Tracker)
mesh_extractor = MeshExtractor()
normalizer = ProcrustesNormalizer()
biohasher = RegionBioHasher(secret_key=SECRET_KEY)

# Pipeline Components for Stateful Streaming
video_processor = VideoProcessor(secret_key=SECRET_KEY)
matcher = Matcher()
uncertainty_estimator = UncertaintyEstimator()

# In-memory storage for the latest alerts from video chunks
active_alerts = []


class RegistrationResult(BaseModel):
    user_id: int
    first_name: str
    message: str


def decode_image(file_bytes: bytes) -> np.ndarray:
    """Safely decodes streaming byte payload to OpenCV BGR array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")
    return img


@app.post("/register", response_model=RegistrationResult)
async def register_user(
    first_name: str = Form(...),
    last_name: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Registers a new identity. 
    Strict implementation: Extract topological features, hash them, write to DB, and immediately discard the raw image buffer.
    """
    # 1. Decode image into RAM buffer
    img_bytes = await image.read()
    frame = decode_image(img_bytes)
    
    # 2. Extract Mesh
    # Note: For registration we assume one clear frontal face
    meshes = mesh_extractor.extract_multiple_meshes(frame)
    if len(meshes) == 0:
        raise HTTPException(status_code=400, detail="No face detected for registration.")
        
    mesh = meshes[0] # primary face
    
    # 3. Normalize
    norm_mesh = normalizer.normalize(mesh)
    
    # 4. Hash (Revocable transformation)
    hash_results = biohasher.generate_hash(norm_mesh)
    
    session = db_manager.get_session()
    try:
        # 5. Store Identity + Metadata
        new_user = User(first_name=first_name, last_name=last_name, metadata_json={"source": "api_upload"})
        session.add(new_user)
        session.flush() # Secure auto-increment ID
        
        # 6. Store JSON Hashes exclusively.
        # Ensure we cast numpy int8 arrays to Python standard lists for JSON serialization
        json_hashes = {k: v.tolist() for k, v in hash_results["hashes"].items()}
        
        biometric_hash = BiometricHash(user_id=new_user.id, region_hashes=json_hashes)
        session.add(biometric_hash)
        session.commit()
        
        # Original image 'frame' natively garbage collected out of scope. No file is written to storage.
        return RegistrationResult(user_id=new_user.id, first_name=first_name, message="Secure Biometric Token registered.")
        
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        session.close()


@app.post("/process_frame")
async def process_frame(image: UploadFile = File(...)):
    """
    Ingests a chronological frame from a CCTV stream. 
    1. Runs Object Tracker/Buffer Update
    2. Runs Math Matching
    3. Generates Alerts
    """
    global active_alerts
    img_bytes = await image.read()
    frame = decode_image(img_bytes)
    
    # 1. Run Pipeline (Tracker ID association, Face Hash extraction, Temporal Buffer Memory push)
    # This automatically drops occluded faces according to the SORT heuristics.
    tracked_faces = video_processor.process_frame(frame)
    
    # Pull current DB records once to simulate FAISS query layout
    session = db_manager.get_session()
    db_records = session.query(BiometricHash).all()
    records_cache = [
        {"user_id": r.user_id, "hashes": {k: np.array(v) for k, v in r.region_hashes.items()}}
        for r in db_records
    ]
    session.close()
    
    frame_matches = []

    # 2. Match Phase
    for track_id, data in tracked_faces.items():
        if not data['is_valid']:
            continue
            
        # Pull Temporal stability algorithms from Phase 2
        buffer_frames = video_processor.buffer.get_buffer(track_id)
        if not buffer_frames:
            continue
            
        latest_hash = buffer_frames[-1]['hash']
        latest_mesh = buffer_frames[-1]['mesh']
        
        var_temporal = video_processor.buffer.compute_temporal_variance(track_id)
        consistency = video_processor.buffer.compute_consistency(track_id)
        landmark_stability = video_processor.buffer.compute_landmark_stability(track_id)
        
        # Map dynamic 468 landmark structural stabilities back into Regional Blocks
        # For prototype bridging, we use baseline identical weights or actual visibility.
        regional_weights = buffer_frames[-1].get('visibilities', {key: 1.0 for key in latest_hash.keys()})

        # Matcher Brute Force scan execution (Phase 3 Core)
        best_id, sim, best_db_hash = matcher.find_best_match(latest_hash, regional_weights, records_cache)
        
        if best_id is not None:
            # We must append similarity into buffer for consistency tracking next frame
            video_processor.buffer.buffers[track_id][-1]['similarity'] = sim
            
            # Predict noise perturbation volatility specifically on the matched vector
            var_perturb = uncertainty_estimator.compute_perturbation_variance(
                normalized_mesh=latest_mesh, 
                biohasher=video_processor.hasher, 
                db_hashes=best_db_hash, 
                region_weights=regional_weights
            )
            
            # Final Decision Gates
            decision_payload = matcher.evaluate_identity(best_id, sim, var_temporal, var_perturb, consistency)
            
            payload = {
                "track_id": track_id,
                "bbox": data['bbox'].tolist(),
                "match": decision_payload
            }
            frame_matches.append(payload)
            
            if decision_payload['decision'] == 'accepted':
                active_alerts.append(payload)
                
    # Keep rolling alert buffer to last 100 for memory
    if len(active_alerts) > 100:
        active_alerts = active_alerts[-100:]

    return {"tracked_faces_count": len(tracked_faces), "results": frame_matches}


@app.get("/alerts")
async def get_alerts():
    """Returns successfully verified identities representing an accepted match."""
    global active_alerts
    return {"status": "ok", "alerts": active_alerts}
