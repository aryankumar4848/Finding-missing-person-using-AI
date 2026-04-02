import cv2
import numpy as np
import time
from matching_engine.database import DatabaseManager, BiometricHash
from matching_engine.matcher import Matcher
from video_pipeline.video_processor import VideoProcessor
from ml_service.uncertainty_estimator import UncertaintyEstimator

class PipelineTester:
    """
    Executes the End-to-End local validation of the Real-time Video Processor logic.
    Mocks FastAPI streaming entirely inside the console.
    """
    def __init__(self, source=0):
        # Allow initializing from a local mp4 file or directly tapping the system webcam
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        # Load local SQLite database bindings
        self.db_manager = DatabaseManager()
        self.db_manager.init_db()
        
        self.video_processor = VideoProcessor(secret_key="ExpSecKey2026")
        self.matcher = Matcher(sim_threshold=0.75, unc_threshold=0.08, k_thresh=3)
        self.uncertainty = UncertaintyEstimator()
        
        print("Pipeline Initialized. Compiling Database constraints...")
        self.db_records = self._load_database_state()

    def _load_database_state(self) -> list:
        session = self.db_manager.get_session()
        records = session.query(BiometricHash).all()
        # Format explicitly for FAISS/Brute-force array matching dictionary expected by Matcher
        cache = [{"user_id": r.user_id, "hashes": {k: np.array(v) for k, v in r.region_hashes.items()}} for r in records]
        session.close()
        return cache

    def start_loop(self):
        if not self.cap.isOpened():
            print(f"Error: Unable to explicitly open video source: {self.source}")
            return
            
        print("\n=== STARTING REAL-TIME TRACKING INGESTION ===")
        print("Press 'q' to terminate the visual loop.\n")
        
        frame_idx = 0
        fps_start = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream reached.")
                break
                
            frame_idx += 1
            
            # Step 1: Execute Video Processor (Mesh -> Tracking -> Normalization -> BioHashing -> Temporal Cache)
            tracked_faces = self.video_processor.process_frame(frame)
            
            if not tracked_faces:
                continue

            print(f"--- FRAME {frame_idx} ---")
            
            # Step 2: Extract from Matcher evaluating strict Phase 3 criteria
            for track_id, data in tracked_faces.items():
                if not data['is_valid']:
                    print(f"  Track ID {track_id} | Extractor Failed (Likely fully occluded/coasting)")
                    continue
                    
                buffer = self.video_processor.buffer.get_buffer(track_id)
                if not buffer:
                    continue
                    
                latest_frame = buffer[-1]
                latest_hash = latest_frame['hash']
                latest_mesh = latest_frame['mesh']
                
                # Fetch dynamically bounded mathematical stabilities across sliding window
                weights = latest_frame.get('visibilities', {k: 1.0 for k in latest_hash.keys()})
                var_temporal = self.video_processor.buffer.compute_temporal_variance(track_id)
                consistency = self.video_processor.buffer.compute_consistency(track_id)
                
                # Match against local disk Cache
                best_id, sim, db_hash = self.matcher.find_best_match(latest_hash, weights, self.db_records)
                
                # Maintain temporal validation feedback loop structure required by consistency metric
                if best_id is not None:
                    self.video_processor.buffer.buffers[track_id][-1]['similarity'] = sim
                    var_pert = self.uncertainty.compute_perturbation_variance(latest_mesh, self.video_processor.hasher, db_hash, weights)
                    
                    decision = self.matcher.evaluate_identity(best_id, sim, var_temporal, var_pert, consistency)
                    
                    final_id = decision['match_id'] if decision['match_id'] else "Unknown"
                    print(f"  Track ID: {track_id}")
                    print(f"  Matched Identity: {final_id}")
                    print(f"  Similarity Score: {sim:.3f}")
                    print(f"  Uncertainty Score: {decision['uncertainty']:.4f}")
                    print(f"  Consistency Score: {consistency}/15")
                    if final_id == "Unknown":
                        print(f"  REJECT REASON: {decision['details']}")
                else:
                    print(f"  Track ID: {track_id} | Matched Identity: Unknown | No close topology found.")

        self.cap.release()

if __name__ == "__main__":
    # Can substitute 0 mechanically for the system webcam
    tester = PipelineTester(source=0)
    # tester.start_loop() # Commented dynamically to prevent blocking execution without a monitor
    print("Script correctly generated. To test physically on your local OSX machine with a GUI, run: `python test_pipeline.py` and uncomment start_loop()")
