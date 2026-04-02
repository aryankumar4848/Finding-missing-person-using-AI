import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

class DatasetPreparer:
    """
    Downloads LFW and generates dynamic structural bounds for testing.
    Now structured for multi-severity levels.
    """
    def __init__(self, output_dir: str = "experiments/dataset"):
        self.output_dir = output_dir
        self.severities = {
            'blur': [5, 11, 21],
            'noise': [10, 25, 40],
            'low_light': [0.6, 0.4, 0.2],
            'occlusion': [0.2, 0.4, 0.6]
        }
        
        # Original clean baseline
        os.makedirs(os.path.join(output_dir, 'original'), exist_ok=True)
        
        # Degradation severities
        for deg, levels in self.severities.items():
            for lvl in levels:
                os.makedirs(os.path.join(output_dir, f"{deg}_{lvl}"), exist_ok=True)

    def download_lfw(self, min_faces: int = 20):
        print("Downloading LFW... (this may take a moment)")
        lfw = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=1.0)
        return lfw.images, lfw.target, lfw.target_names

    def apply_degradations_multi(self, rgb_float: np.ndarray) -> dict:
        base_img = (rgb_float * 255).astype(np.uint8)
        results = {'original': cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)}
        
        # Blur
        for b in self.severities['blur']:
            blurred = cv2.GaussianBlur(base_img, (b, b), 0)
            results[f'blur_{b}'] = cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR)
            
        # Low Light
        for a in self.severities['low_light']:
            low = cv2.convertScaleAbs(base_img, alpha=a, beta=0)
            results[f'low_light_{a}'] = cv2.cvtColor(low, cv2.COLOR_RGB2BGR)
            
        # Noise
        for n in self.severities['noise']:
            noise_mat = np.random.normal(0, n, base_img.shape)
            noisy = np.clip(base_img.astype(np.float32) + noise_mat, 0, 255).astype(np.uint8)
            results[f'noise_{n}'] = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
            
        # Occlusion
        for occ in self.severities['occlusion']:
            occ_img = base_img.copy()
            h, w = base_img.shape[:2]
            start_y = int(h * (1.0 - occ))
            occ_img[start_y:h, :] = 0
            results[f'occlusion_{occ}'] = cv2.cvtColor(occ_img, cv2.COLOR_RGB2BGR)
            
        return results

    def process_and_save(self, limit: int = 150):
        """Processes and writes to disk."""
        images, targets, target_names = self.download_lfw()
        process_count = min(len(images), limit)
        
        for i in range(process_count):
            img_normalized = images[i]
            person_name = target_names[targets[i]].replace(' ', '_')
            variants = self.apply_degradations_multi(img_normalized)
            
            for deg_name, cv_img in variants.items():
                save_path = os.path.join(self.output_dir, deg_name, f"{person_name}_{i:04d}.jpg")
                cv2.imwrite(save_path, cv_img)

if __name__ == "__main__":
    preparer = DatasetPreparer()
    # Executing localized limit format for rapid test iteration compilation speed
    preparer.process_and_save(limit=80) 
    print("Multi-severity dataset finalized.")
