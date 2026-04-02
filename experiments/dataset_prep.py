import os
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people

class DatasetPreparer:
    """
    Downloads the LFW dataset and generates synthetic CCTV degradation analogs
    required to validate the mathematical boundaries formulated in Phase 1 & 2.
    """
    def __init__(self, output_dir: str = "dataset"):
        self.output_dir = output_dir
        self.paths = {
            'original': os.path.join(output_dir, 'original'),
            'blurred': os.path.join(output_dir, 'blurred'),
            'low_light': os.path.join(output_dir, 'low_light'),
            'noisy': os.path.join(output_dir, 'noisy'),
            'occluded': os.path.join(output_dir, 'occluded')
        }
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def download_lfw(self, min_faces_per_person: int = 15):
        """Fetches the LFW dataset via scikit-learn."""
        print(f"Downloading/Loading LFW faces (min {min_faces_per_person} faces per person)...")
        # color=True to get RGB images for MediaPipe
        lfw_people = fetch_lfw_people(min_faces_per_person=min_faces_per_person, color=True, resize=1.0)
        return lfw_people.images, lfw_people.target, lfw_people.target_names

    def apply_cctv_degradations(self, rgb_image_float: np.ndarray) -> dict:
        """
        Applies mathematical optical degradation mimicking real CCTV anomalies.
        Inputs are normalized floats [0.0, 1.0] from sklearn.
        Returns uint8 [0, 255] arrays.
        """
        # Convert to standard 8-bit OpenCV processing format
        base_img = (rgb_image_float * 255).astype(np.uint8)
        
        # 1. Gaussian Blur: Simulates fundamentally misaligned lens focus or motion blur
        blurred = cv2.GaussianBlur(base_img, (21, 21), 0)
        
        # 2. Low Light (Underexposed): Simulates terrible nighttime capture causing contrast collapse
        low_light = cv2.convertScaleAbs(base_img, alpha=0.25, beta=0)
        
        # 3. Noise: Simulates sensor read noise and high ISO grain
        noise_matrix = np.random.normal(0, 40, base_img.shape) # Gaussian noise, sigma=40
        noisy = np.clip(base_img.astype(np.float32) + noise_matrix, 0, 255).astype(np.uint8)
        
        # 4. Partial Occlusion: Simulates physical obstruction (mask, scarf, random pole)
        occluded = base_img.copy()
        h, w = base_img.shape[:2]
        # Drop a black mask over the jaw/mouth area (bottom third of the crop)
        start_y = int(h * 0.6)
        occluded[start_y:h, :] = 0
        
        # Convert RGB to BGR for standard cv2.imwrite compatibility
        imgs = {
            'original': cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR),
            'blurred': cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR),
            'low_light': cv2.cvtColor(low_light, cv2.COLOR_RGB2BGR),
            'noisy': cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
            'occluded': cv2.cvtColor(occluded, cv2.COLOR_RGB2BGR)
        }
        return imgs

    def process_and_save(self, limit: int = 200):
        """Processes the dataset and saves the generated augmentations to strictly partitioned folders."""
        images, targets, target_names = self.download_lfw()
        
        # For experimental brevity, limit processing to N images if testing locally
        process_count = min(len(images), limit)
        print(f"Applying CCTV topologies to {process_count} images...")
        
        for i in range(process_count):
            img_normalized = images[i]
            person_name = target_names[targets[i]].replace(' ', '_')
            
            degraded_variants = self.apply_cctv_degradations(img_normalized)
            
            for deg_type, cv_img in degraded_variants.items():
                filename = f"{person_name}_{i:04d}.jpg"
                save_path = os.path.join(self.paths[deg_type], filename)
                cv2.imwrite(save_path, cv_img)

if __name__ == "__main__":
    preparer = DatasetPreparer(output_dir="experiments/dataset")
    # Limiting to 50 for quick validation sequence, full dataset takes longer
    preparer.process_and_save(limit=50)
    print("Dataset generation finalized into /experiments/dataset/")
