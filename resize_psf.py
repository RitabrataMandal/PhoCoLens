import numpy as np
import cv2
import os
from pathlib import Path

def generate_or_resize_psf(
    target_h=384, 
    target_w=640, 
    source_psf_path="/mnt/data/Ritabrata/MVSGaussian/mvs_training/psf.npy",
    save_path="/mnt/data/Ritabrata/PhoCoLens/psf_640x384.npy"
):
    """
    Generates a PSF matching the target dimensions (H, W).
    """
    print(f"Target Shape: ({target_h}, {target_w})")
    
    # 1. Try to load existing PSF
    if os.path.exists(source_psf_path):
        print(f"Loading source PSF from: {source_psf_path}")
        try:
            psf = np.load(source_psf_path).astype(np.float32)
            print(f"Original Shape: {psf.shape}")
            
            # Check if resize is needed
            if psf.shape != (target_h, target_w):
                # cv2.resize expects (width, height)
                psf = cv2.resize(psf, (target_w, target_h), interpolation=cv2.INTER_AREA)
                print(f"Resized to: {psf.shape}")
            else:
                print("Source PSF already matches target dimensions.")
                
        except Exception as e:
            print(f"Error loading source PSF: {e}")
            psf = None
    else:
        print(f"Source PSF not found at {source_psf_path}")
        psf = None

    # 2. Fallback: Generate Gaussian PSF if loading failed
    if psf is None:
        print("Generating synthetic Gaussian PSF...")
        # Create a grid
        sigma = 4.0  # Spread of the Gaussian
        
        # We generate a small gaussian and pad it, OR generate a full-field gaussian
        # For lensless simulation, usually the PSF is defined over the whole sensor.
        # But a simple gaussian is usually centered and small.
        # Let's create a full-size black image with a centered gaussian.
        
        psf = np.zeros((target_h, target_w), dtype=np.float32)
        
        # Define gaussian kernel size (odd)
        k_size = 31
        ax = np.arange(-(k_size // 2), k_size // 2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Place in center of the 512x640 frame
        cy, cx = target_h // 2, target_w // 2
        h_k, w_k = kernel.shape
        
        # Calculate start/end indices
        y1 = cy - h_k // 2
        y2 = y1 + h_k
        x1 = cx - w_k // 2
        x2 = x1 + w_k
        
        psf[y1:y2, x1:x2] = kernel

    # 3. Normalize (Energy Conservation)
    # The sum of the PSF must be 1.0 so it doesn't brighten/darken the image arbitrarily
    psf /= np.sum(psf)
    print(f"PSF Sum after normalization: {np.sum(psf):.6f}")
    
    # 4. Save
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    np.save(save_path, psf)
    print(f"✅ Saved PSF to: {save_path}")

if __name__ == "__main__":
    # Adjust paths as needed for your specific setup
    generate_or_resize_psf(
        target_h=384,
        target_w=640,
        source_psf_path="/mnt/data/Ritabrata/MVSGaussian/mvs_training/psf.npy",
        save_path="/mnt/data/Ritabrata/PhoCoLens/psf_640x384.npy"
    )