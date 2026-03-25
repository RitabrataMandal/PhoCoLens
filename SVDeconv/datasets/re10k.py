import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from pathlib import Path
import logging

class RE10KDataset(Dataset):
    def __init__(self, args, mode='train'):
        """
        Args:
            mode: 'train' or 'test' 
        """
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        super().__init__()
        self.args = args
        self.mode = mode
        
        # Base dataset directory (e.g., path/to/re10k)
        self.image_dir = Path(args.image_dir)
        
        # Determine paths based on the mode (train or test)
        self.source_root = self.image_dir / mode / "lensless" 
        self.target_root = self.image_dir / mode / "normal"
        
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_root}")
            
        # Dynamically find all scene folders directly inside the mode's lensless folder
        self.target_scenes = sorted([d.name for d in self.source_root.iterdir() if d.is_dir()])
        
        if not self.target_scenes:
            logging.warning(f"No scene directories found in {self.source_root}")
            
        # Scan the selected scenes for images
        self.dataset_pairs = self._scan_dataset()
        
        logging.info(f"RE10K {mode.capitalize()} Set | Found {len(self.dataset_pairs)} image pairs across {len(self.target_scenes)} scenes.")

    def _scan_dataset(self):
        all_files = []
        
        for scan_name in self.target_scenes:
            # Source has an "images" folder, Target does NOT
            scan_folder = self.source_root / scan_name / "images"
            target_folder = self.target_root / scan_name  # <-- FIX: Removed / "images"
            
            if not scan_folder.exists():
                logging.warning(f"Images folder not found for scene {scan_name}. Skipping.")
                continue
                
            images = sorted(list(scan_folder.glob("*.png")))
            
            for img_path in images:
                src_img_name = img_path.name
                
                # Check for corresponding target (.jpg or .png)
                tgt_img_name_jpg = img_path.stem + ".jpg"
                tgt_img_name_png = img_path.stem + ".png"
                
                if (target_folder / tgt_img_name_jpg).exists():
                    tgt_img_name = tgt_img_name_jpg
                elif (target_folder / tgt_img_name_png).exists():
                    tgt_img_name = tgt_img_name_png
                else:
                    continue # Target image doesn't exist, skip to prevent loading errors
                
                all_files.append((scan_name, src_img_name, tgt_img_name))
        
        return all_files

    def __getitem__(self, index):
        scan_name, src_img_name, tgt_img_name = self.dataset_pairs[index]
        
        # Relative path is useful if the training loop wants to save output images
        relative_path = f"{scan_name}/{src_img_name}"
        
        source_path = self.source_root / scan_name / "images" / src_img_name
        target_path = self.target_root / scan_name / tgt_img_name

        # --- Load Source (Lensless PNG) ---
        source_bgr = cv2.imread(str(source_path))
        if source_bgr is None:
            logging.error(f"Failed to load source: {source_path}")
            source = torch.zeros((3, self.args.image_height, self.args.image_width))
        else:
            source_bgr = cv2.resize(source_bgr, (self.args.image_width, self.args.image_height), interpolation=cv2.INTER_AREA)
            source = source_bgr[:, :, ::-1].astype(np.float32) / 255.0
            source = (source - 0.5) * 2.0
            source = torch.from_numpy(np.transpose(source, (2, 0, 1)))

        # --- Load Target (Original GT) ---
        target_bgr = cv2.imread(str(target_path))
        if target_bgr is None:
            logging.error(f"Failed to load target: {target_path}")
            target = torch.zeros((3, self.args.image_height, self.args.image_width))
        else:
            # Resize target to match source dimensions if they differ
            if source_bgr is not None and target_bgr.shape[:2] != source_bgr.shape[:2]:
                h, w = source_bgr.shape[:2]
                target_bgr = cv2.resize(target_bgr, (w, h), interpolation=cv2.INTER_AREA)

            target = target_bgr[:, :, ::-1].astype(np.float32) / 255.0
            target = (target - 0.5) * 2.0
            target = torch.from_numpy(np.transpose(target, (2, 0, 1)))

        return source.float(), target.float(), relative_path

    def __len__(self):
        return len(self.dataset_pairs)