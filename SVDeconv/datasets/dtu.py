import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from pathlib import Path
import logging
import random

class DTUDataset(Dataset):
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
        self.image_dir = Path(args.image_dir)
        
        # --- LOAD SCENE LISTS FROM TEXT FILES ---
        # Assuming text files are in the root of image_dir or a specific subdir
        # Adjust path as needed. Here assuming: data/dtu/dtu_train_all.txt
        train_list_path = "/mnt/data/Ritabrata/MVSGaussian/data/mvsgs/dtu_train_all.txt"
        test_list_path = "/mnt/data/Ritabrata/MVSGaussian/data/mvsgs/dtu_val_all.txt"
        
        self.train_scenes = [scene + "_train" for scene in self._read_list(train_list_path)]
        self.test_scenes  = [scene + "_train"  for scene in self._read_list(test_list_path)]


        
        # Paths
        self.source_root = self.image_dir / "Lenseless"
        
        # Determine Target Root
        if mode == 'train':
            self.target_root = self.image_dir / "Rectified" # Target: Wiener
        else:
            self.target_root = self.image_dir / "Rectified" # Target: GT for Val/Test
            
        # Load Dataset based on Mode
        self.dataset_pairs = self._scan_dataset()
        
        logging.info(f"DTU {mode.capitalize()} Set | Found {len(self.dataset_pairs)} pairs.")

    def _read_list(self, path):
        """Helper to read scene names from a text file."""
        # FIX: Ensure 'path' is a Path object, even if passed as a string
        path = Path(path)
        
        if not path.exists():
            logging.warning(f"Scene list not found at {path}. Returning empty list.")
            return []
            
        with open(path, 'r') as f:
            # Read lines and strip whitespace
            scenes = [line.strip() for line in f.readlines() if line.strip()]
            
        return scenes

    def _scan_dataset(self):
        all_files = []
        
        if not self.source_root.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_root}")

        # Select scenes based on mode
        if self.mode == 'train':
            target_scenes = self.train_scenes
        else: # val or test
            target_scenes = self.test_scenes

        if not target_scenes:
            logging.warning(f"No scenes found for mode {self.mode}. Check your text files.")

        # Walk through ONLY the selected scan folders
        for scan_name in target_scenes:
            scan_folder = self.source_root / scan_name
            
            if not scan_folder.exists():
                logging.warning(f"Scene {scan_name} not found in {self.source_root}. Skipping.")
                continue
                
            images = sorted(list(scan_folder.glob("*.png")))
            
            for img_path in images:
                img_name = img_path.name
                
                # Verify target existence
                if self.mode == 'train':
                    target_path = self.image_dir / "Rectified" / scan_name / img_name
                else:
                    target_path = self.image_dir / "Rectified" / scan_name / img_name
                
                if target_path.exists():
                    all_files.append((scan_name, img_name))
        
        return all_files

    def __getitem__(self, index):
        scan_name, img_name = self.dataset_pairs[index]
        
        # Full relative path for folder creation in validation
        relative_path = f"{scan_name}/{img_name}"
        
        source_path = self.source_root / scan_name / img_name
        target_path = self.target_root / scan_name / img_name

        # Load Source
        source_bgr = cv2.imread(str(source_path))
        if source_bgr is None:
            logging.error(f"Failed source: {source_path}")
            source = torch.zeros((3, self.args.image_height, self.args.image_width))
        else:
            source = source_bgr[:, :, ::-1].astype(np.float32) / 255.0
            source = (source - 0.5) * 2.0
            source = torch.from_numpy(np.transpose(source, (2, 0, 1)))

        # Load Target (Always load for metric calc, even in Test/Val)
        target_bgr = cv2.imread(str(target_path))
        if target_bgr is None:
            # If target missing (rare if _scan_dataset works), create dummy
            target = torch.zeros((3, self.args.image_height, self.args.image_width))
        else:
            target = target_bgr[:, :, ::-1].astype(np.float32) / 255.0
            target = (target - 0.5) * 2.0
            target = torch.from_numpy(np.transpose(target, (2, 0, 1)))

        # Always return 3 items: Source, Target (for PSNR), Path (for saving)
        return source.float(), target.float(), relative_path

    def __len__(self):
        return len(self.dataset_pairs)