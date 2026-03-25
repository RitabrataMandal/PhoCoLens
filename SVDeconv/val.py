"""
Val Script for Phase/Amp mask
"""
# Libraries
from sacred import Experiment
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import logging
import cv2
from pathlib import Path
# Torch Libs
import torch
from torch.nn import functional as F
import time
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from utils.tupperware import tupperware
from models import get_model
from metrics import PSNR
from config import initialise
from skimage.metrics import structural_similarity as ssim
from utils.model_serialization import load_state_dict

# LPIPS
import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Train helpers
from utils.ops import rggb_2_rgb, unpixel_shuffle
from utils.train_helper import load_models, AvgLoss_with_dict

# Experiment, add any observers by command line
ex = Experiment("val")
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")


@ex.config
def config():
    gain = 1.0
    tag = "384"


@ex.automain
def main(_run):
    args = tupperware(_run.config)
    args.batch_size = 1

    # Set device
    device = args.device

    # ADMM or not
    interm_name = "fft" 

    # Get data
    data = get_dataloaders(args)

    # Model
    G, FFT = get_model.model(args)

    # --- PATH CONFIGURATION ---
    ckpt_dir = args.ckpt_dir 
    
    model_gen_path = ckpt_dir / "model_latest.pth"
    model_fft_path = ckpt_dir / "FFT_latest.pth"
    
    print(f"Looking for checkpoints at: {ckpt_dir}")
    
    # --- MANUAL LOADING (Robust) ---
    global_step = 0 # Default initialization
    start_epoch = 0
    
    if model_gen_path.exists() and model_fft_path.exists():
        logging.info(f"Loading model from {model_gen_path}")
        gen_ckpt = torch.load(model_gen_path, map_location=torch.device("cpu"))
        fft_ckpt = torch.load(model_fft_path, map_location=torch.device("cpu"))

        load_state_dict(G, gen_ckpt["state_dict"])
        load_state_dict(FFT, fft_ckpt["state_dict"])
        
        # Extract metadata if available
        if 'global_step' in gen_ckpt:
            global_step = gen_ckpt['global_step']
        if 'epoch' in gen_ckpt:
            start_epoch = gen_ckpt['epoch']
            
    else:
        logging.error(f"Checkpoints not found at {model_gen_path}")
        # Don't return, just warn (allows testing untrained models if needed)
        # return 

    G = G.to(device)
    FFT = FFT.to(device)

    # LPIPS Criterion
    lpips_criterion = loss_fn_alex.to(device)

    _metrics_dict = {
        "PSNR": 0.0,
        "LPIPS_01": 0.0,
        "LPIPS_11": 0.0,
        "SSIM": 0.0,
        "Time": 0.0,
    }
    avg_metrics = AvgLoss_with_dict(loss_dict=_metrics_dict, args=args)

    # Switch to train loader if requested
    if args.val_train:
        logging.info("Generating output for TRAIN set.")
        loader = data.train_loader
        # Adjust output path
        val_path = args.output_dir / "train_output"
    else:
        logging.info("Generating output for VAL set.")
        loader = data.val_loader
        val_path = args.output_dir / "val_output"
        
    logging.info(f"Saving results to: {val_path}")
    val_path.mkdir(exist_ok=True, parents=True)
  
    avg_metrics.reset()
    
    # Calculate total batches for tqdm
    # loader might be None if dataset is empty, though unlikely here
    total_batches = len(loader) if loader else 0
    pbar = tqdm(range(total_batches * args.batch_size), dynamic_ncols=True)

    if args.device == "cuda:0":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        start = end = 0

    acc_time = 0.0
    with torch.no_grad():
        G.eval()
        FFT.eval()
        for i, batch in enumerate(loader):
            metrics_dict = defaultdict(float)

            # Robust unpacking for 2 or 3 items
            if len(batch) == 3:
                source, target, filename = batch
                target = target.to(device)
            else:
                source, filename = batch
                target = None # Handle missing target gracefully

            source = source.to(device)

            if args.device == "cuda:0" and i:
                start.record()
            start_time = time.time()
            
            fft_output = FFT(source)

            # Unpixelshuffle
            fft_unpixel_shuffled = unpixel_shuffle(fft_output, args.pixelshuffle_ratio)
            output_unpixel_shuffled = G(fft_unpixel_shuffled)

            output = F.pixel_shuffle(output_unpixel_shuffled, args.pixelshuffle_ratio)
            
            acc_time += time.time() - start_time

            if args.device == "cuda:0" and i:
                end.record()
                torch.cuda.synchronize()
                metrics_dict["Time"] = start.elapsed_time(end)
            else:
                metrics_dict["Time"] = 0.0

            # Metrics
            if target is not None:
                metrics_dict["PSNR"] += PSNR(output, target).item()
                metrics_dict["LPIPS_01"] += lpips_criterion(
                    output.mul(0.5).add(0.5), target.mul(0.5).add(0.5)
                ).mean().item()
                metrics_dict["LPIPS_11"] += lpips_criterion(output, target).mean().item()

            for e in range(args.batch_size):
                # Visuals
                output_numpy = (
                    output[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                
                # Filename now comes as "scanX/image.png"
                full_name = str(filename[e])
                
                # 1. Extract Scene and Image Name
                if "/" in full_name:
                    scan_folder, img_name = full_name.split("/")[-2:] 
                else:
                    # Fallback if no folder structure found
                    scan_folder = "unknown_scan"
                    img_name = full_name
                
                img_name = img_name.replace(".JPEG", ".png") # Ensure PNG extension
                
                # 2. Create Scene Directory inside Output Dir
                # Result: output/dtu/.../val_output/scan2_train/
                scene_save_path = val_path / scan_folder
                scene_save_path.mkdir(exist_ok=True, parents=True)
                
                path_output = scene_save_path / img_name
                
                # 3. Save (Clipped & Casted)
                cv2.imwrite(
                    str(path_output), (np.clip(output_numpy, 0, 1)[:, :, ::-1] * 255.0).astype(np.uint8)
                )
                
                # SSIM Calculation (No changes needed here)
                if target is not None:
                    target_numpy = (
                        target[e].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                    )
                    metrics_dict["SSIM"] += ssim(
                        target_numpy, output_numpy, multichannel=True, data_range=1.0, channel_axis=-1
                    )

            if target is not None:
                metrics_dict["SSIM"] = metrics_dict["SSIM"] / args.batch_size
                avg_metrics += metrics_dict

            pbar.update(args.batch_size)
            pbar.set_description(
                f"Val Step: {global_step}| PSNR: {avg_metrics.loss_dict['PSNR']:.3f} | SSIM: {avg_metrics.loss_dict['SSIM']:.3f}"
            )

        with open(val_path / "metrics.txt", "w") as f:
            L = [
                f"exp_name:{args.exp_name} trained for {start_epoch} epochs\n",
                f"Inference mode {args.inference_mode}\n",
                "Metrics \n\n",
            ]
            L = L + [f"{k}:{v}\n" for k, v in avg_metrics.loss_dict.items()]
            f.writelines(L)