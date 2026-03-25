import sys
from pathlib import Path

# Ensure basicsr is in path
current_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(current_path / "NullSpaceDiff"))

from models.multi_fftlayer_diff import MultiFFTLayer_diff as SVDeconvLayer_diff
from models.multi_fftlayer import MultiFFTLayer as SVDeconvLayer

from models.fftlayer import FFTLayer
from models.fftlayer_diff import FFTLayer_diff
from models.unet_128 import Unet as Unet_128
from models.unet import UNet270480 as Unet_diff

# Import MSRResNet (Lightweight GAN Generator)
try:
    from basicsr.archs.srresnet_arch import MSRResNet
except ImportError:
    print("Warning: Could not import MSRResNet.")
    MSRResNet = None

# Import SwinIR (Heavier option, if needed)
try:
    from basicsr.archs.swinir_arch import SwinIR
except ImportError:
    SwinIR = None

def get_inversion_and_channels(args):
    is_svd = "svd" in args.exp_name
    is_diff = "diff" in args.exp_name

    if is_svd and not is_diff:
        return SVDeconvLayer, 4 if args.load_raw else 3
    elif is_svd and is_diff:
        return SVDeconvLayer_diff, 6
    elif not is_diff:
        return FFTLayer, 3
    else:
        return FFTLayer_diff, 3

def model(args):
    Inversion, in_c = get_inversion_and_channels(args)

    # 1. Standard U-Net (Original Stage 1)
    if args.model == "unet-128-pixelshuffle-invert":
        return Unet_128(args, in_c=in_c), Inversion(args)
    
    elif args.model == "UNet270480":
        return Unet_diff(args, in_c=in_c), Inversion(args)
        
    # 2. MSRResNet (Lightweight GAN-style Generator)
    elif args.model == "srresnet":
        if MSRResNet is None:
            raise ImportError("MSRResNet module not found.")
            
        # Calculate dimensions for PixelShuffle compatibility
        # Input to G: [B, 12, H/2, W/2] (if RGB and ratio=2)
        # Output of G: [B, 12, H/2, W/2]
        ratio = args.pixelshuffle_ratio
        channels = in_c * (ratio ** 2) # 3 * 4 = 12
        
        return MSRResNet(
            num_in_ch=channels, 
            num_out_ch=channels, 
            num_feat=64,       # Standard channel width
            num_block=16,      # 16 Residual blocks (Good balance of speed/quality)
            upscale=1          # No internal upsampling (we use PixelShuffle outside)
        ), Inversion(args)

    # 3. SwinIR (Heavy Transformer)
    elif args.model == "swinir":
        if SwinIR is None:
            raise ImportError("SwinIR module not found.")
            
        actual_in_chans = in_c * (args.pixelshuffle_ratio ** 2)

        return SwinIR(
            upscale=1,
            in_chans=actual_in_chans,
            img_size=64,
            window_size=8,
            img_range=1.,
            depths=[4, 4, 4, 4], 
            embed_dim=48, 
            num_heads=[4, 4, 4, 4],
            mlp_ratio=2, 
            upsampler='',
            resi_connection='1conv'
        ), Inversion(args)