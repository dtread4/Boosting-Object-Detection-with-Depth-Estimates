# David Treadwell

import numpy as np
from PIL import Image
import random

import torch


def tensor_to_pil(tensor_images):
    """
    Converts a batch (list) of tensor images to PIL images

    Args:
        tensor_images: The batch of tensor format images (C, H, W)

    Returns:
        A list of PIL images
    """
    pil_images = []
    for img in tensor_images:
        # Convert (C, H, W) -> (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        
        # If float [0,1], convert to uint8 [0,255]
        if img_np.dtype == np.float32 or img_np.dtype == np.float64:
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

        pil_images.append(Image.fromarray(img_np))
    
    return pil_images
    

def set_reproducability_settings(random_seed, use_determinism, verbose=True):
    """
    Sets backend random seeds and Torch determinism to ensure experiments are reproducible

    Args:
        random_seed: The random seed to use for training
        use_determinism: A boolean whether to use or not to use determinism
        verbose: Whether to print statement about reproducibility settings
    """
    # Set seeds
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Set rough determinism (note it will not be truly deterministic for Faster R-CNN, but it's better than nothing)
    # TODO maybe look into this more?
    torch.backends.cudnn.deterministic = use_determinism

    if verbose:
        print(f"Using random seed: [{random_seed}] and {"[NOT USING]" if not use_determinism else "[USING]"} determinism")


def set_device(device_type='auto', verbose=True):
    """
    Sets the global device used for training and testing

    Args:
        device_type: The name of the device type (auto, cpu, cuda)
        verbose: Whether to print statement about reproducibility settings

    Returns:
        The device used for training and testing
    """
    if device_type == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f"Training on device [{device_type}]\n")
    return device_type, torch.device(device_type)