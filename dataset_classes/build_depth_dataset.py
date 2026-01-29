# David Treadwell

# The point of this file is to direct it to a folder of images (like VOC2007/JPEGImages)
# The script will create a new directory called "DEPTHImages" in the parent directory (like VOC2007)
# There will be a subdirectory for the specific depth model used
#
# DO NOTE THAT YOU MAY GET SLIGHTLY DIFFERENT OUTPUTS WITH PRE-GENERATED MASKS THAN RUNTIME MASKS BECAUSE
# OF LIMITATIONS AROUND BATCHING AND RESIZING

import argparse
from tqdm import tqdm
import os
import sys

from PIL import Image
import numpy as np

from dataset_classes.depth_model import DepthModel
from dataset_classes.utils import get_depth_outputs_path
from training_code.utils.utility import set_reproducability_settings, set_device


def create_output_directory(image_dir, model_name):
    """
    Creates the output directory for a given model and input image directory

    Args:
        image_dir: Input directory of images to create masks for
        model_name: The depth model's name

    Returns:
        The output directory for the images
    """
    output_dir = get_depth_outputs_path(image_dir, model_name)

    # Check if output path already exists
    if os.path.exists(output_dir):
        print(f"Warning! Output path {output_dir} already exists! Current items may be overwritten!")
        in_val = input("Continue? [y/n]")

        if in_val.lower() == 'n':
            print("Exiting...")
            sys.exit(1)

    print()
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def create_depth_maps(depth_model, image_dir, output_dir):
    # For each image in the input directory, calculate the depth mask and save it to the output directory
    # TODO could speed this up by batching, but since it is theoretically a one-time cost (per model), haven't done that yet
    image_names = os.listdir(image_dir)
    for image_name in tqdm(image_names, desc=f'Creating depth maps for all images in [{image_dir}]'):
        img_path = os.path.join(image_dir, image_name)
        img = Image.open(img_path).convert("RGB")
        depth_mask = depth_model.calculate_depth_map([img])[0]
        output_name = image_name.rsplit('.', 1)[0] + ".npy"  # Find last period, get everything before it, add .npy
        np.save(os.path.join(output_dir, output_name), depth_mask)


def main(args):
    # Set reproducibility settings first (default to maximize reproducibility)
    set_reproducability_settings(random_seed=42, use_determinism=True)

    # Set device
    device_type, device = set_device()

    # Convert input directory to a path
    image_dir = os.path.normpath(args.image_dir)

    # Create the output directory for the depth masks
    output_dir = create_output_directory(image_dir, args.depth_model_name)
    
    # Load the depth model (if any; will be None if not)
    depth_model = DepthModel.initialize_depth_model(
        model_name=args.depth_model_name,
        device=device
    )

    # Generate the masks
    create_depth_maps(depth_model, image_dir, output_dir)


if __name__ == "__main__":
    # Parse path to config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True, type=str, 
                        help="Path to the image directory you want to create depth maps for")
    parser.add_argument('--depth_model_name', required=True, type=str,
                        help='The name of the depth model to create depth maps for')

    # Read YAML path
    args = parser.parse_args()

    main(args)
