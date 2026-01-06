# David Treadwell
#
# Use this script to check that your DepthAnywhere3 installation was correct.
# It also shows how the process will calculate depth maps (with resizing) in the main training/inference scripts
#
# Adapted from the official DA3's HuggingFace page: https://huggingface.co/depth-anything/DA3-BASE
# and the da3 notebook on the official repo: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/notebooks/da3.ipynb

import torch
import argparse

from PIL import Image
import cv2

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths', required=True, type=str, nargs='+',
                        help="The image paths to get the depth map for")
    args = parser.parse_args()

    image_paths = args.image_paths
    images = [Image.open(img_pth).convert("RGB") for img_pth in image_paths]
    min_size = min(min(img.size) for img in images)
    print("Min image dimension:", min_size)
    images_resized = [img.resize((min_size, min_size), Image.Resampling.BILINEAR) for img in images]

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3-small")
    model = model.to(device=device)

    # Run inference on images (this is done here how it will be in the main program)
    prediction = model.inference(
        images_resized,
        process_res=min_size
    )

    # Display depth map
    fig, axes = plt.subplots(2, len(images))

    if len(images) == 1:
        axes = axes.reshape(2, 1)

    # Display original image to left of depth map
    for i in range(len(images)):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title("Original image")
        axes[0, i].axis('off')

        # Display the depth map
        axes[1, i].imshow(visualize_depth(cv2.resize(prediction.depth[i], images[i].size, cv2.INTER_LINEAR), cmap="Spectral"))
        axes[1, i].set_title("Estimated depth map")
        axes[1, i].axis('off')

    plt.show()

