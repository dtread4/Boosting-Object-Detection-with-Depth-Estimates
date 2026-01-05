# David Treadwell
#
# Use this script to check that your DepthAnywhere3 installation was correct
# Adapted from the official DA3's HuggingFace page: https://huggingface.co/depth-anything/DA3-BASE
# and the da3 notebook on the official repo: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/notebooks/da3.ipynb

import torch
import argparse

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True, type=str,
                        help="The image path to get the depth map for")
    args = parser.parse_args()

    # Load model from Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnything3.from_pretrained("depth-anything/da3-small")
    model = model.to(device=device)

    # Run inference on images
    images = [args.image_path]  # List of image paths, PIL Images, or numpy arrays
    prediction = model.inference(
        images
    )

    # Access results
    print(prediction.depth.shape)        # Depth maps: [N (image index), H, W] float32

    # Display depth map
    fig, axes = plt.subplots(2, 1)

    # Display original image to left of depth map
    axes[0].imshow(prediction.processed_images[0])
    axes[0].set_title("Original image")
    axes[0].axis('off')

    # Display the depth map
    axes[1].imshow(visualize_depth(prediction.depth[0], cmap='Spectral'))
    axes[1].set_title("Estimated depth map")
    axes[1].axis('off')

    plt.show()

