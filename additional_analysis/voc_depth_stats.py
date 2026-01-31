# David Treadwell
#
# This file is meant to calculate the mean and standard deviation of precomputed depth masks,
# given a specific depth model, for the VOC dataset
# Creates an output directory in the ./additional_analysis directory 
# Yes, REQUIRES a config file like for training
#
# ASSUMES DEPTH MASKS HAVE ALREADY BEEN PRECOMPUTED


import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf
import csv

import numpy as np
import matplotlib.pyplot as plt

from training_code.utils.dataloader import build_dataloader
from dataset_classes.depth_model import DepthModel
from training_code.utils.utility import get_depth_masks


def make_out_dir_model(model_name):
    """
    Create the output directory for this model's statistics

    Args:
        model_name: The name of the model used for depth statistics

    Returns:
        Path to the output directory
    """
    out_dir_stats = os.path.join('additional_analysis', 'depth_stats')
    os.makedirs(out_dir_stats, exist_ok=True)
    out_dir_voc = os.path.join(out_dir_stats, 'VOC_2007_2012')
    os.makedirs(out_dir_voc, exist_ok=True)
    out_dir_model = os.path.join(out_dir_voc, model_name)
    os.makedirs(out_dir_model, exist_ok=True)
    return out_dir_model


def save_to_csv(mean, std, out_dir):
    """
    Saves the mean and standard deviation to a CSV

    Args:
        mean: The mean value
        std: The standard deviation value
        out_dir: The directory to save the CSV to
    """
    out_path = os.path.join(out_dir, 'mean_std.csv')

    with open(out_path, 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["mean", "standard deviation"])
        writer.writerow([mean, std])
    
    print(f"Saved mean and standard deviation for {config.DEPTH_INFO.DEPTH_MODEL} precomputed masks")
    print(f"Output path: {out_path}")


def save_histogram(bins, hist_counts, out_dir):
    """
    Saves a histogram

    Args:
        bins: What bins/buckets values fall into
        hist_counts: The actual counts within the bins
        out_dir: The output directory to save the histogram to
    """
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plt.bar(
        bin_centers,
        hist_counts,
        width=0.05,
        align="center",
        edgecolor="black",
        linewidth=1.0        
    )
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title("Histogram of values in depth masks")

    out_path = os.path.join(out_dir, "hist_counts.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved histogram to {out_path}")


def main(config):
     # Set this larger to make the process go faster - it doesn't matter here, and these are small arrays
    config.SOLVER.BATCH_SIZE = 128 

    # Get dataloader and depth model
    dataloader = build_dataloader(config, "TRAIN")
    print(f"Calculating depth masks' statistics for {len(dataloader.dataset)} masks in train split\n")

    depth_model = DepthModel.initialize_depth_model(
        model_name=config.DEPTH_INFO.DEPTH_MODEL,
        precomputed_depth=True,
        device=None
    )

    # Output directory creation
    out_dir = make_out_dir_model(config.DEPTH_INFO.DEPTH_MODEL)

    # running totals for statistics
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    # Histogram setup: 20 bins, width 0.05
    bins = np.arange(0, 1.0001, 0.05)  # 0.0 → 1.0 inclusive
    hist_counts = np.zeros(len(bins) - 1, dtype=np.int64)

    desc = f'Calculating mean and standard deviation for {config.DEPTH_INFO.DEPTH_MODEL} depth masks'
    for images, targets, image_paths in tqdm(dataloader, desc=desc):
        # Load depth masks
        depth_masks = get_depth_masks(config.DEPTH_INFO.PRECOMPUTED_DEPTH, depth_model, images, image_paths)
        
        # Add to total
        for mask in depth_masks:
            total_sum += mask.sum()
            total_sq_sum += (mask * mask).sum()
            total_count += mask.size

            # histogram
            counts, _ = np.histogram(mask, bins=bins)
            hist_counts += counts
    
    # Compute statistics and save
    mean = round(total_sum / total_count, 6)
    std = round(np.sqrt(total_sq_sum / total_count - (mean * mean)), 6)

    save_to_csv(mean, std, out_dir)
    save_histogram(bins, hist_counts, out_dir)
    

if __name__ == "__main__":
     # Parse path to config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, 
                        help="Path to configuration YAML file containing setup information.")
    args = parser.parse_args()

    # Read YAML path
    config = OmegaConf.load(args.config_path)

    main(config)