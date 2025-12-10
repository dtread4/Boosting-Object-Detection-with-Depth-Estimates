# David Treadwell
# dtread4@live.com
# 
# This file splits the PASCAL VOC 2007 and 2012 datasets into their respective train/val/test sets
# This script is required preprocessing because the best source for the datasets combined them
# Unzip the two directories and then use this script on each of them separately (will separate them in place)
# Source link: https://www.kaggle.com/datasets/vijayabhaskar96/pascal-voc-2007-and-2012

import os
import shutil
import argparse
from tqdm import tqdm


def main(args):
    # Get the root path to the dataset from args
    root = args.voc_path

    # Add the test split if using the 2007 set only, as there is no test split in the 2012 dataset
    splits = ["train", "val"]
    if '2007' in root:
        splits.append('test')

    # Copy each file to the split output directory
    for split in splits:
        out_img = os.path.join(root, "SplitJPEG", split)
        out_ann = os.path.join(root, "SplitAnn", split)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_ann, exist_ok=True)

        # Get ID's of images in split (file names, not indexes)
        split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file) as f:
            ids = [x.strip() for x in f]

        # Copy the images/annotations in the split to their own output directory
        for img_id in tqdm(ids, desc=f'Moving files in {split} split'):
            jpeg = f"{img_id}.jpg"
            xml = f"{img_id}.xml"

            shutil.copy(os.path.join(root, "JPEGImages", jpeg), out_img)
            shutil.copy(os.path.join(root, "Annotations", xml), out_ann)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--voc_path', required=True,
                        help='Path to the directory containing the the VOC files. ' \
                        'This directory should be used for just one of the 2007 or 2012 datasets, not both.' \
                        'This also means that this should script should be used separately for both datasets.'
                        )
    
    args = parser.parse_args()

    main(args)