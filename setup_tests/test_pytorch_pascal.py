# David Treadwell
#
# Use this script to check that your PASCAL VOC download was correct

import argparse
import torchvision
import matplotlib.pyplot as plt
import numpy as np

    # --- Configuration ---
if __name__ == "__main__":
    # Use command line to get path to directory
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', required=True, type=str,
                        help="Put the path to the folder containing the VOCdevkit directory." \
                        "Example: './data/VOCdevkit', put './data'")
    parser.add_argument('--year', type=str, default='2007', choices=['2007', '2012'],
                        help="The dataset year to use. Either 2007 or 2012. Default is 2007.")
    parser.add_argument('--image_set', type=str, default='train', choices=['train', 'val', 'test'],
                        help="The dataset split to use. Either train, val, or test. Default is train." \
                        "Note that 2012 does not have a test split!")
    parser.add_argument('--image_index', type=int, default=0,
                        help="The index of the image to view. No safety checks for out of bounds! Default is 0.")
    args = parser.parse_args()

    # Extract command line arguments
    data_root = args.voc_path
    voc_year = args.year
    image_set = args.image_set
    item_index = args.image_index   

    # Ensure the root directory exists
    print(f"Loading dataset from: {data_root}, Year: {voc_year}, Set: {image_set}")

    # 1. Initialize the dataset
    # We use VOCDetection even if just displaying an image, as it handles the main dataset structure
    dataset = torchvision.datasets.VOCDetection(
        root=data_root,
        year=voc_year,
        image_set=image_set,
        download=False # Set to False as the data is already downloaded
    )

    print(f"Dataset loaded successfully with {len(dataset)} items.")

    # 2. Retrieve a specific image and its target (annotations)
    image_pil, target = dataset[item_index]

    # 3. Display the image and some metadata

    # Convert PIL image to a format suitable for matplotlib
    image_np = np.array(image_pil)

    # Display information about the image's annotations (e.g., objects present)
    print("\n--- Annotation Details (Target Dictionary) ---")
    img_id = target['annotation']['filename']
    img_size = target['annotation']['size']
    objects = target['annotation']['object']

    print(f"Filename: {img_id}")
    print(f"Size (W/H/D): {img_size['width']}x{img_size['height']}x{img_size['depth']}")
    print(f"Number of objects detected: {len(objects)}")

    if len(objects) > 0:
        print("\nFirst 3 objects:")
        # 'objects' can be a dictionary if only one object exists, or a list of dicts.
        if isinstance(objects, dict):
                print(f"- Class: {objects['name']}, Bbox: {objects['bndbox']}")
        else:
            for obj in objects[:3]:
                print(f"- Class: {obj['name']}, Bbox: {obj['bndbox']}")

    plt.figure(figsize=(10, 5))
    plt.imshow(image_np)
    plt.title(f"Image Index {item_index}, name: {img_id}\nFrom PASCAL VOC {voc_year} {image_set} set")
    plt.axis('off')
    plt.show()