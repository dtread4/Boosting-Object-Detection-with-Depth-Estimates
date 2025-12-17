# David Treadwell
#
# Use this script to check that your PyCOCO download was correct

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import skimage.io as io
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_path', required=True, type=str,
                        help="Put the path to the folder called 'coco' that contains the 'images' and 'annotations' folders." \
                        "Example: './coco' should have './coco/annotations' and '.coco/images' structure.")
    parser.add_argument('--data_type', type=str, default='train2017', choices=['train2017', 'val2017'],
                        help="The data split (train or test) to check with.")
    parser.add_argument('--image_index', type=int, default=0,
                        help="The index of the image to view. No safety checks for out of bounds! Default is 0." \
                        "Note that this may not correspond to the same index as the file names appear in due to the" \
                        "pycoco API's method for creating indexes.")
    args = parser.parse_args()

    # Define the data directory where you downloaded the COCO dataset
    annFile = os.path.join(args.coco_path, 'annotations', f'annotations_trainval2017', 'annotations', 
                           f'instances_{args.data_type}.json')
    imgPath = os.path.join(args.coco_path, 'images', args.data_type)

    # Load COCO with the pycoco API
    coco = COCO(annFile)
    img_ids = coco.getImgIds()

    # Load the image
    img_id = img_ids[args.image_index]
    img_info = coco.loadImgs(img_id)[0]
    print(coco.loadImgs(img_id))
    image_file_path = os.path.join(imgPath, img_info['file_name'])

    # Display the image
    I = io.imread(image_file_path)
    plt.imshow(I)
    plt.axis('off')
    plt.title(f"Image Name: {img_info['file_name']}")
    plt.show()