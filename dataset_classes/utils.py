# David Treadwell

import os

DEPTH_DIR = 'DEPTHImages'


def get_depth_outputs_path(image_dir, depth_model_name):
    """
    Gets the output directory name for a given model and input image directory

    Args:
        image_dir: Input directory of images to create masks for
        depth_model_name: The depth model's name

    Returns:
        The directory path for the depth model
    """
    parent_dir = os.path.dirname(image_dir)
    child_dir = os.path.basename(image_dir)
    output_dir = os.path.join(parent_dir, DEPTH_DIR, child_dir, depth_model_name)
    return output_dir