# David Treadwell

import torchvision.transforms as T

def get_transforms(train=True):
    """
    Returns the necessary transformations to apply to images when initially loaded

    Args:
        train: Flag to only applies augmentation-based transformations to images in the training set. Defaults to True.

    Returns:
        The set of transformations to use, ready for use by the dataset
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)