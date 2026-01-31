# David Treadwell

from torch.utils.data import DataLoader

from dataset_classes.VOC_dataset import VOCDataset


def collate_fn(batch):
    """
    Collation function to properly handle variable-sized targets (since the number of bounding boxes may differ per image)

    Args:
        batch: The batch of images/targets

    Returns:
        A tuple containing images in a list and targets in a list of dicts
    """
    return tuple(zip(*batch))


def build_dataloader(config, split):
    """
    Builds a dataloader for a single train/val/test split

    Args:
        config: The configurations to build the dataset/dataloader with
        split: Which split to build for

    Returns:
        _description_
    """
    # TODO update this to select a dataset when using COCO
    dataset = VOCDataset(
        root=config.DATASETS.ROOT,
        years=config.DATASETS[split].YEARS,
        split=split
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.SOLVER.BATCH_SIZE if split in ["TRAIN", "TRAINVAL"] else 1,
        shuffle=(split.upper() == "TRAIN"),
        num_workers=config.DATALOADER.NUM_WORKERS,
        collate_fn=collate_fn
    )
    return dataloader