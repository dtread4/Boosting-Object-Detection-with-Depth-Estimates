# David Treadwell

import os
import xml.etree.ElementTree as ET

import cv2

import torch
from torch.utils.data import Dataset, ConcatDataset

import torchvision.transforms as T

from pycocotools.coco import COCO


VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

CLASS2IDX = {cls: i + 1 for i, cls in enumerate(VOC_CLASSES)}


class VOCDataset(Dataset):
    """
    Dataset class for managing the combined VOC 2007/2012 datasets
    """
    def __init__(self, root, years, split, transforms=None, depth_model=None):
        """
        Dataset constructor

        Args:
            root: The root directory containing 'VOCdevkit'
            year: list of years, e.g., [2007], [2012], [2007, 2012]
            split: Which split of data (train/val/trainval/test) to manage
            transforms: Transforms to apply to data. Defaults to None.
            depth_model: An already-initialized depth model (on the correct device)
        """
        self.datasets = []

        # Create datasets for each year specified
        for year in years:
            self.datasets.append(
                VOCSingleYearDataset(root, year, split, transforms, depth_model)
            )

        # Concatenate all years into a single dataset
        if len(self.datasets) == 1:
            self.dataset = self.datasets[0]
        else:
            self.dataset = ConcatDataset(self.datasets)

    def __len__(self):
        """
        Returns the length of images managed by the Dataset

        Returns:
            Integer count of images in the specified split
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Gets the specified images and their bbox annotations, and converts to PyTorch tensors

        Args:
            idx: The index of the image to load

        Returns:
            The image and the target annotations as tensors
        """
        return self.dataset[idx]


class VOCSingleYearDataset(Dataset):
    """
    Dataset class for managing either the VOC 2007/2012 data and use in PyTorch Dataloader objects
    """
    def __init__(self, root, year, split, transforms=None, depth_model=None):
        """
        Dataset constructor

        Args:
            root: The root directory containing 'VOCdevkit'
            year: Which year to manage data for
            split: Which split of data (train/val/trainval/test) to manage
            transforms: Transforms to apply to data. Defaults to None.
            depth_model: An already-initialized depth model (on the correct device)
        """
        # Set class attributes
        self.root = root
        self.year = year
        self.split = split
        self.transforms = transforms
        self.depth_model = depth_model

        # Set images that the dataset will manage
        self.image_ids = load_voc_ids(self.root, self.year, self.split)

        self.base_path = os.path.join(
            root, "VOCdevkit", f"VOC{year}"
        )

    def __len__(self):
        """
        Returns the length of images managed by the Dataset

        Returns:
            Integer count of images in the specified split
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Gets the specified images and their bbox annotations, and converts to PyTorch tensors

        Args:
            idx: The index of the image to load

        Returns:
            The image and the target annotations as tensors
        """
        img_id = self.image_ids[idx]

        # Load the image and apply any transformations
        img_path = os.path.join(
            self.base_path, "JPEGImages", f"{img_id}.jpg"
        )
        print(img_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate depth (if it is being used)
        if self.depth_model is not None:
            depth_map_tensor = torch.from_numpy(self.depth_model.calculate_depth_map(img)).unsqueeze(0)
            print(f"Image shape: {img_tensor.shape}")
            print(f"Depth shape: {depth_map_tensor.shape}")

        # Convert the image to a tensor
        img_tensor = T.Compose([T.ToTensor()])(img)

        # Combine the image and the depth mask
        if self.depth_model is not None:
            img_tensor = torch.cat([img_tensor, depth_map_tensor], dim=0)
        
        # TODO remove
        print(img_tensor.shape)
        for c in range(img_tensor.shape[0]):
            print(torch.min(img_tensor[c]), torch.max(img_tensor[c]))
        print()

        if self.transforms:
            img_tensor = self.transforms(img_tensor)  # TODO ultimately will want to separate pre-depth and post-depth estimate transforms
        # TODO test that transformations are happening correctly

        # Load the annotations
        ann_path = os.path.join(
            self.base_path, "Annotations", f"{img_id}.xml"
        )
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # Extract bounding boxes and labels
        boxes, labels = [], []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            boxes.append([
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text)
            ])
            labels.append(CLASS2IDX[obj.find("name").text])

        # Convert to tensors and summarize into a single target dictionary
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        return img_tensor, target


def load_voc_ids(root, year, split):
    """
    Utility function to help loading specified data split for the VOC 2007/2012 files    

    Args:
        root: Root path to dataset
        year: The year of the dataset to load the split for
        split: Which split (train/val/trainval/test) to load

    Returns:
        File names (without extensions) of the images to load as part of the split
    """
    split_file = os.path.join(
        root,
        "VOCdevkit",
        f"VOC{year}",
        "ImageSets",
        "Main",
        f"{split}.txt"
    )
    with open(split_file) as f:
        return [line.strip() for line in f]
    

def voc_to_coco(dataset):
    """Converts the VOC dataset to COCO style for evaluation

    Args:
        dataset: The dataset to convert to COCO style

    Returns:
        The coco-style ground truth for the VOC dataset
    """
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "VOC dataset in COCO format"},
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": cls}
            for i, cls in enumerate(VOC_CLASSES)
        ]
    }

    ann_id = 1
    for img_id in range(len(dataset)):
        _, target = dataset[img_id]

        coco_gt.dataset["images"].append({
            "id": img_id
        })

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1

            coco_gt.dataset["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt.createIndex()
    return coco_gt