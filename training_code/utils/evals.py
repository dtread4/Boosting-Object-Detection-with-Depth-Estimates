# David Treadwell

from tqdm import tqdm

import torch
import torchvision.transforms as T

import numpy as np

from pycocotools.cocoeval import COCOeval

from dataset_classes.VOC_dataset import voc_to_coco, VOC_CLASSES

from training_code.utils.transformations import get_transforms
from training_code.utils.utility import get_depth_masks


def run_inference(model, dataloader, device, depth_model, precomputed_depth):
    """
    Gets model outputs for a dataset (should be val/test dataset)

    Args:
        model: The model to evaluate with
        loader: The dataloader managing the dataset to evaluate with
        device: The device being used for evaluation
        depth_model: The depth model to estimate depth maps with (if one is being used, else None)
        precomputed_depth: Whether precomputed depth is being used or not

    Returns:
        The set of predictions and confidence scores for each image/predicted object
    """
    model.eval()
    predictions = []
    test_transforms = get_transforms(train=False)

    image_id = 0

    # Evaluate each image
    with torch.no_grad():
        for images, _, image_paths in tqdm(dataloader, desc='Current epoch\'s inference batches'):
            # Convert 8-bit images to tensors directly 
            # required before concatenating depth masks, 8-bit unit images need different transformation here 
            # than float32 0-1 depth masks
            images = [T.ToTensor(images)]

            # Calculate depth maps if using depth
            if depth_model is not None:
                # Pass actual images if using runtime depth maps, otherwise pass image paths
                # TODO setup this way to minimize loading from disk
                depth_masks = get_depth_masks(precomputed_depth, depth_model, images, image_paths)
                depth_masks_tensor = [torch.from_numpy(dm).unsqueeze(0) for dm in depth_masks]
                images = [torch.cat([images[i], depth_masks_tensor[i]], dim=0) for i in range(len(images))]

            # Apply additional transformations
            images = test_transforms(images)

            images = [img.to(device) for img in images]
            outputs = model(images)

            # For each predicted object, store the output information
            for output in outputs:
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # [x, y, w, h]
                        "score": float(score)
                    })

                image_id += 1
    
    return predictions


def evaluate_loss(model, loader, device, depth_model, precomputed_depth):
    """
    Evaluates a model's mAP on a dataset (should be the val/test set)

    Args:
        model: The model to evaluate with
        loader: The dataloader managing the dataset to evaluate with
        device: The device being used for evaluation
        device: The depth model to estimate depth maps with (if one is being used, else None)
        precomputed_depth: Whether precomputed depth is being used or not

    Returns:
        the coco evaluation results
    """
    # Get predictions from model
    coco_gt = voc_to_coco(loader.dataset)
    predictions = run_inference(model, loader, device, depth_model, precomputed_depth)

    # Make sure any predictions were generated - if not, return None to designate this
    if len(predictions) == 0:
        print("WARNING: no valid predictions for this set")
        return None    

    # Evaluate the results in the COCO format
    coco_detections = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_detections, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


def per_class_ap50(coco_eval):
    """
    Calculates each class's AP50 value

    Args:
        coco_eval: The coco-style evaluation results

    Returns:
        A dictionary of each class's AP50 results
    """
    per_class_scores = {}

    # Get the AP50 for each class
    iou50_idx = 0
    for k, class_name in enumerate(VOC_CLASSES):
        p = coco_eval.eval["precision"][iou50_idx, :, k, 0, 2]  # [IoU threshold, recall thresholds, class index, area range (0: all), max detections (2: 100)]
        p = p[p > -1]  # Missing/invalid entries are removed by this line (so undefined recall is ignored)

        per_class_scores[class_name] = float(np.mean(p)) if p.size else float("nan")  # Safety check to ensure valid classes account for

    return per_class_scores


def coco_stats_to_dict(coco_eval):
    """
    Puts the coco statistics into a dictionary format (from a list)

    Args:
        coco_eval: The coco-style evaluation results

    Returns:
        The dictionary format for the results
    """
    results_dict = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AP_small": coco_eval.stats[3],
        "AP_medium": coco_eval.stats[4],
        "AP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11],
    }
    results_dict['per_class_AP50'] = per_class_ap50(coco_eval)
    return results_dict


def get_best_epoch_val(val_list):
    """
    Gets the best (maximum) validation result

    Args:
        val_list: List containing a per-epoch validation results

    Returns:
        The best validation result and the epoch it occurred at
    """
    best_epoch_index = int(np.argmax(val_list))
    best_val = val_list[best_epoch_index]
    best_epoch = best_epoch_index + 1
    return best_val, best_epoch