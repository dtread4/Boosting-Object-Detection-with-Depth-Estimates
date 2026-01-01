# David Treadwell

import argparse
import yaml
import os
from tqdm import tqdm
from omegaconf import OmegaConf
import json

import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import torch.optim as optim

from torch.amp import autocast, GradScaler

from pycocotools.cocoeval import COCOeval

from dataset_classes.VOC_dataset import VOCDataset, voc_to_coco, VOC_CLASSES
from dataset_classes.utility import get_transforms


# TODO separate into multiple files (classes?) for cleanliness


def set_reproducability_settings(config, verbose=True):
    """
    Sets backend random seeds and Torch determinism to ensure experiments are reproducible

    Args:
        config: The configuration file used for setup
        verbose: Whether to print statement about reproducibility settings
    """
    # Set seeds
    torch.manual_seed(config.UTILITY.SEED)
    torch.cuda.manual_seed_all(config.UTILITY.SEED)
    random.seed(config.UTILITY.SEED)
    np.random.seed(config.UTILITY.SEED)

    # Set rough determinism (note it will not be truly deterministic for Faster R-CNN, but it's better than nothing)
    # TODO maybe look into this more?
    torch.backends.cudnn.deterministic = config.UTILITY.DETERMINISTIC

    if verbose:
        print(f"Using random seed: [{config.UTILITY.SEED}] and {"[NOT USING]" if not config.UTILITY.DETERMINISTIC else "[USING]"} determinism")


def set_device(device_type='auto', verbose=True):
    """
    Sets the global device used for training and testing

    Args:
        device_type: The name of the device type (auto, cpu, cuda)
        verbose: Whether to print statement about reproducibility settings

    Returns:
        The device used for training and testing
    """
    if device_type == 'auto':
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f"Training on device [{device_type}]\n")
    return device_type, torch.device(device_type)


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
        split=split,
        transforms=get_transforms(train=(split == "TRAIN"))
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.SOLVER.BATCH_SIZE if split == "TRAIN" else 1,
        shuffle=(split == "train"),
        num_workers=config.DATALOADER.NUM_WORKERS,
        collate_fn=collate_fn
    )
    return dataloader


def build_model(config):
    """
    Builds a model from either a pre-trained model's weights or from default

    Args:
        config: configurations to use 

    Returns:
        A Faster R-CNN model
    """
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if config.MODEL.PRETRAINED else None,
        num_classes=config.MODEL.NUM_CLASSES,
        min_size=config.MODEL.MIN_SIZE,
        max_size=config.MODEL.MAX_SIZE
    )
    return model


def get_parameter_groups(config, model):
    """
    Returns parameter groups with the proper weight decay applied for the optimizer
    Prevents biases and normalization layers from having weight decay, as well as ignoring parameters that
    have no associated gradients to improve training efficiency
    See here for more explanation on why this function is beneficial: 
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2

    Args:
        config: The set of configuration parameters
        model: The model being trained

    Returns:
        The set of parameters to optimizer training for
    """
    params = []

    # TODO confirm that this isn't that different than the previous method where bias/bn/norm layers also had decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Do not apply weight decay to biases or normalization layers
        if name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            params.append({
                "params": [param],
                "weight_decay": 0.0,
            })
        else:
            params.append({
                "params": [param],
                "weight_decay": config.SOLVER.WEIGHT_DECAY
            })

    return params


def build_optimizer_scheduler(config, model):
    """
    Creates an optimizer and scheduler for use in training

    Args:
        config: The configurations to set up optimizer/scheduler with
        model: The model being optimized

    Returns:
        The created optimizer and scheduler object
    """
    params = get_parameter_groups(config, model)

    # Set up the optimizer
    optimizer_name = config.SOLVER.OPTIMIZER.lower()

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(
            params,
            lr=config.SOLVER.BASE_LR,
            betas=tuple(config.SOLVER.BETAS),
            eps=config.SOLVER.EPS
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.SOLVER.BASE_LR,
            betas=tuple(config.SOLVER.BETAS),
            eps=config.SOLVER.EPS
        )
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.SOLVER.BASE_LR,
            momentum=config.SOLVER.MOMENTUM
        )
    else:
        raise ValueError(f"ERROR: Unsupported optimizer type {config.SOLVER.OPTIMIZER}")

    # Set up the scheduler
    scheduler_name = config.SOLVER.SCHEDULER.lower()
    if scheduler_name == "multistep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.SOLVER.LR_STEPS,
            gamma=config.SOLVER.LR_GAMMA
        )
    elif scheduler_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.SOLVER.LR_GAMMA
        )
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.SOLVER.MAX_EPOCHS
        )
    else:
        raise ValueError(f"Unknown LR scheduler: {config.SOLVER.SCHEDULE}")
    
    return optimizer, scheduler


def run_inference(model, dataloader, device):
    """
    Gets model outputs for a dataset (should be val/test dataset)

    Args:
        model: The model to evaluate with
        loader: The dataloader managing the dataset to evaluate with
        device: The device being used for evaluation

    Returns:
        The set of predictions and confidence scores for each image/predicted object
    """
    model.to(device)
    model.eval()
    predictions = []

    image_id = 0

    # Evaluate each image
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc='Current epoch\'s inference batches'):
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


def evaluate_loss(model, loader, device):
    """
    Evaluates a model's mAP on a dataset (should be the val/test set)

    Args:
        model: The model to evaluate with
        loader: The dataloader managing the dataset to evaluate with
        device: The device being used for evaluation

    Returns:
        the coco evaluation results
    """
    # Get predictions from model
    coco_gt = voc_to_coco(loader.dataset)
    predictions = run_inference(model, loader, device)    

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


def save_checkpoint(config, model, epoch, val_AP, optimizer=None, scheduler=None):
    """
    Saves a model/optimizer/scheduler checkpoint

    Args:
        config: The configurations used for training
        model: The model being trained
        epoch: The current training epoch
        optimizer: The optimizer used in training. Defaults to None.
        scheduler: The scheduler used in training. Defaults to None.
    """
    os.makedirs(config.OUTPUT.DIR, exist_ok=True)

    # Set up model dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict()
    }

    # Set up optimizer/scheduler dictionary
    # TODO make these actually save
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler_state'] = scheduler.state_dict()
    
    # Create output path and save
    # TODO update this to only save best/last epoch models
    output_model_path = os.path.join(
        config.OUTPUT.DIR,
        f"model_epoch_{epoch + 1}.pth"
    )

    torch.save(checkpoint, output_model_path)

    # Save AP results to JSON file
    results_dict = coco_stats_to_dict(val_AP)
    output_value_path = os.path.join(
        config.OUTPUT.DIR,
        f"val_results_epoch_{epoch + 1}.json"
    )
    with open(output_value_path, 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)


def save_train_val_plot(train_epochs, train_losses_per_batch, val_ap50, output_dir):
    """
    Saves a graph of the train loss and validation AP50 over the course of the training

    Args:
        train_epochs: List of epoch number for each batch of training loss
        train_losses_per_batch: Each training batch's loss in a list
        val_ap50: The validation AP50 for each epoch
        output_dir: The output directory to save to
    """
    # Get the best validation AP50
    best_epoch = int(np.argmax(val_ap50))
    best_ap50 = val_ap50[best_epoch]

    fig, ax1 = plt.subplots()

    # Plot training losses
    ax1.plot(
        train_epochs,
        train_losses_per_batch,
        color="blue",
        label="Training loss per batch"
    )
    ax1.set_ylabel("Training loss per batch", color='blue')

    # Set x axis (applies to full graph)
    ax1.set_xlabel("Epoch")

    # Plot the validation AP50s
    ax2 = ax1.twinx()
    ax2.plot(
        range(1, len(val_ap50) + 1),  # tick marks (one per epoch)
        val_ap50,
        color='orange',
        marker='o',
        label='Val AP50'
    )
    ax2.set_ylabel("Val AP50", color='orange')

    # Set x-axis limits and integer ticks
    max_epoch = len(val_ap50)
    ax1.set_xlim(-0.5, max_epoch + 0.5)
    ax1.set_xticks(range(max_epoch + 1))

    # Finish design
    fig.suptitle(
        f"Train/Validation loss | best val AP50 {best_ap50:.4f} from epoch {best_epoch}"
    )
    plt.tight_layout()

    # Save to output directory
    out_path = os.path.join(output_dir, 'train_val_graph.png')
    plt.savefig(out_path)
    plt.close(fig)


def save_lr_scheduler_plot(lr_history, output_dir):
    """
    Plots the learning rate for each epoch in the training regime

    Args:
        lr_history: List of learning rate during each epoch
        output_dir: The output directory to save the graph to
    """
    max_epoch = len(lr_history)

    plt.figure()
    plt.step(
        range(1, len(lr_history) + 1),
        lr_history,
        color='green',
        where='post'
        )
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.title('Learning rate through training regime')
    plt.tight_layout()

    # Set the x-ticks
    plt.xlim(-0.5, max_epoch + 0.5)
    plt.xticks(range(1, max_epoch + 1))

    output_path = os.path.join(output_dir, "learning_rate_schedule.png")
    plt.savefig(output_path)
    plt.close()


def get_current_learning_rate(optimizer):
    """
    Gets the learning rate for the optimizer
    This should be called before calling 'scheduler.step()'

    Args:
        optimizer: The optimizer being used to train the model

    Raises:
        RuntimeError: If there were no trainable parameters (excluding bias, normalization layers) in the optimizer's
                      parameter groups, an error will be raised

    Returns:
        The learning rate for the valid trainable parameters
    """
    epoch_lr = None
    for param_group in optimizer.param_groups:
        if param_group['weight_decay'] > 0:
            return param_group['lr']
        
    if epoch_lr is None:
        raise RuntimeError("No parameter group with weight decay > 0 found! Exiting.")


def train(config):
    """
    Trains a full pass through all epochs

    Args:
        config: The configurations to train with
    """
    # Set the device
    device_type, device = set_device(config.UTILITY.DEVICE)

    # Build dataloaders
    train_loader = build_dataloader(config, "TRAIN")
    val_loader = build_dataloader(config, "VAL")

    # Build optimizer and scheduler
    model = build_model(config).to(device)
    optimizer, scheduler = build_optimizer_scheduler(config, model)

    # Create a scaler
    scaler = GradScaler(enabled=config.SOLVER.AMP)

    # Train for config-defined number of epochs
    train_epochs = []  # For simple matching between train loss and epoch
    train_losses_per_batch = []
    val_ap50 = []  # Each epoch's validation AP50
    lr_history = []  # Learning rate used for each training epoch

    for epoch in tqdm(range(config.SOLVER.MAX_EPOCHS), desc='Training epochs'):
        model.train()
        train_loss = 0
        num_batches = 0

        # Loop over all batches each epoch
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc='Current training epoch\'s batches')):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Calculate loss for batch
            with autocast(enabled=config.SOLVER.AMP, device_type=device_type):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            # Backwards pass for batch
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update overall epoch's training loss
            batch_train_loss = loss.item()
            train_loss += batch_train_loss
            num_batches += 1

            # Get fractional epoch progress for epoch based on batch counts
            epoch_progress = epoch + (batch_idx + 1) / len(train_loader)

            # Record batch training performance
            train_epochs.append(epoch_progress)
            train_losses_per_batch.append(batch_train_loss)

        # Get the learning rate used for parameters that have their learning rate adjusted (decayed)
        lr_history.append(get_current_learning_rate(optimizer))

        # Update scheduler after all batches for epoch trained on
        scheduler.step()

        # Calculate mean training loss and validation loss
        train_loss = train_loss / num_batches
        coco_eval = evaluate_loss(model, val_loader, device)
        val_AP = {
            "AP": coco_eval.stats[0],     # mAP@[0.5:0.95]
            "AP50": coco_eval.stats[1],   
            "AP75": coco_eval.stats[2]
        }
        val_ap50.append(val_AP['AP50'])

        # Print to output file
        # TODO send this to an output file, flush as it comes in
        print(f"Epoch {epoch + 1}: train loss: {train_loss:.2f}")
        print(f"Validation stats:\
              AP: {val_AP['AP']} | AP50: {val_AP['AP50']} | AP75: {val_AP['AP75']}")

        # TODO update this to save only best/last epoch models (maybe remove temp ones from previous epochs when training completes?)
        # TODO add training loss to saved checkpoint dict
        save_checkpoint(config, model, epoch, coco_eval)

    # Save the training and validation loss graph and the learning rate graph
    save_train_val_plot(train_epochs, train_losses_per_batch, val_ap50, config.OUTPUT.DIR)
    save_lr_scheduler_plot(lr_history, config.OUTPUT.DIR)
            

def main(config):
    # Set reproducibility settings first
    set_reproducability_settings(config)

    # Run the training regime
    train(config)


if __name__ == "__main__":
    # Parse path to config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True, type=str, 
                        help="Path to configuration YAML file containing setup information.")
    args = parser.parse_args()

    # Read YAML path
    config = OmegaConf.load(args.config_path)
    OmegaConf.set_readonly(config, False)  # Makes the configurations editable

    main(config)
