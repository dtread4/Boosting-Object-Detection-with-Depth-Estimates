# David Treadwell

import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import torch

from torch.amp import autocast, GradScaler

from dataset_classes.depth_model import DepthModel

from training_code.models.faster_rcnn import build_faster_rcnn

from training_code.utils.dataloader import build_dataloader
from training_code.utils.evals import evaluate_loss 
from training_code.utils.graphing import save_train_val_plot, save_lr_scheduler_plot
from training_code.utils.optimizer import build_optimizer_scheduler, get_current_learning_rate
from training_code.utils.outputs import save_checkpoint, keep_only_best_and_last_saved_models, organize_val_results_files
from training_code.utils.utility import set_device, set_reproducability_settings, tensor_to_pil


def train(config):
    """
    Trains a full pass through all epochs

    Args:
        config: The configurations to train with
    """
    # Set the device
    device_type, device = set_device(config.UTILITY.DEVICE)

    # Load the depth model (if any; will be None if not)
    depth_model = DepthModel.initialize_depth_model(
        model_name=config.DEPTH_INFO.DEPTH_MODEL,
        device=device
    )

    # Build dataloaders
    train_loader = build_dataloader(config, "TRAIN")  # TODO these should be specified, either TRAIN/VAL or TRAINVAL/TEST
    val_loader = build_dataloader(config, "VAL")

    # Build optimizer and scheduler
    use_depth = config.DEPTH_INFO.DEPTH_MODEL is not None
    detection_model = build_faster_rcnn(config, add_depth_channel=use_depth).to(device)
    optimizer, scheduler = build_optimizer_scheduler(config, detection_model)

    # Create a scaler
    scaler = GradScaler(enabled=config.SOLVER.AMP)

    # Train for config-defined number of epochs
    train_epochs = []  # For simple matching between train loss and epoch
    train_losses_per_batch = []
    val_ap50 = []  # Each epoch's validation AP50
    lr_history = []  # Learning rate used for each training epoch

    for epoch in tqdm(range(config.SOLVER.MAX_EPOCHS), desc='Training epochs'):
        detection_model.train()
        train_loss = 0
        num_batches = 0

        # Loop over all batches each epoch
        for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc='Current training epoch\'s batches')):
            # Calculate depth maps if using depth
            if depth_model is not None:
                # TODO it's possible that non-determinism could lead to issues, since depth maps could differ each epoch
                # TODO worth experimenting with pre-calculated depth maps to assess this
                pil_images = tensor_to_pil(images)
                depth_masks = depth_model.calculate_depth_map(pil_images)
                depth_masks_tensor = [torch.from_numpy(dm).unsqueeze(0) for dm in depth_masks]
                images = [torch.cat([images[i], depth_masks_tensor[i]], dim=0) for i in range(len(images))]

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Calculate loss for batch
            with autocast(enabled=config.SOLVER.AMP, device_type=device_type):
                loss_dict = detection_model(images, targets)
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
        mean_train_loss = train_loss / num_batches
        coco_eval = evaluate_loss(detection_model, val_loader, device, depth_model)

        # Print results for epoch
        if coco_eval is None:
            val_AP = {
                "AP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0
            }
        else:
            val_AP = {
                "AP": coco_eval.stats[0],     # mAP@[0.5:0.95]
                "AP50": coco_eval.stats[1],   
                "AP75": coco_eval.stats[2]
            }
        val_ap50.append(val_AP['AP50'])

        # Print to output file
        # TODO send this to an output file, flush as it comes in
        print(f"Epoch {epoch + 1}: train loss: {mean_train_loss:.2f}")
        print(f"Validation stats:\
              AP: {val_AP['AP']} | AP50: {val_AP['AP50']} | AP75: {val_AP['AP75']}")
        
        # Save epoch's model and validation results
        save_checkpoint(config, detection_model, epoch, coco_eval, mean_train_loss, optimizer, scheduler)

    # Save the training and validation loss graph and the learning rate graph
    save_train_val_plot(train_epochs, train_losses_per_batch, val_ap50, config.OUTPUT.DIR)
    save_lr_scheduler_plot(lr_history, config.OUTPUT.DIR)

    # Remove all but best epoch and last epoch
    keep_only_best_and_last_saved_models(config, val_ap50)
    organize_val_results_files(config, val_ap50)


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

    main(config)
