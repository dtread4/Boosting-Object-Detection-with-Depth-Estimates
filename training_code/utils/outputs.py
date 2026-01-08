# David Treadwell

import os
import json
import shutil

import torch

from training_code.utils.evals import get_best_epoch_val, coco_stats_to_dict

def keep_only_best_and_last_saved_models(config, val_ap50):
    """
    Removes all model state dictionary files except the best epoch's and the last epoch's

    Args:
        config: The configurations used for training
        val_ap50: The validation AP50 for each epoch
    """
    # Get what epochs to keep
    best_ap50, best_epoch = get_best_epoch_val(val_ap50)
    last_epoch = len(val_ap50)

    # Remove unneeded models
    for filename in os.listdir(config.OUTPUT.DIR):
        if filename.startswith("model_epoch_") and filename.endswith(".pth"):
            epoch = int(filename.split("_")[-1].split(".")[0])
            if epoch not in {best_epoch, last_epoch}:
                os.remove(os.path.join(config.OUTPUT.DIR, filename))

    # Print results
    print(f"Kept model state dictionaries for epochs {best_ap50} (best validation epoch) and {last_epoch} (last epoch)")


def organize_val_results_files(config, val_ap50):
    """
    Makes a subdirectory for validation results, then copies the best/last epochs back to the main output directory

    Args:
        config: The configurations used for training
        val_ap50: The validation AP50 for each epoch
    """
    # Get what epochs to keep in main output directory
    best_ap50, best_epoch = get_best_epoch_val(val_ap50)
    last_epoch = len(val_ap50)

    # Create a subdirectory to contain all of the validation results files
    val_results_dir = os.path.join(config.OUTPUT.DIR, "val_results")
    os.makedirs(val_results_dir, exist_ok=True)

    # Move all results files to the directory
    for filename in os.listdir(config.OUTPUT.DIR):
        if filename.startswith("val_results_epoch_") and filename.endswith(".json"):
            shutil.move(
                os.path.join(config.OUTPUT.DIR, filename),
                os.path.join(val_results_dir, filename)
            )

    # Copy the best and last epoch's results files back to the original directory
    shutil.copy( 
        os.path.join(val_results_dir, f"val_results_epoch_{best_epoch}.json"),
        os.path.join(config.OUTPUT.DIR, f"val_results_best_epoch_{best_epoch}.json")
        )
    shutil.copy(
        os.path.join(val_results_dir, f"val_results_epoch_{last_epoch}.json"),
        os.path.join(config.OUTPUT.DIR, f"val_results_last_epoch_{last_epoch}.json")
        )
    

def save_checkpoint(config, model, epoch, coco_eval, mean_train_loss, optimizer, scheduler):
    """
    Saves a model/optimizer/scheduler checkpoint

    Args:
        config: The configurations used for training
        model: The model being trained
        epoch: The current training epoch
        coco_eval: The validation results in COCO format
        optimizer: The optimizer used in training. Defaults to None.
        scheduler: The scheduler used in training. Defaults to None.
    """
    os.makedirs(config.OUTPUT.DIR, exist_ok=True)

    # Set up model state dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }
    
    # Create output path and save the model
    output_model_path = os.path.join(
        config.OUTPUT.DIR,
        f"model_epoch_{epoch + 1}.pth"
    )
    torch.save(checkpoint, output_model_path)

    # Save AP results to JSON file if they were produced
    # Silently does not save file if none were produced, but a warning message should have been printed earlier
    if coco_eval is not None:
        results_dict = coco_stats_to_dict(coco_eval)
        results_dict['mean_train_loss'] = mean_train_loss
        output_value_path = os.path.join(
            config.OUTPUT.DIR,
            f"val_results_epoch_{epoch + 1}.json"
        )
        with open(output_value_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)