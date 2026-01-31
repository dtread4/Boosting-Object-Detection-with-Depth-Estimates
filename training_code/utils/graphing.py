# David Treadwell

import os

import numpy as np
import matplotlib.pyplot as plt

from training_code.utils.evals import get_best_epoch_val


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
    best_ap50, best_epoch = get_best_epoch_val(val_ap50)

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

    # Enforce range of 0-1 with ticks every 0.1 (-0.05 to 1.05 for padding)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))

    # Set x-axis limits and integer ticks
    max_epoch = len(val_ap50)
    ax1.set_xlim(-0.5, max_epoch + 0.5)
    ax1.set_xticks(range(1, max_epoch + 1))
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