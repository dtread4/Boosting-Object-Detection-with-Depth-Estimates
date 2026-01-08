# David Treadwell

import torch.optim as optim

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
    