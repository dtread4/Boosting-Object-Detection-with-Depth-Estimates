# David Treadwell

import torch

from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet50


def build_faster_rcnn(config, add_depth_channel):
    """
    Builds a model from either a pre-trained model's weights or from default

    Args:
        config: configurations to use 
        add_depth_channel: Whether to add a depth channel or not. Defaults to False (no depth channel, 3-channel inputs)

    Returns:
        A Faster R-CNN model
    """
    model_weights = "DEFAULT" if config.MODEL.PRETRAINED else None

    if add_depth_channel:
        # Adjust the input channels from 3 to 4
        resnet_backbone = resnet50(weights=model_weights)
        print(f"ResNet50 backbone's first convolution before depth addition: {resnet_backbone.conv1}")
        resnet_backbone.conv1 = torch.nn.Conv2d(
            in_channels=4,  # 4 is RGB (3) + Depth (1) = 4 channels, RGB-D
            out_channels=resnet_backbone.conv1.out_channels,
            kernel_size=resnet_backbone.conv1.kernel_size,
            stride=resnet_backbone.conv1.stride,
            padding=resnet_backbone.conv1.padding,
            bias=resnet_backbone.conv1.bias is not None,
        )
        print(f"ResNet50 backbone's first convolution after depth addition: {resnet_backbone.conv1}")

        # Create the FPN backbone with the updated ResNet50 model
        fpn_backbone = BackboneWithFPN(
            resnet_backbone,
            return_layers={"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"},
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )

        print("Updated ResNet50 FPN backbone to include depth channel\n")

        faster_rcnn_model = FasterRCNN(
            weights=model_weights,
            backbone=fpn_backbone,
            num_classes=config.MODEL.NUM_CLASSES,
            min_size=config.MODEL.MIN_SIZE,
            max_size=config.MODEL.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406, 0.396],  # TODO setup to update the depth mean per model
            image_std=[0.229, 0.224, 0.225, 0.287],  # TODO setup to update the depth std per model
        )

        return faster_rcnn_model
    
    else:
        print("Using default Faster R-CNN backbone with RGB inputs\n")

        faster_rcnn_model = fasterrcnn_resnet50_fpn(
            weights=model_weights,
            num_classes=config.MODEL.NUM_CLASSES,
            min_size=config.MODEL.MIN_SIZE,
            max_size=config.MODEL.MAX_SIZE
        )
    return faster_rcnn_model