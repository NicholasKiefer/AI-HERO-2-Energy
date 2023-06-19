import torch
from pathlib import Path
from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2


def MaskRCNN(in_channels=5, num_classes=2, trainable_backbone_layers=5, image_mean=None, image_std=None, **kwargs):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225, 0.225, 0.225]
        
    model = maskrcnn_resnet50_fpn_v2(
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        image_mean=image_mean,
        image_std=image_std
    )
    model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)

    return model

def yolo(in_channels=5, num_classes=2, *args, **kwargs):
        return Yolo(in_channels, num_classes, *args, **kwargs)
    

class Yolo(torch.nn.Module):
    def __init__(self, in_channels, num_classes, *args, **kwargs) -> None:
         super().__init__()
         self.convert = nn.Conv2d(in_channels, 3, 1, 1, 0, bias=False)
         self.yolonet = None  # load real fucking fancy yolo model here here
         
    def forward(self, x, labels=None):
        x = self.convert(x)
        x = self.yolonet(x)
        if self.training:
            loss = self.loss_func(x, labels)
            return loss
        else:
            # eval mode, masks are needed for .shape, boxes are preds of yolonet
            return {"boxes": x, "masks": x}
    
    def loss_func(self, x, labels):
        # here import fancy fucking loss calculation
        return self.yolonet.get_loss(x, labels)
