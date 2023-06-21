from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN, MaskRCNNHeads
from torchvision.models import resnet34, resnet50, resnet18

from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import _default_anchorgen, FasterRCNN, FastRCNNConvFCHead, RPNHead


def bigMaskRCNN(in_channels=5, num_classes=2, trainable_backbone_layers=5, image_mean=None, image_std=None, **kwargs):
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


def smallMaskRCNN(in_channels=5, num_classes=2, trainable_backbone_layers=5, image_mean=None, image_std=None, **kwargs):
    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406, 0.5, 0.5]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225, 0.225, 0.225]
    
    # backbone = resnet34(progress=True)
    # backbone = resnet18(progress=True)
    backbone = resnet50(progress=True)
    
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
    rpn_anchor_generator = _default_anchorgen()
    rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=nn.BatchNorm2d
    )
    mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        mask_head=mask_head,
        image_mean=image_mean, image_std=image_std,
        **kwargs,
    )
    # model.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False).requires_grad_(True)
    model.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).requires_grad_(True)
    return model
