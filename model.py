import torch
from pathlib import Path
from torch import nn

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from ultralytics import YOLO


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

def yolo(in_channels=5, num_classes=2, trainable_backbone_layers=5, image_mean=None, image_std=None, **kwargs):
    """
    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)
          
    this also needs to deal with normalizing the image, or we do this in another branch
    """
    yolo_hub = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    return YOLO_converted()


class YOLO_converted(YOLO):
    def __init__(self, model: str | Path = 'yolov8n.pt') -> None:
        super().__init__(model, task="segment")
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
    def __call__(self, images, targets=None):
        # need to use __call__ or 
        pass
    
    def zero_grad(self):
        self._check_is_pytorch_model()
        self.model.zero_grad()
        
    def eval(self):
        self._check_is_pytorch_model()
        self.model.eval()
        