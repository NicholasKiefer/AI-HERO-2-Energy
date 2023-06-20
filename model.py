import torch
from pathlib import Path
from torch import nn
import argparse
import numpy as np

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from pytorch_YOLOv4.tool.darknet2pytorch import Darknet
from pytorch_YOLOv4.train import Yolo_loss
from pytorch_YOLOv4.models import Yolov4


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

def yolo(in_channels=5, num_classes=2, batch_size=-1, *args, **kwargs):
    assert batch_size != -1
    return Yolo(in_channels, num_classes, batch_size, *args, **kwargs)
    

class Yolo(torch.nn.Module):
    def __init__(self, in_channels, num_classes, batch_size, *args, **kwargs) -> None:
         super().__init__()
         self.convert = nn.Conv2d(in_channels, 3, 1, 1, 0, bias=False)
         self.yolonet = Yolov4(False, num_classes - 1, )
         self.yololoss = Yolo_loss(num_classes - 1, batch=batch_size)
         
    def forward(self, img, y=None):
        
        # x is (c, h, w)
        x = self.convert(img)
        
        if self.training:
            x = self.yolonet(x)
            loss = self.loss_func(x, y)
            return loss
        else:
            self.yolonet.head.inference = True
            outputs = self.yolonet(x)
            # convert region boxes to our output, boxes are [batch, num1 + num2 + num3, 1, 4]
            res = []

            for img, boxes, confs in zip(img, outputs[0], outputs[1]):
                img_height, img_width = img.shape[:2]
                boxes = boxes.squeeze(2).cpu().detach().numpy()
                boxes[...,2:] = boxes[...,2:] - boxes[...,:2] # Transform [x1, y1, x2, y2] to [x1, y1, w, h]
                boxes[...,0] = boxes[...,0]*img_width
                boxes[...,1] = boxes[...,1]*img_height
                boxes[...,2] = boxes[...,2]*img_width
                boxes[...,3] = boxes[...,3]*img_height
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                confs = confs.cpu().detach().numpy()
                labels = np.argmax(confs, axis=1).flatten()
                labels = torch.as_tensor(labels, dtype=torch.int64)
                scores = np.max(confs, axis=1).flatten()
                scores = torch.as_tensor(scores, dtype=torch.float32)
                res.append({
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                })
            
            ret = [{"boxes": x, "masks": x} for _ in x]  # list of batch size tensors shape [num_instances, 4]
            return ret
    
    def loss_func(self, x, y):
        self.yololoss.to(x.device)
        bboxes = y["boxes"]  # convert this probably
        bboxes_pred = x
        loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = self.yololoss(bboxes_pred, bboxes)
        return loss
