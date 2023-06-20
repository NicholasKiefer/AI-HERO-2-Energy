import json
import pathlib

import numpy as np
import os
import torch
import torch.utils.data
import torchvision as tv
import albumentations as ab

from PIL import Image, ImageDraw

from typing import Optional


class DroneImages(torch.utils.data.Dataset):
    def __init__(self, root: str = 'data', max_images: Optional[int] = None, downsample_ratio:Optional[int] = None, augment:Optional[bool] = False):
        self.root = pathlib.Path(root)
        self.parse_json(self.root / 'descriptor.json')
        self.downsample_ratio = downsample_ratio
        self.augment = augment
        
        if downsample_ratio is not None:
            sampled_path = self.root.parent / (self.root.name + f"_{self.downsample_ratio}")
            os.makedirs(sampled_path, exist_ok=True)
            self.sampled = {key: sampled_path / f"{self.images[key]}" for key in self.ids if os.path.exists(sampled_path / f"{self.images[key]}")}
        
        self.transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.ops.Permute(0, 2, 1),  # makes (c, h, w)
        ])
        
        if augment:
            self.augmentation = ab.Compose([ab.RandomCrop(480, 640), ], ab.BboxParams("pascal_voc"))  # these need equivalent for bounding box/seg mask

    def parse_json(self, path: pathlib.Path, max_images: Optional[int] = None):
        """
        Reads and indexes the descriptor.json

        The images and corresponding annotations are stored in COCO JSON format. This helper function reads out the images paths and segmentation masks.
        """
        with open(path, 'r') as handle:
            content = json.load(handle)

        images = content['images']
        annotations = content['annotations']

        self.ids = [entry['id'] for entry in images]
        self.images = {entry['id']: self.root / pathlib.Path(entry['file_name']).name for entry in images}

        # add all annotations into a list for each image
        self.polys = {}
        self.bboxes = {}
        for entry in annotations:
            image_id = entry['image_id']
            self.polys.setdefault(image_id, []).append(entry['segmentation'])
            self.bboxes.setdefault(image_id, []).append(entry['bbox'])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a drone image and its corresponding segmentation mask.

        The drone image is a tensor with dimensions [H x W x C=5], where
            H - height of the image
            W - width of the image
            C - (R,G,B,T,H) - five channels being red, green and blue color channels, thermal and depth information

        The corresponding segmentation mask is binary with dimensions [H x W].
        """
        image_id = self.ids[index]
        save = False
        file_path = self.images[image_id]
        if self.downsample_ratio is not None:
            if self.sampled.get(image_id, None) is not None:
                file_path = self.sampled[image_id]
                print(f'Using saved image {image_id}')
            else:
                print(f'Saving image {image_id}')
                save = True

        # deserialize the image from disk
        x = np.load(file_path)

        polys = self.polys[image_id]
        bboxes = self.bboxes[image_id]
        masks = []
        # generate the segmentation mask on the fly
        for poly in polys:
            mask = Image.new('L', (x.shape[1], x.shape[0],), color=0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(poly[0], fill=1, outline=1)
            masks.append(np.array(mask))

        masks = torch.tensor(np.array(masks))

        labels = torch.tensor([1 for a in polys], dtype=torch.int64)

        boxes = torch.tensor(bboxes, dtype=torch.float)
        # bounding boxes are given as [x, y, w, h] but rcnn expects [x1, y1, x2, y2]
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        y = {
            'boxes': boxes,  # FloatTensor[N, 4]
            'labels': labels,  # Int64Tensor[N]
            'masks': masks,  # UIntTensor[N, H, W]
        }
        x = self.transforms(x)
        x = x / 255.
        
        if self.augment:
            augmented = self.augment(image=x, mask=masks, bboxes=boxes)
            x = augmented["image"]
            y["boxes"] = augmented["bboxes"]
            y["masks"] = augmented["mask"]
        
        if self.downsample_ratio is not None and save:
            # either downsample by interpolate
            # x = torch.nn.functional.interpolate(x, scale_factor=self.downsample_ratio,)
            # but faster is downsample by e.g. maxpooling
            x = torch.nn.functional.max_pool2d(x, kernel_size=(self.downsample_ratio, self.downsample_ratio))

            small_x = (x * 255).permute(2, 0, 1).numpy().astype(np.int64)
            #new_file_path = file_path.split("/")[-1]
            #save_path = f"{self.root}_{self.downsample_ratio}/{new_file_path}"
            save_path = self.root.parent / (self.root.name + f"_{self.downsample_ratio}") / file_path.name
            np.save(save_path, small_x)
            self.sampled[image_id] = save_path
        return x, y
