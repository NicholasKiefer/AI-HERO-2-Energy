import torch

from dataset import DroneImages
from train import collate_fn


def data_tests():
    data = DroneImages(root='../datasets/raw_data', max_images=None)
    #loader = torch.utils.data.DataLoader(
    #    data,
    #    batch_size=len(data),
    #    shuffle=True,
    #    drop_last=True,
    #    collate_fn=collate_fn)

    all_heights = []
    all_widths = []
    for _, image in data:
        boxes = image['boxes']
        box_heights = boxes[:, 3] - boxes[:, 1]
        all_heights.append(box_heights)
        box_widths = boxes[:, 2] - boxes[:, 0]
        all_widths.append(box_widths)

    widths = torch.concat(all_widths, dim=0)
    heights = torch.concat(all_heights, dim=0)

    print('Height:')
    print(f'mean: {heights.mean()}, median: {heights.median()}, std: {heights.std()}, min: {heights.min()}, max: {heights.max()}')
    print('Widths:')
    print(f'mean: {widths.mean()}, median: {widths.median()}, std: {widths.std()}, min: {widths.min()}, max: {widths.max()}')


if __name__ == '__main__':
    data_tests()
