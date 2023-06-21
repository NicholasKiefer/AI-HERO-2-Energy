#!/usr/bin/env python

import argparse
import random
import os

import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


from dataset import DroneImages
from metric import to_mask, IntersectionOverUnion
from model import MaskRCNN
from tqdm import tqdm

def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def get_device_ddp(slurm_localid) -> torch.device:
    return f'cuda:{slurm_localid}' if torch.cuda.is_available() else f'cpu:{slurm_localid}'


def predict(hyperparameters: argparse.Namespace):
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    # init ddp 
    slurm_localid = int(os.getenv("SLURM_LOCALID"))  # get local node id 
    world_size = int(os.getenv("SLURM_NPROCS")) # Get overall number of processes.
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    # slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    slurm_localid = int(os.getenv("SLURM_LOCALID"))
    gpus_per_node = torch.cuda.device_count()
    if rank==0:
        print(f'Used GPUs: {gpus_per_node}')  

    # determines the execution device, i.e. CPU or GPU
    device = get_device_ddp(slurm_localid=slurm_localid)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            world_size=world_size, 
                            init_method="env://")

    # set up the dataset
    drone_images = DroneImages(hyperparameters.root)
    # test_data = drone_images

    train_fraction = 0.02
    valid_fraction = 0.01
    tests_fraction = 1. - (train_fraction + valid_fraction)
    train_data, test_data, _ = torch.utils.data.random_split(drone_images, [train_fraction, valid_fraction, tests_fraction])

    test_sampler = torch.utils.data.distributed.DistributedSampler(
                        test_data,
                        num_replicas=torch.distributed.get_world_size(),
                        rank=torch.distributed.get_rank(),
                        drop_last=True)

    # initialize the U-Net model
    model = MaskRCNN()
    if hyperparameters.model:
        if rank == 0:
            print(f'Restoring model checkpoint from {hyperparameters.model}')
        model.load_state_dict(torch.load(hyperparameters.model))
    model.to(device)

    # wrap model with ddp
    model = DDP(model,
                device_ids=[slurm_localid],
                output_device=slurm_localid)
    
    # set the model in evaluation mode
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=hyperparameters.batch, collate_fn=collate_fn,sampler=test_sampler)

    # test procedure
    test_metric = IntersectionOverUnion(task='multiclass', num_classes=2)
    test_metric = test_metric.to(device)
    
    for i, batch in enumerate(tqdm(test_loader, desc='test ')):
        x_test, test_label = batch
        x_test = list(image.to(device) for image in x_test)
        test_label = [{k: v.to(device) for k, v in l.items()} for l in test_label]

        # score_threshold = 0.7
        with torch.no_grad():
            test_predictions = model(x_test)
            test_metric(*to_mask(test_predictions, test_label))
    if rank == 0:
        test_iou = torch.distributed.all_reduce(torch.tensor(test_metric.compute(),device=device))   
        print(f'Test IoU: {test_iou}')
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=1, help='batch size', type=int)
    parser.add_argument('-m', '--model', default='checkpoint.pt', help='model checkpoint', type=str)
    parser.add_argument('-s', '--seed', default=42, help='constant random seed for reproduction', type=int)
    parser.add_argument('root', help='path to the data root', type=str)

    arguments = parser.parse_args()
    predict(arguments)
