#!/usr/bin/env python

import argparse
import random
import os

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from dataset import DroneImages
from metric import to_mask, IntersectionOverUnion
from model import bigMaskRCNN, smallMaskRCNN
from tqdm import tqdm


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def get_device_ddp(slurm_localid) -> torch.device:
    return f'cuda:{slurm_localid}' if torch.cuda.is_available() else f'cpu:{slurm_localid}'


def train(hyperparameters: argparse.Namespace):
    # get local node id 
    slurm_localid = int(os.getenv("SLURM_LOCALID"))
    # set fixed seeds for reproducible execution
    random.seed(hyperparameters.seed)
    np.random.seed(hyperparameters.seed)
    torch.manual_seed(hyperparameters.seed)

    # init ddp 
    world_size = int(os.getenv("SLURM_NPROCS")) # Get overall number of processes.
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    # slurm_job_gpus = os.getenv("SLURM_JOB_GPUS")
    slurm_localid = int(os.getenv("SLURM_LOCALID"))
    gpus_per_node = torch.cuda.device_count()
    if rank==0:
        print(f'Used GPUs: {gpus_per_node}')
    # gpu = rank % gpus_per_node
    # assert gpu == slurm_localid
    device = get_device_ddp(slurm_localid=slurm_localid)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", 
                            rank=rank, 
                            world_size=world_size, 
                            init_method="env://")
    # determines the execution device, i.e. CPU or GPU


    # set up the dataset
    drone_images = DroneImages(hyperparameters.root, downsample_ratio=None, augment=False)
    train_fraction = 0.9
    valid_fraction = 0.1
    tests_fraction = 1. - (train_fraction + valid_fraction)
    train_data, valid_data, _ = torch.utils.data.random_split(drone_images, [train_fraction, valid_fraction, tests_fraction])

    # distributed sampling of the dataset
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                        train_data,
                        num_replicas=torch.distributed.get_world_size(),
                        rank=torch.distributed.get_rank(),
                        shuffle=True,
                        drop_last=True)
    
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
                    valid_data,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=True,
                    drop_last=True)
    

    # initialize MaskRCNN model
    model = smallMaskRCNN()
    model.to(device)
    
    # wrap model with ddp
    model = DDP(model,
                device_ids=[slurm_localid],
                output_device=slurm_localid)
    # set up optimization procedure
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.lr)
    best_iou = 0.

    writer = SummaryWriter()

    
    # start the actual training procedure
    for epoch in range(hyperparameters.epochs):
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=hyperparameters.batch,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler)
        # set the model into training mode
        model.train()

        # training procedure
        train_loss = 0.0
        train_metric = IntersectionOverUnion(task='multiclass', num_classes=2)
        train_metric = train_metric.to(device)
        
        for i, batch in enumerate(tqdm(train_loader, desc='train', disable=True if rank != 0 else False)):
            model.train()
            x, label = batch
            x = list(image.to(device) for image in x)
            label = [{k: v.to(device) for k, v in l.items()} for l in label]
            model.zero_grad()
            optimizer.zero_grad()
            losses = model(x, label)
            loss = sum(l for l in losses.values())

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loss = 0
            # compute metric
            with torch.no_grad():
                model.eval()
                train_predictions = model(x)
                #for i,item in enumerate(train_predictions):
                #    train_predictions[i]['masks'] = torch.transpose(item['masks'],-1,-2)
                train_metric(*to_mask(train_predictions, label))
            label = 0
            x = 0
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("iou/train", train_metric.compute().item(), epoch)

        train_loss /= len(train_loader)
        train_loader = 0

        # set the model in evaluation mode
        model.eval()
        model.zero_grad()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        # test_loader = torch.utils.data.DataLoader(valid_data, batch_size=hyperparameters.batch, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=hyperparameters.batch,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=valid_sampler)

        # test procedure
        test_metric = IntersectionOverUnion(task='multiclass', num_classes=2)
        test_metric = test_metric.to(device)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader, desc='test ', disable=True if rank != 0 else False)):
                x_test, test_label = batch
                x_test = list(image.to(device) for image in x_test)
                test_label = [{k: v.to(device) for k, v in l.items()} for l in test_label]

                # score_threshold = 0.7
                test_predictions = model(x_test)
                #for i,item in enumerate(test_predictions):
                #    test_predictions[i]['masks'] = torch.transpose(item['masks'],-1,-2)
                test_metric(*to_mask(test_predictions, test_label))
                test_predictions = 0
                x_test = 0
                test_label = 0

        writer.add_scalar("iou/test", test_metric.compute().item())
        # output the losses

        # Compute average distributed train loss.
        torch.distributed.all_reduce(torch.tensor(train_loss,device=device)) # Allreduce rank-local mini-batch losses.
        train_loss /= world_size # Average allreduced rank-local mini-batch losses over all ranks.
        # average train IoU
        train_iou = train_metric.compute()
        torch.distributed.all_reduce(torch.tensor(train_iou,device=device))
        train_iou /= world_size
        # average test IoU
        test_iou = test_metric.compute()
        torch.distributed.all_reduce(torch.tensor(test_iou,device=device))      
        test_iou /= world_size 
        
        if rank == 0:
            print(f'Epoch {epoch}')
            print(f'\tTrain loss: {train_loss}')
            print(f'\tTrain IoU:  {train_iou}')
            print(f'\tTest IoU:   {test_iou}')
            # save the best performing model on disk
            if test_iou > best_iou:
                best_iou = test_iou
                print('\tSaving better model\n')
                torch.save(model.state_dict(), 'checkpoint_test_big.pt')
            else:
                print('\n')
        
        test_loader = 0
        train_metric = 0
        test_metric = 0
        train_iou = 0
        test_iou = 0
        train_loss = 0
        losses = 0
        x_test = 0
        test_predictions = 0
        x = 0
        label = 0
        torch.cuda.empty_cache()
    writer.flush()
    writer.close()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', default=1, help='batch size', type=int)
    # parser.add_argument('-e', '--epochs', default=100, help='number of training epochs', type=int)
    parser.add_argument('-e', '--epochs', default=10, help='number of training epochs', type=int)
    parser.add_argument('-l', '--lr', default=1e-4, help='learning rate of the optimizer', type=float)
    parser.add_argument('-s', '--seed', default=42, help='constant random seed for reproduction', type=int)
    # parser.add_argument('root', help='path to the data root', type=str)
    parser.add_argument('--root', default='/hkfs/work/workspace/scratch/dz4120-energy-train-data/', help='path to the data root', type=str)

    arguments = parser.parse_args()
    train(arguments)
