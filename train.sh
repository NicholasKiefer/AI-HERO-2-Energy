#!/bin/bash

#SBATCH --job-name=AI-HERO_E3_training
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/hkfs/work/workspace/scratch/ih5525-E3/results/slurm-%j.out

export CUDA_CACHE_DISABLE=1

# export OMP_NUM_THREADS=76# Change 5-digit MASTER_PORT as you wish, SLURM will raise Error if duplicated with others.
export MASTER_PORT=12340

# Get the first node name as master address.
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR



data_workspace=/hkfs/work/workspace/scratch/ih5525-E3/datasets/raw_data
group_workspace=/hkfs/work/workspace/scratch/ih5525-E3

module load compiler/gnu/11
module load mpi/openmpi/4.0
module load lib/hdf5/1.12
module load devel/cuda/11.8

source ${group_workspace}/energy_venv/bin/activate
srun python ${group_workspace}/AI-HERO-2-Energy/train.py --batch 1 --epochs 50 --lr 1e-3 --root ${data_workspace}
