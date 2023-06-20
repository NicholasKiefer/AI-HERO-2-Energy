#!/bin/bash

#SBATCH --job-name=AI-HERO_energy_baseline_training
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=76
#SBATCH --time=20:00:00

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76

data_workspace=/hkfs/work/workspace/scratch/ih5525-E3/datasets/
group_workspace=/hkfs/work/workspace/scratch/ih5525-E3/AI-HERO-2-Energy/ih5525-E3/

module load compiler/gnu/11
module load mpi/openmpi/4.0
module load lib/hdf5/1.12
module load devel/cuda/11.8

source ${group_workspace}/energy_venv/bin/activate
srun python ${group_workspace}/train.py --batch 16 --epochs 10 --lr 1e-3 --root ${data_workspace}
