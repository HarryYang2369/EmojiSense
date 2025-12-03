#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=swabhas_1765
#SBATCH --job-name=train_age
#SBATCH --output=train_age_%j.log

module purge
module load ver/2506
module load gcc/15.2.0
module load python/3.13.2

# Activate Environment (Parent Folder)
source ../venv/bin/activate
export HF_HOME=../hf_cache

echo "Starting Step 3: Age Parameter Training..."
python train_age.py
