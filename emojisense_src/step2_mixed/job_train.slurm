#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --account=swabhas_1765
#SBATCH --job-name=train_mix
#SBATCH --output=train_mix_%j.log

# 1. Load Modules
module purge
module load ver/2506
module load gcc/15.2.0
module load python/3.13.2

# 2. Activate Venv (Parent Directory)
source ../venv/bin/activate

# 3. Point HF Cache to Parent Directory
export HF_HOME=../hf_cache

# 4. Run
echo "Training Mixed Model..."
python train_mixed.py
