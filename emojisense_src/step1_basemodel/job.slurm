#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=swabhas_1765
#SBATCH --job-name=emoji_train
#SBATCH --output=emoji_train_%j.log

# 1. Clear old modules
module purge

# 2. Unlock the new software stack (CRITICAL STEP)
module load ver/2506

# 3. Load the compiler and python
# Note: I am using the versions you found. 
# If 15.2.0 fails inside the job, change this to gcc/13.3.0
module load gcc/15.2.0 
module load python/3.13.2

# 4. Activate your virtual environment
source venv/bin/activate

# 5. Debugging: Print loaded modules to log so we know what happened
module list

# 6. Force Hugging Face to use your project folder for cache
# (This protects your 100GB home quota)
export HF_HOME=./hf_cache

# 7. Run the training
echo "Starting training on $(hostname)"
python train.py
