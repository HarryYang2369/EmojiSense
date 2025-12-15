#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=swabhas_1765
#SBATCH --job-name=eval_bleu
#SBATCH --output=eval_bleu_%j.log

# 1. Load Modules (Exact same as training)
module purge
module load ver/2506
module load gcc/15.2.0
module load python/3.13.2

# 2. Activate Environment
source venv/bin/activate
export HF_HOME=./hf_cache

# 3. Run Evaluation
echo "Starting Evaluation of All 3 Models..."
python bleu.py
