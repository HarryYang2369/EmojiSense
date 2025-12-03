#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=swabhas_1765
#SBATCH --job-name=gen_age
#SBATCH --output=gen_age_%j.log

module purge
module load ver/2506
module load gcc/15.2.0
module load python/3.13.2

source ../venv/bin/activate
export HF_HOME=../hf_cache

echo "Generating Age Data with Mistral..."
python gen_age_llm.py
