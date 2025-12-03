#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --account=swabhas_1765
#SBATCH --job-name=gen_boomer
#SBATCH --output=gen_boomer_%j.log

module purge
module load ver/2506
module load gcc/15.2.0
module load python/3.13.2

source ../venv/bin/activate
export HF_HOME=../hf_cache

python gen_boomer_20k.py
