#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=main
#SBATCH --account=swabhas_1765
#SBATCH --job-name=gen_mix
#SBATCH --output=gen_mix_%j.log

module purge
module load ver/2506
module load gcc/15.2.0
module load python/3.13.2 # Match your VENV version

# Point to the venv in the parent folder
source ../venv/bin/activate

echo "Generating Augmented Data..."
python generate_mixed.py
