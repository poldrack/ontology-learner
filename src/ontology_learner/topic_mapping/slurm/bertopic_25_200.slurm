#!/bin/sh

#SBATCH --job-name=bertopic_25_200
#SBATCH -p beta
#SBATCH --nodes=1
#SBATCH -A marlowe-m000047
#SBATCH -G 1
#SBATCH --time=1:00:00
#SBATCH --error=slurm-%A_%a.err

module load slurm
module load cudatoolkit
module load cudnn/cuda12/9.3.0.75

srun bertopic.sh 25 200
