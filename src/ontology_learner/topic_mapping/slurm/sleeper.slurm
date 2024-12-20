#!/bin/sh

#SBATCH --job-name=test
#SBATCH -p beta
#SBATCH --nodes=1
#SBATCH -A marlowe-m000047
#SBATCH -G 8
#SBATCH --time=12:00:00
#SBATCH --error=slurm-%A_%a.err

module load slurm
module load cudatoolkit
module load cudnn/cuda12/9.3.0.75

source /projects/m000047/ontology-learner/.venv/bin/activate
sleep 6000
