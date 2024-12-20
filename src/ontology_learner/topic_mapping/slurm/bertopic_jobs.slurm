#!/bin/sh

#SBATCH --job-name=JOBNAME
#SBATCH -p beta
#SBATCH --nodes=1
#SBATCH -A marlowe-m000047
#SBATCH -G 8
#SBATCH --time=12:00:00
#SBATCH --error=slurm-%A_%a.err

module load slurm
module load cudatoolkit
module load cudnn/cuda12/9.3.0.75



bash bertopic.sh 25 50
bash bertopic.sh 25 100
bash bertopic.sh 25 200
bash bertopic.sh 50 50
bash bertopic.sh 50 100
bash bertopic.sh 50 200
bash bertopic.sh 100 50
bash bertopic.sh 100 100
bash bertopic.sh 100 200
bash bertopic.sh 200 50
bash bertopic.sh 200 100
bash bertopic.sh 200 200
