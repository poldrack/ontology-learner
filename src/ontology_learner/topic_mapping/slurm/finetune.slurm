#!/bin/sh

#SBATCH --job-name=embed_fit
#SBATCH -p beta
#SBATCH --nodes=1
#SBATCH -A marlowe-m000047
#SBATCH -G 8
#SBATCH --time=48:00:00
#SBATCH --error=slurm-%A_%a.err

module load slurm
module load cudatoolkit
module load cudnn/cuda12/9.3.0.75

source /projects/m000047/ontology-learner/.venv/bin/activate
python -m torch.distributed.run /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/embedding_finetuning.py
