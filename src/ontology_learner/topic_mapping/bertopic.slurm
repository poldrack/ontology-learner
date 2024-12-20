#!/bin/sh

#SBATCH --job-name=bertopic
#SBATCH -p beta
#SBATCH --nodes=1
#SBATCH -A marlowe-m000047
#SBATCH -G 8
#SBATCH --time=8:00:00
#SBATCH --error=slurm-%A_%a.err

module load slurm
module load cudatoolkit
module load cudnn/cuda12/9.3.0.75

srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 25 50
srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 50 50
srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 50 100
srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 50 200
srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 100 100
srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 100 200
srun /projects/m000047/ontology-learner/src/ontology_learner/topic_mapping/bertopic.sh 200 200

