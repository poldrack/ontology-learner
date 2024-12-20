#!/usr/bin/env bash
# arguments are n_neighbors and min_cluster_size

#eval "$(conda shell.bash hook)"
#conda activate bertopic
python -m torch.distributed.run fit_bertopic_model.py --n_neighbors $1 --min_cluster_size $2
