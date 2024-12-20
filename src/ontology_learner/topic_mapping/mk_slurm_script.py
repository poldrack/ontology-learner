#! /usr/bin/env python3

from pathlib import Path

if __name__ == '__main__':

    # read stub file
    with open('bertopic.slurm_stub', 'r') as f:
        stub = f.read()
    
    runfile = open('bertopic_jobs.slurm', 'w')
    runfile.write(stub + '\n\n')
    for n_neighbors in [25, 50, 100, 200]:
        for min_cluster_size in [50, 100, 200]:
            runfile.write(f'bash bertopic.sh {n_neighbors} {min_cluster_size}\n')
    runfile.close()
