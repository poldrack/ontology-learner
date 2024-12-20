#! /usr/bin/env python3

from pathlib import Path

if __name__ == '__main__':

    runfile = open('run_bertopic_jobs.sh', 'w')
    for n_neighbors in [25, 50, 100, 200]:
        for min_cluster_size in [50, 100, 200]:
            runfile.write(f'bash bertopic.sh {n_neighbors} {min_cluster_size}\n')
    runfile.close()
