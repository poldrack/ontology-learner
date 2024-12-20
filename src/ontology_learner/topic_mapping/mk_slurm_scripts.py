#! /usr/bin/env python3

from pathlib import Path

if __name__ == '__main__':

    # read stub file
    with open('bertopic.slurm_stub', 'r') as f:
        stub = f.read()
    
    runfile = open('run_bertopic_jobs.sh', 'w')
    slurm_dir = Path('slurm')
    slurm_dir.mkdir(exist_ok=True)
    for n_neighbors in [25, 50, 100, 200]:
        for min_cluster_size in [50, 100, 200]:
            jobname = f'bertopic_{n_neighbors}_{min_cluster_size}'
            commands = stub.replace('JOBNAME', jobname)
            commands = commands.replace('NNEIGBORS', str(n_neighbors))
            commands = commands.replace('MINCLUST', str(min_cluster_size))
            jobfile = slurm_dir / f'{jobname}.slurm'
            with open(jobfile, 'w') as f:
                f.write(commands)
            runfile.write(f'sbatch {jobfile}\n')
    runfile.close()
