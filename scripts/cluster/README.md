# Submission scripts for supercomputing clusters

Currently, we have a template for submitting jobs on Slurm job queue systems (expanse, bridges2, etc.).
You can find the template in `submit_slurm.sh`.

### Some notes on Expanse
On Expanse, there are 128 cores on each node. This means you can run simulation utilizing shared memory parallelism with all 128 cores. Since Expanse charge based on core/hour and not node/hour, it is recommended that you run simulations requesting only the cores you need for your computation.

If you are using all 128 cores, use the `#SBATCH` directives below:
```
#SBATCH -p compute   # Using compute here since we are using all cores anyway so they will charge at the 128cores/hour rate
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
```

If you are not using all 128 cores on a single node (say you only need 4 cores for your shared memory computation), use the `#SBATCH` directives below:
```
#SBATCH -p shared  # Using shared here so that they will only charge at the 4cores/hour rate
#SBATCH -N 1
#SBATCH --ntasks-per-node=4  # this is somewhat counter-intuitive but will only speed up as expected when using shared queue partition
```

Please note the difference in `--ntasks-per-node` directives for the two different queue partitions (`compute` and `shared`). The behavior is slightly and perhaps unintuitively different, depending on which partition you are using.
