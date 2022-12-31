# Submission scripts for supercomputing clusters

Currently, we have a template for submitting jobs on Slurm job queue systems (expanse, bridges2, etc.).
You can find the template in `submit_slurm.sh`.

### Some notes on Expanse
On Expanse, there are 128 cores on each node. This means you can run simulation utilizing shared memory parallelism with all 128 cores. Since Expanse charge based on core/hour and not node/hour, it is recommended that you run simulations requesting only the cores you need for your computation.

If you are using all 128 cores, use the `#SBATCH` directives below:
```
#SBATCH -p compute   # Using compute here since we are using all cores anyway so they will charge at the 128cores/hour rate
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
```

If you are not using all 128 cores on a single node (say you only need 4 cores for your shared memory computation), use the `#SBATCH` directives below:
```
#SBATCH -p shared  # Using shared here so that they will only charge at the 4cores/hour rate
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
```

### Some notes on Stampede2
On Stampede2, setting up memory is not available please do not set memory.

`anaconda` is not available as a module and needs to be installed manually. 

Additionally, while setting up `sopht` the `PYTHONPATH` variable holds the path to old Python 2.7
version hdf5. To avoid this remnant from affecting `sopht` installation, run the following command
after you create the Conda environment and before you install the package:
```
echo $PYTHONPATH
unset PYTHONPATH
echo $PYTHONPATH
```
This should remove the older references and ensure smooth installation of `sopht`.