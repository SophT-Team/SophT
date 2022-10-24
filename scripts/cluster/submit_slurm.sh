#!/bin/bash

#SBATCH -J test_job
#SBATCH -o %N.%j.o         # Name of stdout output file
#SBATCH -e %N.%j.e         # Name of stderr error file
#SBATCH -p compute                      # Queue (partition) name
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH -t 00:10:00                    # Run time (hh:mm:ss)
#SBATCH --mail-user=email@email.com
#SBATCH --mail-type=all                # Send email at begin, end, or fail of job
#SBATCH --account=TG-MCB190004

# Other commands must follow all #SBATCH directives...

# file to be executed
PROGNAME="flow_past_sphere_case.py"

# print some details
date
echo Job name: $SLURM_JOB_NAME
echo Execution dir: $SLURM_SUBMIT_DIR
echo Number of processes: $SLURM_NTASKS

# load anaconda and activate environment
module load anaconda3
conda activate sopht-examples-env
which python

# set smp num threads the same as ---cpus-per-task or --ntasks-per-node
# see README.md for details
SMP_NUM_THREADS=32
export OMP_NUM_THREADS=$SMP_NUM_THREADS

# execute the program
python ${PROGNAME} --num_threads=$SMP_NUM_THREADS
