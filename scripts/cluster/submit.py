"""

#!/bin/bash

#SBATCH -p RM-shared                        # Partition name
#SBATCH -J job_name                         # Job name
(https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E)
#SBATCH -o %x_%j.out                        # Name of stdout output file
#SBATCH -e %x_%j.err                        # Name of stderr error file
#SBATCH -N 1                                # Number of nodes
#SBATCH --ntasks-per-node=32                # MPI cores per node
#SBATCH -c 2
#SBATCH --mem=64G
#SBATCH -t 48:00:00                         # Time limit (hh:mm:ss)
#SBATCH --mail-user=NetID@illinois.edu      # Email notification
#SBATCH --mail-type=ALL                     # Notify on state change (BEGIN/END/FAIL/ALL)
#SBATCH --account=mcb200029p

module load anaconda3
date                                        # Print date

echo Job name: $SLURM_JOB_NAME              # Print job name
echo Execution dir: $SLURM_SUBMIT_DIR       # Print submit directory
echo Number of processes: $SLURM_NTASKS     # Print number of processes

source activate (env_name)
conda env list

export OMP_NUM_THREADS=4
$PROJECT/.conda/envs/(env_name)/bin/python -u (program_name)

"""


def create_submit_file(
    program_name,
    environment_name,
    output_file_name=None,
    error_file_name=None,
    partition="RM-shared",
    node=1,
    ntasks_per_node=4,
    cpus_per_task=4,
    memory=64,
    time="48:00:00",
    account="mcb200029p",
    verbose=False,
    mail_user=None,
    mail_type=None,
):

    filename = "submit_" + program_name.replace(".py", ".sh")
    f = open(filename, "w")
    f.writelines(
        [
            "#!/bin/bash\n",
            "\n",
            f"#SBATCH -p {partition}\n",
            f"#SBATCH -J {program_name.replace('.py', '')}\n",
            f"#SBATCH -N {node}\n",
            f"#SBATCH --cpus-per-task={cpus_per_task}\n",
            f"#SBATCH -t {time}\n",
            f"#SBATCH --ntasks-per-node={ntasks_per_node}\n",
            f"#SBATCH --account={account}\n",
            f"#SBATCH --mem={memory}G\n",
        ]
    )

    if not output_file_name:
        output_file_name = "%x_%j.out"
    if not error_file_name:
        error_file_name = "%x_%j.err"

    f.write(f"#SBATCH -o {output_file_name}\n")
    f.write(f"#SBATCH -e {error_file_name}\n")

    if mail_user:
        f.write(f"#SBATCH --mail-user={mail_user}\n")
        if not mail_type:
            mail_type = "ALL"
        f.write(f"#SBATCH --mail-type={mail_type}\n")

    if verbose:
        f.write("SBATCH -v\n")

    f.writelines(
        [
            "\n",
            "module load anaconda3\n",
            "date\n",
            "\n",
            "echo Job name: $SLURM_JOB_NAME\n",
            "echo Execution dir: $SLURM_SUBMIT_DIR\n",
            "echo Number of processes: $SLURM_NTASKS\n",
            "\n",
            f"source activate {environment_name}\n",
            f"export OMP_NUM_THREADS={cpus_per_task}\n",
            f"~/.conda/envs/{environment_name}/bin/python -u {program_name} --num_threads {cpus_per_task}\n",
            "\n",
        ]
    )

    f.close()


if __name__ == "__main__":
    program_name = "run_tapered_arm_and_sphere_with_flow.py"
    environment_name = "sopht-examples-env"
    partition = "shared"
    time = "06:00:00"
    ntasks_per_node = 1
    cpus_per_task = 32
    account = "uic409"
    mail_user = "atekinal"


    create_submit_file(
        program_name=program_name,
        environment_name=environment_name,
        time=time,
        partition=partition,
        ntasks_per_node=ntasks_per_node,
        cpus_per_task=cpus_per_task,
        account = account,
        mail_user = mail_user,
    )
