"""

#!/bin/bash

#SBATCH -p shared                        # Partition name
#SBATCH -J job_name                         # Job name
(https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E)
#SBATCH -o %x_%j.out                        # Name of stdout output file
#SBATCH -e %x_%j.err                        # Name of stderr error file
#SBATCH -N 1                                # Number of nodes
#SBATCH --ntasks-per-node=1                # MPI cores per node
#SBATCH --cpus-per-task=32
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
    partition="shared",
    num_nodes=1,
    num_threads=4,
    memory=64,
    time="48:00:00",
    account="mcb200029p",
    verbose=False,
    mail_user=None,
    mail_type=None,
):

    ntasks_per_node = 1
    if partition == "shared":
        ntasks_per_node = num_threads

    filename = "submit_" + program_name.replace(".py", ".sh")
    f = open(filename, "w")
    f.writelines(
        [
            "#!/bin/bash\n",
            "\n",
            f"#SBATCH -p {partition}\n",
            f"#SBATCH -J {program_name.replace('.py', '')}\n",
            f"#SBATCH -N {num_nodes}\n",
            f"#SBATCH -t {time}\n",
            f"#SBATCH --ntasks-per-node={ntasks_per_node}\n",
            f"#SBATCH --account={account}\n",
            f"#SBATCH --mem={memory}G\n",
        ]
    )
    # only use cpus per task for compute node jobs (see readme)
    if partition == "compute":
        f.write(f"#SBATCH --cpus-per-task={num_threads}\n")

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
        f.write("#SBATCH -v\n")

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
            f"export OMP_NUM_THREADS={num_threads}\n",
            f"~/.conda/envs/{environment_name}/bin/python -u {program_name} --num_threads {num_threads}\n",
            "\n",
        ]
    )

    f.close()


if __name__ == "__main__":
    program_name = "run_tapered_arm_and_sphere_with_flow.py"
    environment_name = "sopht-examples-env"
    partition = "compute"
    time = "06:00:00"
    num_threads = 32
    account = "uic409"
    mail_user = "atekinal"

    create_submit_file(
        program_name=program_name,
        environment_name=environment_name,
        time=time,
        partition=partition,
        num_threads=num_threads,
        account=account,
        mail_user=mail_user,
    )
