"""

#!/bin/bash

#SBATCH -p shared                        # Partition name
#SBATCH -J job_name                         # Job name
(https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E)
#SBATCH -o %x_%j.out                        # Name of stdout output file
#SBATCH -e %x_%j.err                        # Name of stderr error file
#SBATCH -N 1                                # Number of nodes
#SBATCH --ntasks-per-node=32                # cores to use per node
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

export OMP_NUM_THREADS=32
$PROJECT/.conda/envs/(env_name)/bin/python -u (program_name)

"""
from typing import Optional


def create_submit_file(
    program_name: str,
    environment_name: str,
    cluster_info_dict: dict,
    output_file_name: Optional[str] = None,
    error_file_name: Optional[str] = None,
    partition: str = "shared",
    num_nodes: int = 1,
    num_threads: int = 4,
    memory: int = 64,
    time: str = "48:00:00",
    verbose: bool = False,
    mail_user: Optional[str] = None,
    mail_type: Optional[str] = None,
    other_cli_arguments: str = "",
) -> None:

    filename = "submit_" + program_name.replace(".py", ".sh")
    f = open(filename, "w")
    f.writelines(
        [
            "#!/bin/bash\n",
            "\n",
            f"#SBATCH -p {cluster_info_dict.get(partition)}\n",
            f"#SBATCH -J {program_name.replace('.py', '')}\n",
            f"#SBATCH -N {num_nodes}\n",
            f"#SBATCH -t {time}\n",
            f"#SBATCH --ntasks-per-node={num_threads}\n",
            f"#SBATCH --account={cluster_info_dict.get('account')}\n",
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
            f"python -u {program_name} --num_threads {num_threads} {other_cli_arguments}\n",
            "\n",
        ]
    )

    f.close()


if __name__ == "__main__":
    expanse_info_dict = {"account": "uic409", "shared": "shared", "compute": "compute"}
    bridges_info_dict = {
        "account": "mcb200029p",
        "shared": "RM-shared",
        "compute": "RM",
    }
    program_name = "run_tapered_arm_and_sphere_with_flow.py"
    environment_name = "sopht-examples-env"
    partition = "compute"
    time = "06:00:00"
    num_threads = 32
    mail_user = "atekinal"

    create_submit_file(
        program_name=program_name,
        environment_name=environment_name,
        cluster_info_dict=expanse_info_dict,
        time=time,
        partition=partition,
        num_threads=num_threads,
        mail_user=mail_user,
    )
