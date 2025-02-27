#!/bin/bash

#SBATCH --account=CHEM030406
#SBATCH --job-name=hoomdTest
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=100M

## Direct output to the following files.
## (The %j is replaced by the job id.)
##SBATCH -e hoomdTest%j.txt
#SBATCH -o hoomdTest%j.txt

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job:
echo "Running on host $(hostname)"
#echo "Started on $(date)"
#echo "Directory is $(pwd)"
echo "Slurm job ID is ${SLURM_JOBID}"
echo "This jobs runs on the following machines:"
echo "${SLURM_JOB_NODELIST}"
printf "\n"



# Submit
source /user/home/kq21278/miniforge3/etc/profile.d/conda.sh
conda activate parallel_hoomd
module load openmpi/5.0.3 cuda

python init.py ID=9 rodLength=5 numSolvents=0 numRods=250 drivingForceMagnitude=0
mpirun -n 8 python runSim.py ID=9
python animator.py ID=9
python analyser.py ID=9


# Output the end time
#printf "\n\n"
#echo "Ended on: $(date)"
