#!/bin/bash

#SBATCH --account=CHEM030406
#SBATCH --job-name=hoomdTest
#SBATCH --partition=teach_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem-per-cpu=100M

## Direct output to the following files.
## (The %j is replaced by the job id.)
##SBATCH -e hoomdTest%j.txt
#SBATCH -o hoomdTest%j.txt

# Change to working directory, where the job was submitted from.
cd "${SLURM_SUBMIT_DIR}"

# Record some potentially useful details about the job:
#echo "Running on host $(hostname)"
#echo "Started on $(date)"
#echo "Directory is $(pwd)"
#echo "Slurm job ID is ${SLURM_JOBID}"
#echo "This jobs runs on the following machines:"
#echo "${SLURM_JOB_NODELIST}"
#printf "\n\n"
echo "time(s),average\n"

# Submit
source /user/home/kq21278/miniforge3/etc/profile.d/conda.sh
conda activate parallel_hoomd
module load openmpi/5.0.3 cuda

#mpirun -n 8 python runSim.py
srun --mpi=pmix python runSim.py
wait
python animator.py

# nearest neighbours, pass parameters to different runs of programs
# So want to have one folder with one copy of init.py and runSim.py, but run init.py with parameters i.e. python init.py dt=0.001 ssef=0.1 
# This way we only have to change parameters according to these changes. Probably pass through something like node ID somehow so that it writes out the initial file and metadata with that id name, for input to runSim.py
# This then has to know which files its looking for to then make its own unique files.

# srun (mpi parameters) .py &
# srun (^) .p &
# wait

# Output the end time
#printf "\n\n"
#echo "Ended on: $(date)"
