#!/bin/bash
#SBATCH --job-name=iam_the_parralel_one # Job name
#SBATCH --output=output.txt # Standard output file
#SBATCH --error=error.txt # Standard error file
#SBATCH --tasks=36 # Number of tasks per node
#SBATCH --nodes=1 # Number of nodes
#SBATCH --time=0-01:00 # Maximum runtime (D-HH:MM)
#SBATCH --nodelist=calypso[2] # Specific nodes [Optional]
#SBATCH --gpus-per-node=1 # Require GPUs
#SBATCH -N 1
#srun="srun -n1 -N1 --exclusive"

# Define parallel arguments:
#parallel="parallel -N 1 --delay .2 -j $SLURM_NTASKS --joblog parallel_joblog --resume"
# -N 1              is number of arguments to pass to each job
# --delay .2        prevents overloading the controlling node on short jobs
# -j $SLURM_NTASKS  is the number of concurrent tasks parallel runs, so number of CPUs allocated
# --joblog name     parallel's log file of tasks it has run
# --resume          parallel can use a joblog and this to continue an interrupted run (job resubmitted)

#$parallel "$srun ./run_serpentin arg1:{1}" ::: {1..10}
for j in {1..10}
do
for i in {1..8}
do
    mpirun -n 36 ./run_serpentin $i
done
done

