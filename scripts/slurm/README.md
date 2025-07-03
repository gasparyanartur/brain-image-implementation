# Launching scripts on SLURM

In my case, I launch scripts on Berzelius, a SLURM cluster. 
If running on a different cluster, some adaptations might be needed.


#SBATCH --job-name=$JOBNAME
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 100G
#SBATCH --output logs/%x/slurm/%j.out
#SBATCH --time 2-00:00:00
#SBATCH --account 