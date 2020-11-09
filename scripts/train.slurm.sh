#!/bin/bash
#SBATCH --output=slurm_out/slurm-%A_%a.out
#SBATCH --error=slurm_out/slurm-%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:4
#SBATCH --mem=50g
#SBATCH --cpus-per-task=10
#SBATCH -t 0
#SBATCH --exclude=compute-0-19,compute-0-15,compute-0-17,compute-0-31
##SBATCH --nodelist=compute-0-26


alpha=0.1
free_bits=5

taskid=${SLURM_ARRAY_TASK_ID}

declare -a arch_list=("yelp_large" "yelp_large_s" "yelp_large_xs")
arch=${arch_list[$taskid]}

# bash scripts/train.sh -g 0 -a ${alpha} -f
bash scripts/train.sh -g 0,1,2,3 -d yelp_large -s ${arch} -r 1 -a 1 -f 0
# bash scripts/train.sh -g 0 -c lm_baseline -d yelp_large -s ${arch}

