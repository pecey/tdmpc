#!/bin/bash
domain=$1
task=$2
tea=$3
policy=$4
value=$5
group_name=$6

today=$(date +'%m-%d-%Y')

# Run identifier
rid=$(date +%s)

# Path for slurm logs
SLURM_LOG_PATH=/N/scratch/palchatt/TDMPC/slurm_logs/${domain}-${task}/${today}
mkdir -p ${SLURM_LOG_PATH}

# Path for WandB logs
wandb_path=/N/scratch/palchatt/TDMPC/wandb/${domain}-${task}
mkdir -p ${wandb_path}

# Path for logging
logdir=/N/scratch/palchatt/TDMPC/results/${domain}-${task}
mkdir -p ${logdir}

# For triggering using GPU
jid1=$(sbatch --export=ALL,domain=${domain},task=${task},wandb_path=${wandb_path},logdir=${logdir},rid=${rid},group_name=${group_name},policy=${policy},value=${value},tea=${tea} --parsable -J tdmpc-${domain}-${task}-${group_name} -o ${SLURM_LOG_PATH}/${rid}_${domain}_${task}_%a.out  -e ${SLURM_LOG_PATH}/${rid}_${domain}_${task}_%a.err /N/u/palchatt/BigRed200/tdmpc/slurm_scripts/exp_gpu.sh)
