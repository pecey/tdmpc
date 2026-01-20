#!/bin/bash

#SBATCH -p gpu
#SBATCH -A r00189
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-2

group_name=${group_name}

logdir=${logdir}

domain=${domain}
task=${task}
tea=${tea}
policy=${policy}
value=${value}


seeds=(10 20 30)
seed=${seeds[$SLURM_ARRAY_TASK_ID]}

# seed=10

today=$(date +'%m-%d-%Y')

run_name=${today}-${domain}-${task}-${rid}-${seed}

# module load python/3.12.4

start_time=`(date +%s)`
echo Start time: ${start_time}

if [[ "${SERVER_NAME}" == "BR200" ]]; then
    echo "Activating /N/slate/palchatt/tdmpc_py3.8 venv"
    source /N/slate/palchatt/tdmpc_py3.8/bin/activate
else
    echo "Activating /N/slate/palchatt/tdmpc_py3.8_quartz"
    source /N/slate/palchatt/tdmpc_py3.8_quartz/bin/activate
fi

export WANDB_DIR=${wandb_path}

cd /N/u/palchatt/BigRed200/tdmpc/
 
# To evaluate model
PYTHONPATH=. python src/train.py task=${domain}-${task} tea=${tea} wandb_project=tdmpc-${domain}-${task} exp_name=${group_name}-${rid} wandb_run_name=run-${rid}-${seed} use_policy=${policy} use_val_backup=${value} seed=${seed} 

end_time=`(date +%s)`
echo End time: ${end_time}