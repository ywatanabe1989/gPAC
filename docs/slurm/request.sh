#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-04 04:38:00 (ywatanabe)"
# File: ./.claude-worktree/gPAC/docs/slurm/request.sh

#SBATCH --job-name=gpac-persistent
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@3600

# In SLURM:

# `--ntasks`: Total number of parallel processes across all nodes
# - Here: 4 processes (one per GPU)

# `--cpus-per-task`: CPU cores allocated to each task
# - Here: 8 CPU cores per process/task
# - Total CPU cores = ntasks × cpus-per-task = 4 × 8 = 32 cores

# Think of:
# - Each task as a separate process that can use one GPU
# - Each task gets 8 CPU cores to support its GPU operations

# Parameters
PERSISTENT_FILE="./slurm-persistent.txt"

# Export node information for other scripts to use
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST" > $PERSISTENT_FILE
echo "SLURM_JOB_ID=$SLURM_JOB_ID" >> $PERSISTENT_FILE
echo "SLURM_HEAD_NODE=$(hostname)" >> $PERSISTENT_FILE
echo "SLURM_HEAD_IP=$(hostname -i | awk '{print $1}')" >> $PERSISTENT_FILE
echo "SLURM_NNODES=$SLURM_JOB_NUM_NODES" >> $PERSISTENT_FILE
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}" >> $PERSISTENT_FILE
echo "TIMESTAMP=$(date '+%Y%m%d_%H%M%S')" >> $PERSISTENT_FILE

echo "Node information saved:"
cat $PERSISTENT_FILE

usage() {
    cat << 'EOH'
Usage: sbatch request.sh

Requests persistent 4-GPU A100 node for 7 days with auto-resubmission.

Resources:
  - 4 GPUs (A100)
  - 32 CPU cores (8 per GPU)
  - 256GB RAM
  - 7 days runtime

After allocation, use:
  ./slurm_run.sh <command>  # Run commands on node
  ./slurm_login.sh          # SSH login to node
EOH
}

cleanup() {
    rm -f $PERSISTENT_FILE
}
trap cleanup EXIT

resubmit_job() {
    sbatch $0
}
trap 'resubmit_job' USR1

main() {
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        usage
        exit 0
    fi

    if [[ -n "$SLURM_JOB_ID" ]]; then
        echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST" > $PERSISTENT_FILE
        echo "SLURM_JOB_ID=$SLURM_JOB_ID" >> $PERSISTENT_FILE
        echo "SLURM_HEAD_NODE=$(hostname)" >> $PERSISTENT_FILE
        echo "SLURM_HEAD_IP=$(hostname -i | awk '{print $1}')" >> $PERSISTENT_FILE
        echo "SLURM_NNODES=$SLURM_JOB_NUM_NODES" >> $PERSISTENT_FILE
        echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE}" >> $PERSISTENT_FILE
        echo "TIMESTAMP=$(date '+%Y%m%d_%H%M%S')" >> $PERSISTENT_FILE

        echo "Node information saved:"
        cat $PERSISTENT_FILE

        while true; do
            sleep 3600
        done
    else
        if [[ -f "$PERSISTENT_FILE" ]]; then
            echo "Node already allocated. Use:"
            echo "  ./slurm_run.sh <command>  # to run commands"
            echo "  ./slurm_login.sh          # to login"
            exit 1
        fi
    fi
}

main "$@"

# sbatch ./docs/slurm/slurm_request.sh

# EOF
