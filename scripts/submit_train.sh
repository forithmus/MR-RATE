#!/bin/bash
# --- JOB METADATA ---
#SBATCH --job-name=mrrate_train_mock
#SBATCH --output=mrrate_train_mock_%j.out
#SBATCH --account=a135
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -p debug
#SBATCH --mem=450G
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00

# --- CONFIGURABLE PARAMETERS ---
FUSION_MODE=${FUSION_MODE:-"late"}
POOLING_STRATEGY=${POOLING_STRATEGY:-"simple_attn"}
SPACE=${SPACE:-"native_space"}
NORMALIZER=${NORMALIZER:-"zscore"}

DATA_FOLDER="/iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/mock_data"
JSONL_FILE="/iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/mock_data/reports.jsonl"
RESULTS_FOLDER="/iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/mr_rate_results"
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-101}
LR=${LR:-1e-5}
RESUME=${RESUME:-0}
USE_WANDB=${USE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-"mr-rate"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}

# --- VALIDATION ---
if [[ ! "$FUSION_MODE" =~ ^(early|mid_cnn|late|late_attn)$ ]]; then
    echo "Error: Invalid fusion mode '$FUSION_MODE'"
    exit 1
fi
if [[ ! "$POOLING_STRATEGY" =~ ^(simple_attn|cross_attn|gated)$ ]]; then
    echo "Error: Invalid pooling strategy '$POOLING_STRATEGY'"
    exit 1
fi

# --- JOB INFO ---
echo "==================================================================================="
echo "MR-RATE Training (Mock Data)"
echo "==================================================================================="
echo "FUSION MODE:      $FUSION_MODE"
echo "POOLING STRATEGY: $POOLING_STRATEGY"
echo "SPACE:            $SPACE"
echo "NORMALIZER:       $NORMALIZER"
echo "NUM_TRAIN_STEPS:  $NUM_TRAIN_STEPS"
echo "LR:               $LR"
echo "RESUME:           $RESUME"
echo "USE_WANDB:        $USE_WANDB"
echo "SLURM_JOB_ID:     $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_NNODES:     $SLURM_NNODES"
echo "Running on host:  $(hostname)"
echo "Start time:       $(date)"
echo "==================================================================================="
echo ""

# --- ENVIRONMENT ---
export PYTHONUNBUFFERED=1
source /iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/venv/bin/activate
export PYTHONPATH="/iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/MR-RATE/mr_rate:/iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/MR-RATE/vision_encoder:/iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/MR-RATE/scripts:$PYTHONPATH"

cd /iopsstor/scratch/cscs/ihamamci/ct_clip/mr-rate-github/MR-RATE/scripts

# --- DISTRIBUTED SETUP ---
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"

export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1

# --- BUILD EXTRA FLAGS ---
EXTRA_FLAGS=""
if [[ "$RESUME" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --resume"
fi
if [[ "$USE_WANDB" == "1" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --wandb --wandb_project $WANDB_PROJECT"
    if [[ -n "$WANDB_RUN_NAME" ]]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --wandb_run_name $WANDB_RUN_NAME"
    fi
fi

# --- LAUNCH ---
GPUS_PER_NODE=4

srun -N $SLURM_NNODES -n $SLURM_NNODES bash -c "
accelerate launch \
    --multi_gpu \
    --num_machines=$SLURM_NNODES \
    --num_processes=$((SLURM_NNODES * $GPUS_PER_NODE)) \
    --mixed_precision bf16 \
    --machine_rank=\$SLURM_PROCID \
    --main_process_ip=$head_node_ip \
    --main_process_port=29500 \
    run_train.py \
        --fusion_mode $FUSION_MODE \
        --pooling_strategy $POOLING_STRATEGY \
        --space $SPACE \
        --normalizer $NORMALIZER \
        --data_folder $DATA_FOLDER \
        --jsonl_file $JSONL_FILE \
        --results_folder $RESULTS_FOLDER \
        --num_train_steps $NUM_TRAIN_STEPS \
        --lr $LR \
        $EXTRA_FLAGS
"

# --- JOB COMPLETION ---
echo ""
echo "==================================================================================="
echo "Job finished with exit code $?."
echo "End time: $(date)"
echo "==================================================================================="
