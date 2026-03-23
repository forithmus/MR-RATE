#!/bin/bash
# --- JOB METADATA ---
#SBATCH --job-name=mrrate_train
#SBATCH --output=logs/mrrate_train_%j.out
#SBATCH --error=logs/mrrate_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=450G
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00

# --- CONFIGURABLE PARAMETERS ---
ENCODER=${ENCODER:-"vjepa2"}
VJEPA21_CHECKPOINT=${VJEPA21_CHECKPOINT:-""}
CHUNK_SIZE=${CHUNK_SIZE:-64}
FUSION_MODE=${FUSION_MODE:-"late"}
POOLING_STRATEGY=${POOLING_STRATEGY:-"simple_attn"}
NORMALIZER=${NORMALIZER:-"percentile"}
SPACE=${SPACE:-"native_space"}

DATA_FOLDER=${DATA_FOLDER:-"/path/to/data"}
JSONL_FILE=${JSONL_FILE:-"/path/to/reports.jsonl"}
RESULTS_FOLDER=${RESULTS_FOLDER:-"./mr_rate_results"}
SPLITS_CSV=${SPLITS_CSV:-""}
SPLIT=${SPLIT:-"train"}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-""}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-100001}
LR=${LR:-1e-5}
RESUME=${RESUME:-0}
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-"mr-rate"}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-""}

# --- ENVIRONMENT (adjust these to your setup) ---
ENV_PYTHON=${ENV_PYTHON:-"python"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"

# --- VALIDATION ---
if [[ ! "$ENCODER" =~ ^(vjepa2|vjepa21|vjepa2_sliding|vjepa21_sliding)$ ]]; then
    echo "Error: Invalid encoder '$ENCODER'. Choose from: vjepa2, vjepa21, vjepa2_sliding, vjepa21_sliding"
    exit 1
fi
if [[ ! "$FUSION_MODE" =~ ^(early|mid_cnn|late|late_attn)$ ]]; then
    echo "Error: Invalid fusion mode '$FUSION_MODE'"
    exit 1
fi
if [[ ! "$POOLING_STRATEGY" =~ ^(simple_attn|cross_attn|gated)$ ]]; then
    echo "Error: Invalid pooling strategy '$POOLING_STRATEGY'"
    exit 1
fi
if [[ "$ENCODER" == *"vjepa21"* && -z "$VJEPA21_CHECKPOINT" ]]; then
    echo "Error: --vjepa21_checkpoint is required for encoder '$ENCODER'"
    exit 1
fi

# --- JOB INFO ---
echo "==================================================================================="
echo "MR-RATE Training"
echo "==================================================================================="
echo "ENCODER:          $ENCODER"
[[ "$ENCODER" == *"sliding"* ]] && echo "CHUNK_SIZE:       $CHUNK_SIZE"
[[ "$ENCODER" == *"vjepa21"* ]] && echo "VJEPA21_CKPT:     $VJEPA21_CHECKPOINT"
echo "FUSION MODE:      $FUSION_MODE"
echo "POOLING STRATEGY: $POOLING_STRATEGY"
echo "NORMALIZER:       $NORMALIZER"
echo "DATA_FOLDER:      $DATA_FOLDER"
echo "NUM_TRAIN_STEPS:  $NUM_TRAIN_STEPS"
echo "LR:               $LR"
echo "RESUME:           $RESUME"
echo "USE_WANDB:        $USE_WANDB"
echo "SLURM_JOB_ID:     $SLURM_JOB_ID"
echo "SLURM_NNODES:     $SLURM_NNODES"
echo "Running on host:  $(hostname)"
echo "Start time:       $(date)"
echo "==================================================================================="
echo ""

# --- ENVIRONMENT ---
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE_DIR}/mr_rate:${CODE_DIR}/vision_encoder:${CODE_DIR}/scripts:$PYTHONPATH"

cd "${CODE_DIR}/scripts"
mkdir -p logs

# --- DISTRIBUTED SETUP ---
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"

export NCCL_TIMEOUT=7200
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- BUILD EXTRA FLAGS ---
EXTRA_FLAGS=""
EXTRA_FLAGS="$EXTRA_FLAGS --encoder $ENCODER"

if [[ "$ENCODER" == *"vjepa21"* ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --vjepa21_checkpoint $VJEPA21_CHECKPOINT"
fi
if [[ "$ENCODER" == *"sliding"* ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --chunk_size $CHUNK_SIZE"
fi
if [[ -n "$SPLITS_CSV" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --splits_csv $SPLITS_CSV --split $SPLIT"
fi
if [[ -n "$PRETRAINED_WEIGHTS" ]]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --pretrained_weights $PRETRAINED_WEIGHTS"
fi
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
$ENV_PYTHON -m accelerate.commands.launch \
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
        --normalizer $NORMALIZER \
        --space $SPACE \
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
