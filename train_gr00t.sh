#!/usr/bin/env bash
#SBATCH -p part-group_25b505
#SBATCH --nodelist=aic-gh2b-310049
#SBATCH --job-name=train_gr00t_hsr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --gpus=2
#SBATCH --mem=500G
#SBATCH --output=logs/gr00t-ft-%j.out
#SBATCH --error=logs/gr00t-ft-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<your mail>

module load hpcx/v2.18.1-cuda12
module load cuda/12.6.3
module load cudnn/9.10.0.56_cuda12
module load nccl/2.24.3-1-cuda-12.6.3

### Fix the path. ###
export CONDA_ENVS_PATH=/home/group_25b505/group_6/workspace/user_00083_25b505/conda_envs
export HF_HOME=/home/group_25b505/group_6/workspace/user_00083_25b505/huggingface_home
cd /home/group_25b505/group_6/workspace/user_00083_25b505/Isaac-GR00T_N1d6
### Fix the path. ###

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_n1.6 # Conda environment with ffmpeg<=7 installed

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

uv sync --refresh --python 3.10
uv pip install -e .

export OMP_NUM_THREADS=28

RUN_ID="$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=2

BASE_MODEL_PATH="nvidia/GR00T-N1.6-3B"
DATASET_PATH="/home/group_25b505/group_6/data/airoa-hsr-all-v1.0-202505-09-success-stat-curation_weblab-kyutech-fastlabel-telexistence-final"
EMBODIMENT_TAG="NEW_EMBODIMENT"
### Fix the path. ###
MODALITY_CONFIG_PATH="/home/group_25b505/group_6/workspace/user_00083_25b505/Isaac-GR00T_N1d6/examples/HSR/hsr_config.py"
### Fix the path. ###

OUTPUT_DIR="outputs/hsr_full_$RUN_ID"
SAVE_TOTAL_LIMIT=5
SAVE_STEPS=2000
MAX_STEPS=2000
GLOBAL_BATCH_SIZE=128
DATALOADER_NUM_WORKERS=8

COLOR_JITTER_BRIGHTNESS=0.3
COLOR_JITTER_CONTRAST=0.4
COLOR_JITTER_SATURATION=0.5
COLOR_JITTER_HUE=0.08

ARGS=(
  --base-model-path "$BASE_MODEL_PATH"
  --dataset-path "$DATASET_PATH"
  --embodiment-tag "$EMBODIMENT_TAG"
  --modality-config-path "$MODALITY_CONFIG_PATH"
  --num-gpus "$NUM_GPUS"
  --output-dir "$OUTPUT_DIR"
  --save-total-limit "$SAVE_TOTAL_LIMIT"
  --save-steps "$SAVE_STEPS"
  --max-steps "$MAX_STEPS"
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --color-jitter-params brightness "$COLOR_JITTER_BRIGHTNESS" contrast "$COLOR_JITTER_CONTRAST" saturation "$COLOR_JITTER_SATURATION" hue "$COLOR_JITTER_HUE"
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)

# ---- launch ----
mkdir -p "logs" "$OUTPUT_DIR"

CMD=(
  uv run torchrun
  --nproc_per_node="$NUM_GPUS"
  gr00t/experiment/launch_finetune.py
  "${ARGS[@]}"
)

printf 'Running: '
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
