#!/usr/bin/env bash
#SBATCH --time 48:00:00
#SBATCH --gpus 4
#SBATCH --cpus-per-task 1
module load hpcx/v2.18.1-cuda12
module load cuda/12.6.3
module load cudnn/9.10.0.56_cuda12
module load nccl/2.24.3-1-cuda-12.6.3
echo "[INFO]: Modules loaded"

source /home/group_25b505/group_6/workspace/user_00031_25b505/miniconda3/bin/activate
conda activate gr00t

cd /home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T

#python scripts/gr00t_finetune.py --dataset-path /home/group_25b505/group_6/data/tmc_new --num-gpus 4 --output_dir ckpt-tmc_new --video_backend torchvision_av --batch-size 128 --data_config hsr --max_steps 20000 --save_steps 5000

python scripts/gr00t_finetune_ep.py --dataset-path /home/group_25b505/dataset/hsr/processed/2025-07-v3.0-success-only --num-gpus 4 --output_dir ckpt-microwave --video_backend torchvision_av --batch-size 32 --data_config hsr_v2 --max_steps 40000 --save_steps 5000