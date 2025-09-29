#!/bin/bash

#SBATCH --partition part-group_25b505
#SBATCH --gpus 8
#SBATCH --time 336:00:00
#SBATCH --output my_outputs/logs/%x-%j.out
module load hpcx/v2.18.1-cuda12
module load cuda/12.6.3
module load cudnn/9.10.0.56_cuda12
module load nccl/2.24.3-1-cuda-12.6.3
echo "[INFO]: Modules loaded"

source /home/group_25b505/group_6/workspace/user_00031_25b505/miniconda3/bin/activate
conda activate gr00t

cd /home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T

#dataset_dir=${ABCI_GROUP_DIR}/data/libero/libero_object_image

#steps=60000
warmup_ratio=0.02
#batch_size=128

#task=2025-06-v1.0_012725
#task=2025-06-v3.0
#dataset_dir=/home/group_25b505/group_6/workspace/user_00085_25b505/data/hsr_${task}

#task=2025-05-06-07-v3.0 #TODO
#task=2025-05-06-07-v3.0-success-only #TODO
task=widowx #TODO
#task=2025-07-v3.0 #TODO
dataset_dir1=/home/group_25b505/group_6/workspace/user_00077_25b505/SimplerEnv/Isaac-GR00T/input/widowx_data_InternData/InternData-BridgeV2

#steps=2000000
steps=60000
#steps=100000
#steps=50000

#batch_size=8
#batch_size=16
#batch_size=32
#batch_size=64
batch_size=128
#batch_size=256

#num_workers=8
#num_workers=16
num_workers=32
#num_workers=64

num_gpus=8
#num_gpus=4
#num_gpus=1

lr=1e-4

#sample_every_n=6
# sample_every_n=3
# hz=$((30 / ${sample_every_n}))


output_dir="./my_outputs/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_gpu-${num_gpus}_lr-${lr}_compile-None" #TODO

mkdir -p ${output_dir}

#    --dataset_path ${dataset_dir} \
#time python scripts/gr00t_finetune.py \

#TODO
#    --dataset_path ${dataset_dir3} \
#

#(time python scripts/gr00t_finetune.py \
(time python scripts/gr00t_finetune_ep_pa.py \
    --output_dir ${output_dir} \
    --video_backend torchvision_av \
    --data_config widowx \
    --dataset_path ${dataset_dir1}\
    --num-gpus ${num_gpus} \
    --dataloader_num_workers ${num_workers} \
    --max_steps ${steps} \
    --batch_size ${batch_size} \
    --warmup_ratio ${warmup_ratio} \
    --learning_rate ${lr} \
) >& ${output_dir}/train.log
#) >& ${output_dir}/train2.log


