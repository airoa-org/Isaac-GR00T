#!/bin/bash

#SBATCH -p part-group_25b505
#SBATCH --nodelist=aic-gh2b-310049
#SBATCH --job-name=test_gr00tn1d5_hsr
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=500G
#SBATCH --output=logs/gr00t-ft-%j.out
#SBATCH --error=logs/gr00t-ft-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<email@email>


module load hpcx/v2.18.1-cuda12
module load cuda/12.6.3
module load cudnn/9.10.0.56_cuda12
module load nccl/2.24.3-1-cuda-12.6.3
echo "[INFO]: Modules loaded"

### Fix here ###
export CONDA_ENVS_PATH=/home/group_25b505/group_6/workspace/user_00083_25b505/conda_envs
export HF_HOME=/home/group_25b505/group_6/workspace/user_00083_25b505/huggingface_home
cd /home/group_25b505/group_6/workspace/user_00083_25b505/Isaac-GR00T_gr6HSR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gr00t_gr6HSR
### Fix here ###

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56
export OPENBLAS_NUM_THREADS=56
export NUMEXPR_NUM_THREADS=56

#dataset_dir=${ABCI_GROUP_DIR}/data/libero/libero_object_image

#steps=60000
warmup_ratio=0.02
#batch_size=128

#task=2025-06-v1.0_012725
#task=2025-06-v3.0
#dataset_dir=/home/group_25b505/group_6/workspace/user_00085_25b505/data/hsr_${task}

#task=2025-05-06-07-v3.0 #TODO
#task=2025-05-06-07-v3.0-success-only #TODO
task=airoa-hsr-all-v1.0-202505-09-success-stat-curation #TODO
#task=2025-07-v3.0 #TODO
#dataset_dir1=/home/group_25b505/group_6/workspace/user_00085_25b505/projects/Isaac-GR00T/demo_data/2025-07-v3.0-success-only_aist
#dataset_dir1=/home/group_25b505/group_6/workspace/user_00085_25b505/projects/Isaac-GR00T/demo_data/2025-07-v3.0-success-only_aist_Microwave_the_item
#dataset_dir2=/home/group_25b505/group_6/workspace/user_00085_25b505/projects/Isaac-GR00T/demo_data/2025-08-v3.0-success-only_aist
#dataset_dir5=/home/group_25b505/dataset/hsr/processed/stage2-final-68tasks/airoa-hsr-all-v1.0-202504-202512-task45-success
#dataset_dir8=/home/group_25b505/dataset/hsr/processed/stage2-final-68tasks/airoa-hsr-all-v1.0-202504-202512-task48-success
dataset_dir1=/home/group_25b505/group_6/data/airoa-hsr-all-v1.0-202505-09-success-stat-curation

#steps=2000000
# steps=30000
steps=300
#steps=100000
#steps=50000

#batch_size=8
#batch_size=16
#batch_size=32
#batch_size=64
batch_size=256
#batch_size=256

num_workers=8
#num_workers=16
#num_workers=32
#num_workers=64

# num_gpus=8
num_gpus=4
# num_gpus=1

lr=1e-4

#sample_every_n=6
sample_every_n=1
hz=$((10 / ${sample_every_n}))

#output_dir="./my_outputs/hsr/${task}/steps-${steps}_wr-${warmup_ratio}_bsz-${batch_size}"
#output_dir="./my_outputs/hsr/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_compile-None"
#output_dir="./my_outputs/hsr/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_gpu-${num_gpus}_compile-max-autotune"
#output_dir="./my_outputs/hsr/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_gpu-${num_gpus}_compile-None"
#output_dir="./my_outputs/hsr/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_gpu-${num_gpus}_compile-None"
output_dir="./my_outputs/hsr/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_gpu-${num_gpus}_lr-${lr}_compile-None_hz-${hz}" #TODO
#output_dir="./my_outputs/hsr/${task}/steps-${steps}_bsz-${batch_size}_workers-${num_workers}_gpu-${num_gpus}_lr-1e-5_compile-None" #TODO

mkdir -p ${output_dir}

#    --dataset_path ${dataset_dir} \
#time python scripts/gr00t_finetune.py \

#TODO
#    --dataset_path ${dataset_dir3} \
#

#(time python scripts/gr00t_finetune.py \
time python scripts/gr00t_finetune_ep_pa.py \
    --sample_every_n ${sample_every_n} \
    --output_dir ${output_dir} \
    --video_backend torchvision_av \
    --data_config hsr_v2 \
    --dataset_path ${dataset_dir1}\
    --num-gpus ${num_gpus} \
    --dataloader_num_workers ${num_workers} \
    --max_steps ${steps} \
    --batch_size ${batch_size} \
    --warmup_ratio ${warmup_ratio} \
    --lr-scheduler-backend custom \
    --custom-lr-scheduler gr00t.utils.lr_schedulers:get_cosine_with_soft_restarts_schedule_with_warmup \
    --custom-lr-scheduler-kwargs-json '{"restart_steps":[50,150]}' \
    --learning_rate ${lr}
