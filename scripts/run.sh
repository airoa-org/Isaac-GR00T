#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -P gag51454
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
cd /groups/gaf51379/rfm/Isaac-GR00T
module load nvhpc/24.9
module load hpcx/2.20
source ~/.bashrc
conda activate gr00tv2

nvidia-smi
#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/realur5demoenv1_1.0.0_lerobot --num-gpus 1 --batch-size 128 --data_config ur5rmb --output_dir ckpt

#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/MujocoHsrTidyup_20250617_093824 --num-gpus 1 --batch-size 128 --data_config hsrrmb --output_dir ckpt_rmb --video-backend torchvision_av

#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task --num-gpus 8 --batch-size 128 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5 --video-backend torchvision_av --max_steps 60000
#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task_pg --num-gpus 8 --batch-size 32 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_pg --video-backend torchvision_av --max_steps 60000
#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task_pg --num-gpus 1 --batch-size 128 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_pg_128 --video-backend torchvision_av --max_steps 60000

#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task_pg --num-gpus 1 --batch-size 128 --data_config ur5rmb_v2 --output_dir ckpt_debug --video-backend torchvision_av --max_steps 1000
#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task_pg --num-gpus 1 --batch-size 128 --data_config ur5rmb_v2 --output_dir ckpt_debug --max_steps 1000

# python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task --num-gpus 1 --batch-size 16 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_debug --video-backend torchvision_av --max_steps 110
# python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task --num-gpus 1 --batch-size 32 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_debug --video-backend torchvision_av --max_steps 110
# python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task --num-gpus 1 --batch-size 64 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_debug --video-backend torchvision_av --max_steps 110
# python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task --num-gpus 1 --batch-size 128 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_debug --video-backend torchvision_av --max_steps 110
# python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task --num-gpus 1 --batch-size 256 --data_config ur5rmb_v2 --output_dir ckpt_rmb_ur5_debug --video-backend torchvision_av --max_steps 110

#python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/MujocoUR5ePick_mtpick --num-gpus 1 --batch-size 128 --data_config ur5rmb_v2 --output_dir ckpt_mtpick --max_steps 60000 --video-backend torchvision_av

python scripts/gr00t_finetune.py --dataset-path /groups/gaf51379/physical-grounding/datasets/lerobot_dataset/koshimaki/RealUR5eDemo_sb_10task_pg_reason --num-gpus 1 --batch-size 32 --data_config ur5rmb_reasoning --output_dir ckpt_rmb_ur5_pg_reasoning_gradacc_4 --video-backend torchvision_av --max_steps 60000 --tune-llm
