#!/bin/bash

#SBATCH --partition part-group_25b505
#SBATCH --gpus 1 
#SBATCH --time 3:00:00
echo "[INFO]: Modules loaded"

module load awscli-v2
export AWS_ACCESS_KEY_ID=2DMPFGDM78P24CGSU7QE
export AWS_SECRET_ACCESS_KEY=zXzu0yeAupmbLwuDbhlkjS12MN8PiIdZRnLCyAyz

aws s3 cp /home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T/ckpt-0925 s3://airoa-fm-development-competition/group6/gr00t-0925 --endpoint-url=https://s3.ap-northeast-1.wasabisys.com --recursive

