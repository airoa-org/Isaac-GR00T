import subprocess
import glob
import sys
import os

lerobot_path = "/home/deepstation/lerobot_dataset/realur5demoenv1_1.0.0_lerobot"
files = glob.glob(f"{lerobot_path}/videos/*/*/*.mp4")

for file in files:
    newfile = os.path.dirname(file)+"/"+os.path.basename(file)[:-4]+"_h264.mp4"
    cmd = f"ffmpeg -i {file} -c:v libx264 -crf 23 -preset fast {newfile}"
    print("Running command: ", cmd)
    subprocess.run(cmd, shell=True)