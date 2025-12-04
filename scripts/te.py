import pandas as pd
import numpy as np
import cv2

data = pd.read_parquet("/groups/gag51454/workspace_oh/hsr/2025-06-v3.0-success-only/data/chunk-000/episode_000000.parquet")

print(data["observation.head_pose.absolute"][0])
print(data["observation.end_effector_pose.absolute"][0])

import torchvision
torchvision.set_video_backend("pyav")
reader = torchvision.io.VideoReader("/groups/gag51454/workspace_oh/hsr/2025-06-v3.0-success-only/videos/chunk-000/observation.image.head/episode_000000.mp4", "video")
frames = []
for frame in reader:
     frames.append(frame["data"].numpy())
frames = np.array(frames)
frames = frames.transpose(0, 2, 3, 1)
print(frames[0])