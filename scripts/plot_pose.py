import numpy as np, cv2

# --- 基本: RPY(XYZ) -> 回転行列 ---
def rpy_to_R_XYZ_intrinsic(roll, pitch, yaw, degrees=True):
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    c, s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(roll),-s(roll)],[0,s(roll),c(roll)]], np.float32)
    Ry = np.array([[c(pitch),0,s(pitch)],[0,1,0],[-s(pitch),0,c(pitch)]], np.float32)
    Rz = np.array([[c(yaw),-s(yaw),0],[s(yaw),c(yaw),0],[0,0,1]], np.float32)
    # Intrinsic XYZ：R = Rz @ Ry @ Rx（ローカル軸順）
    return (Rz @ Ry @ Rx).astype(np.float32)

def T_from_pose_w2o(x, y, z, roll, pitch, yaw, degrees=True):
    """ワールド→オブジェクト (T_w2o) を作る"""
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = rpy_to_R_XYZ_intrinsic(roll, pitch, yaw, degrees)
    T[:3, 3] = np.array([x, y, z], np.float32)
    return T

def T_from_pose_c2w(x, y, z, roll, pitch, yaw, degrees=True):
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = rpy_to_R_XYZ_intrinsic(roll, pitch, yaw, degrees)
    T[:3, 3] = np.array([x, y, z], np.float32)
    return T  # Camera→World


def Rt_object_to_camera_from_w2o_c2w(T_w2o, T_c2w):
    """T_w2o, T_c2w -> R,t (Object→Camera)"""
    T_o2w = np.linalg.inv(T_w2o)   # 物体: O←W の逆で O→W
    T_w2c = np.linalg.inv(T_c2w)   # カメラ: W←C の逆で W→C
    T_o2c = T_w2c @ T_o2w
    R = T_o2c[:3,:3].astype(np.float32)
    t = T_o2c[:3, 3].astype(np.float32)
    return R, t

def draw_axis_arrow(
    image, K, R, t,
    axis="z",              # 'x','y','z','-x','-y','-z' または 3Dベクトル(np.array)
    length=0.10,
    color=(0,255,0),
    thickness=2,
    tip_px=8,
    dist=None,
    both=False,
    origin_obj=np.zeros(3, np.float32),
    direction=+1.0,        # +1 でそのまま / -1 で反転
):
    """R,t は Object→Camera。axis は物体座標系での方向。"""
    # --- 軸ベクトル決定（文字列・ベクトル両対応） ---
    if isinstance(axis, str):
        s = axis.strip().lower()
        sign = -1.0 if s.startswith('-') else 1.0
        key  = s[1:] if s.startswith('-') else s
        base = {"x":np.array([1,0,0],np.float32),
                "y":np.array([0,1,0],np.float32),
                "z":np.array([0,0,1],np.float32)}[key]
        v = base * sign
    else:
        v = np.asarray(axis, np.float32)
        n = np.linalg.norm(v) + 1e-9
        v = v / n

    # 追加のdirectionで反転/拡張
    v = v * float(np.sign(direction) if direction != 0 else 1.0)

    # --- 物体座標で点を用意 ---
    p0 = origin_obj.astype(np.float32).reshape(1,3)
    p1 = (origin_obj + length * v).astype(np.float32).reshape(1,3)
    pts_obj = np.vstack([p0, p1])
    if both:
        p2 = (origin_obj - length * v).astype(np.float32).reshape(1,3)
        pts_obj = np.vstack([pts_obj, p2])

    # --- 投影 ---
    rvec,_ = cv2.Rodrigues(R.astype(np.float32))
    tvec   = t.astype(np.float32).reshape(3,1)
    pts2d,_= cv2.projectPoints(pts_obj, rvec, tvec, K.astype(np.float32), dist)
    pts2d  = pts2d.reshape(-1,2).astype(int)

    # --- 描画 ---
    o  = tuple(pts2d[0]); e1 = tuple(pts2d[1])
    cv2.arrowedLine(image, o, e1, color, thickness, tipLength=0.0)
    cv2.circle(image, e1, tip_px, color, -1)
    if both:
        e2 = tuple(pts2d[2])
        cv2.arrowedLine(image, o, e2, color, thickness, tipLength=0.0)
        cv2.circle(image, e2, tip_px, color, -1)
    return image

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

K = np.array([
    [607.3814086914062,   0.0,                 315.9123840332031],
    [  0.0,               607.2514038085938,   233.77308654785156],
    [  0.0,                 0.0,                 1.0]
], dtype=np.float32)

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")


# ===== 使い方例 =====
# 1) あなたの Absolute（World基準）から行列を作る
#    物体: world_T_object（= Object→World）
w2o = data["observation.end_effector_pose.absolute"][0]
T_w2o = T_from_pose_w2o(w2o[0], w2o[1], w2o[2], w2o[3], w2o[4], w2o[5], degrees=True)
#    カメラ: world_T_camera（= Camera→World）
w2c = data["observation.head_pose.absolute"][0]
T_c2w = T_from_pose_c2w(w2c[0], w2c[1], w2c[2], w2c[3], w2c[4], w2c[5], degrees=True)

# 2) Object→Camera の R,t を計算
R, t = Rt_object_to_camera_from_w2o_c2w(T_w2o=T_w2o, T_c2w=T_c2w)

# 3) 軸矢印を重畳（例：物体+Z方向を片側表示）
# K は 3x3 内パラ, dist は歪み（なければ None）
img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
img = draw_axis_arrow(img, K, R, t, axis="y", length=0.12, color=(0,0,255), thickness=2, dist=None, direction=-1)
cv2.imwrite("overlay_arrow.png", img)


