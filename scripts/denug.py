import numpy as np, cv2, itertools, os

import pandas as pd
import numpy as np
import cv2

data = pd.read_parquet("/groups/gag51454/workspace_oh/hsr/2025-06-v3.0-success-only/data/chunk-000/episode_000000.parquet")

import torchvision
torchvision.set_video_backend("pyav")
reader = torchvision.io.VideoReader("/groups/gag51454/workspace_oh/hsr/2025-06-v3.0-success-only/videos/chunk-000/observation.image.head/episode_000000.mp4", "video")
frames = []
for frame in reader:
     frames.append(frame["data"].numpy())
frames = np.array(frames)
frames = frames.transpose(0, 2, 3, 1)

K = np.array([
    [607.3814086914062,   0.0,                 315.9123840332031],
    [  0.0,               607.2514038085938,   233.77308654785156],
    [  0.0,                 0.0,                 1.0]
], dtype=np.float32)

fx, fy = K[0,0], K[1,1]
cx, cy = K[0,2], K[1,2]
print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")

dist = None  # 歪みがあるなら np.array([...], np.float32)

w2o = data["observation.end_effector_pose.absolute"][0]
w2c = data["observation.head_pose.absolute"][0]


# ---- fill these with YOUR absolutes (deg, meters) ----
# object absolute
xo, yo, zo = w2o[0], w2o[1], w2o[2]
ro, po, yo_ = w2o[3], w2o[4], w2o[5]
# camera absolute
xc, yc, zc = w2c[0], w2c[1], w2c[2]
rc, pc, yc_ = w2c[3], w2c[4], w2c[5]

img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)

def Rx(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],np.float32)
def Ry(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],np.float32)
def Rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32)

def rpy_to_R_XYZ_intrinsic(r,p,y,deg=True):
    if deg: r,p,y = np.deg2rad([r,p,y])
    # intrinsic XYZ == R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return (Rz(y) @ Ry(p) @ Rx(r)).astype(np.float32)

def rpy_to_R_XYZ_extrinsic(r,p,y,deg=True):
    if deg: r,p,y = np.deg2rad([r,p,y])
    # extrinsic XYZ == R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
    return (Rx(r) @ Ry(p) @ Rz(y)).astype(np.float32)

def T_from_pose(r,p,y, x,y_,z, mode="intrinsic"):
    R = rpy_to_R_XYZ_intrinsic(r,p,y) if mode=="intrinsic" else rpy_to_R_XYZ_extrinsic(r,p,y)
    T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=np.array([x,y_,z],np.float32); return T

def make_o2c(
    obj_pose_is_o2w, cam_pose_is_c2w,
    rot_mode, flipY=False
):
    # build object→world OR world→object
    if obj_pose_is_o2w:
        T_o2w = T_from_pose(ro,po,yo_, xo,yo,zo, mode=rot_mode)
    else:
        T_w2o = T_from_pose(ro,po,yo_, xo,yo,zo, mode=rot_mode)
        T_o2w = np.linalg.inv(T_w2o)

    # build camera→world OR world→camera
    if cam_pose_is_c2w:
        T_c2w = T_from_pose(rc,pc,yc_, xc,yc,zc, mode=rot_mode)
        T_w2c = np.linalg.inv(T_c2w)
    else:
        T_w2c = T_from_pose(rc,pc,yc_, xc,yc,zc, mode=rot_mode)

    # optional axis flip (world Y-up → camera Y-down等の粗合わせ用)
    S = np.diag([1,-1,1,1]).astype(np.float32) if flipY else np.eye(4, dtype=np.float32)

    T_o2c = T_w2c @ (S @ T_o2w)   # flipはworld側に掛ける近道（必要なら細調整を）
    return T_o2c

def draw_case(tag, T_o2c):
    R = T_o2c[:3,:3]; t = T_o2c[:3,3]
    # 投影点：原点と+Z矢印
    pts = np.array([[0,0,0],[0,0,0.12]], np.float32)  # 矢印長さ0.12(単位はtと同じ)
    rvec,_ = cv2.Rodrigues(R.astype(np.float32)); tvec=t.reshape(3,1).astype(np.float32)
    pts2d,_ = cv2.projectPoints(pts, rvec, tvec, K, dist)
    p0, p1 = pts2d.reshape(-1,2)
    # 有効性チェック
    z_front = (R[2,:] @ pts[1] + t[2]) > 0  # ざっくり
    H,W = img.shape[:2]
    inside = (0<=p1[0]<W and 0<=p1[1]<H)
    if not (z_front and inside): return False

    vis = img.copy()
    cv2.arrowedLine(vis, tuple(p0.astype(int)), tuple(p1.astype(int)), (0,0,255), 2, tipLength=0.0)
    cv2.circle(vis, tuple(p1.astype(int)), 6, (0,0,255), -1)
    cv2.circle(vis, tuple(p0.astype(int)), 4, (0,255,0), -1)
    cv2.imwrite(f"debug_{tag}.png", vis)
    return True

cases = []
for obj_o2w, cam_c2w, rot_mode, flipY in itertools.product(
    [True, False], [True, False], ["intrinsic","extrinsic"], [False, True]
):
    tag = f"obj{'o2w' if obj_o2w else 'w2o'}__cam{'c2w' if cam_c2w else 'w2c'}__{rot_mode}__flipY{int(flipY)}"
    T_o2c = make_o2c(obj_o2w, cam_c2w, rot_mode, flipY)
    ok = draw_case(tag, T_o2c)
    print(tag, "->", "OK" if ok else "NG")
