import numpy as np, cv2

# --- 回転（roll→pitch→yaw, XYZ intrinsic）---
def rpy_to_R_XYZ_intrinsic(roll, pitch, yaw, degrees=True):
    if degrees:
        roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
    c, s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(roll),-s(roll)],[0,s(roll),c(roll)]], np.float32)
    Ry = np.array([[ c(pitch),0,s(pitch)],[0,1,0],[-s(pitch),0,c(pitch)]], np.float32)
    Rz = np.array([[c(yaw),-s(yaw),0],[s(yaw), c(yaw),0],[0,0,1]], np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def row6_to_T(row6, degrees=True):
    x, y, z, r, p, yw = [float(v) for v in row6]
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = rpy_to_R_XYZ_intrinsic(r, p, yw, degrees)
    T[:3, 3] = np.array([x, y, z], np.float32)
    return T

def idx_to_bgr(i, N):
    if N <= 1: return (0,0,255)
    h = int(180.0 * (i / float(N)))
    hsv = np.uint8([[[h,200,255]]]); bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0,0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def _axis_vec(axis):
    if isinstance(axis, str):
        s = axis.strip().lower()
        sign = -1.0 if s.startswith('-') else 1.0
        key  = s[1:] if s.startswith('-') else s
        base = {"x":np.array([1,0,0],np.float32),
                "y":np.array([0,1,0],np.float32),
                "z":np.array([0,0,1],np.float32)}[key]
        v = base * sign
    else:
        v = np.asarray(axis, np.float32); v /= (np.linalg.norm(v)+1e-9)
    return v

def draw_axis_arrow(image, K, R, t, *, axis="z", length=0.10, color=(0,255,0),
                    thickness=2, tip_px=6, dist=None, origin_obj=np.zeros(3, np.float32)):
    if isinstance(axis, str):
        s = axis.strip().lower()
        sign = -1.0 if s.startswith('-') else 1.0
        key  = s[1:] if s.startswith('-') else s
        base = {"x":np.array([1,0,0],np.float32),
                "y":np.array([0,1,0],np.float32),
                "z":np.array([0,0,1],np.float32)}[key]
        v = base * sign
    else:
        v = np.asarray(axis, np.float32); v /= (np.linalg.norm(v)+1e-9)

    p0 = origin_obj.astype(np.float32).reshape(1,3)
    p1 = (origin_obj + length * v).astype(np.float32).reshape(1,3)
    pts_obj = np.vstack([p0, p1])

    rvec,_ = cv2.Rodrigues(R.astype(np.float32))
    tvec   = t.astype(np.float32).reshape(3,1)
    pts2d,_= cv2.projectPoints(pts_obj, rvec, tvec, K.astype(np.float32), dist)
    p0_2d, p1_2d = pts2d.reshape(-1,2).astype(int)

    cv2.arrowedLine(image, tuple(p0_2d), tuple(p1_2d), color, thickness, tipLength=0.0)
    cv2.circle(image, tuple(p1_2d), tip_px, color, -1)
    return p1_2d

def overlay_pose_list_flexible(
    image_bgr, K,
    poses_w2o=None,          # (N,6) World→Object   [必須]
    poses_w2c=None,          # (N,6) World→Camera   [任意]
    poses_c2w=None,          # (N,6) Camera→World   [任意]
    T_c2w=None,              # (4x4) 固定カメラ     [任意]
    *,
    axis="z", length=0.10, thickness=2, tip_px=6,
    dist=None, origin_obj=np.zeros(3, np.float32),
    cull_back=True, margin_px=5, label=False, degrees=True
):
    assert poses_w2o is not None and len(poses_w2o.shape)==2 and poses_w2o.shape[1]==6, "poses_w2o (N,6) 必須"
    N = poses_w2o.shape[0]
    img = image_bgr
    H, W = img.shape[:2]

    # カメラ姿勢の供給源を決める（優先順：poses_w2c → poses_c2w → T_c2w）
    have_w2c = poses_w2c is not None
    have_c2w = poses_c2w is not None
    assert have_w2c or have_c2w or T_c2w is not None, "カメラ姿勢（poses_w2c / poses_c2w / T_c2w）のいずれかが必要"

    drawn, skipped = [], []
    endpoints = []

    for i in range(N):
        # Object: T_w2o(i) → T_o2w(i)
        T_w2o = row6_to_T(poses_w2o[i], degrees=degrees)
        T_o2w = np.linalg.inv(T_w2o)

        # Camera: そのフレームの T_w2c or T_c2w
        if have_w2c:
            T_w2c = row6_to_T(poses_w2c[i], degrees=degrees)
        elif have_c2w:
            T_c2w_i = row6_to_T(poses_c2w[i], degrees=degrees)
            T_w2c   = np.linalg.inv(T_c2w_i)
        else:
            T_w2c = np.linalg.inv(T_c2w).astype(np.float32)

        # Object→Camera
        T_o2c = (T_w2c @ T_o2w).astype(np.float32)
        R = T_o2c[:3,:3]; t = T_o2c[:3,3]

        # ざっくりカリング（前方＆画面内）
        # v_local = np.array([0,0,1], np.float32)
        # p_tip_o = origin_obj + length * v_local
        # rvec,_  = cv2.Rodrigues(R); tvec=t.reshape(3,1)
        # p2d,_   = cv2.projectPoints(p_tip_o.reshape(1,3), rvec, tvec, K, dist)
        # u, v = p2d.reshape(2)
        # z_front = (R[2,:] @ p_tip_o + t[2]) > 0
        # inside  = (-margin_px <= u < W+margin_px) and (-margin_px <= v < H+margin_px)

        v_local = _axis_vec(axis).astype(np.float32)
        p_tip_o = (origin_obj + length * v_local).astype(np.float32)
        rvec,_  = cv2.Rodrigues(R.astype(np.float32)); tvec=t.reshape(3,1)
        p2d,_   = cv2.projectPoints(p_tip_o.reshape(1,3), rvec, tvec, K.astype(np.float32), dist)
        u, v = p2d.reshape(2)
        z_front = (R @ p_tip_o + t)[2] > 0   # ← 先端点のZ
        inside  = (-margin_px <= u < W+margin_px) and (-margin_px <= v < H+margin_px)

        if cull_back and (not z_front or not inside):
            skipped.append(i); endpoints.append(None); continue

        color = idx_to_bgr(i, N)
        end = draw_axis_arrow(img, K, R, t, axis=axis, length=length, color=color,
                              thickness=thickness, tip_px=tip_px, dist=dist, origin_obj=origin_obj)
        endpoints.append(tuple(end)); drawn.append(i)

        if label:
            cv2.putText(img, f"{i}", tuple(end), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return {"image": img, "drawn_idx": drawn, "skipped_idx": skipped, "endpoints": endpoints}


import pandas as pd
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

def to_numpy_by_key_range(df, key: str, start, end, cols=None, closed: str = "left"):
    """
    closed: "left" -> [start, end) / "right" -> (start, end] / "both" -> [start, end] / "neither" -> (start, end)
    """
    # between は両端含むので、半開区間は条件で表現
    if closed == "both":
        mask = df[key].between(start, end, inclusive="both")
    elif closed == "neither":
        mask = df[key].between(start, end, inclusive="neither")
    elif closed == "right":  # (start, end]
        mask = (df[key] > start) & (df[key] <= end)
    else:  # "left": [start, end)
        mask = (df[key] >= start) & (df[key] < end)

    sub = df.loc[mask, cols]
    return sub.to_numpy()  # 必要なら np.ascontiguousarray(...)


# --- utils: RPY(XYZ intrinsic)とrow6→4x4（あなたの実装と同じ） ---
# def rpy_to_R_XYZ_intrinsic(roll, pitch, yaw, degrees=True):
#     if degrees:
#         roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
#     c, s = np.cos, np.sin
#     Rx = np.array([[1,0,0],[0,c(roll),-s(roll)],[0,s(roll),c(roll)]], np.float32)
#     Ry = np.array([[ c(pitch),0,s(pitch)],[0,1,0],[-s(pitch),0,c(pitch)]], np.float32)
#     Rz = np.array([[c(yaw),-s(yaw),0],[s(yaw), c(yaw),0],[0,0,1]], np.float32)
#     return (Rz @ Ry @ Rx).astype(np.float32)

# --- 1) (vx, vy, omega) を積分してワールド軌跡を作る ---
def integrate_traj_xytheta(
    v_seq,             # shape (N,3) → [vx, vy, omega] ; 単位: m/s, rad/s
    start_xyz,         # (x0, y0, z0) ワールド 初期位置（zは固定で使う）
    theta0=0.0,        # 初期方位（rad）。速度がワールド基準なら使われない
    freq=30.0,         # Hz
    vel_frame="world"  # "world" | "body"
):
    dt = 1.0 / float(freq)
    N  = v_seq.shape[0]
    xs = np.empty(N, np.float32)
    ys = np.empty(N, np.float32)
    thetas = np.empty(N, np.float32)

    x, y = float(start_xyz[0]), float(start_xyz[1])
    th   = float(theta0)

    for k in range(N):
        vx, vy, omega = [float(v) for v in v_seq[k]]
        if vel_frame == "body":
            # ボディ速度 → ワールドへ回転
            c, s = np.cos(th), np.sin(th)
            vx_w =  c*vx - s*vy
            vy_w =  s*vx + c*vy
        else:
            vx_w, vy_w = vx, vy  # そのまま（ワールド基準）

        x  += vx_w * dt
        y  += vy_w * dt
        th += omega * dt

        xs[k], ys[k], thetas[k] = x, y, th

    # zは固定（x0,y0の高さ維持）
    zs = np.full(N, float(start_xyz[2]), np.float32)
    P_w = np.stack([xs, ys, zs], axis=1)  # (N,3)
    return P_w, thetas  # thetasは必要なら使用

# --- 2) ワールド点列を各フレームの画像へ射影して描く ---
def overlay_world_traj_on_frame(
    image_bgr, K, dist,
    T_c2w_k,      # shape (6,) または (4x4)。このフレームの C→W
    pts_w,        # shape (M,3) ワールド軌跡点（0..現在まで）
    color=(0,200,255), thickness=2, skip_behind=True
):
    # C→W から W→C
    if T_c2w_k.shape == (6,):
        T_c2w = row6_to_T(T_c2w_k)
    else:
        T_c2w = T_c2w_k.astype(np.float32)
    T_w2c = np.linalg.inv(T_c2w).astype(np.float32)

    R = T_w2c[:3,:3].astype(np.float32)
    t = T_w2c[:3, 3].astype(np.float32).reshape(3,1)

    # 後方除外（任意）
    if skip_behind:
        # Zc = R @ Pw + t のZ > 0 のものだけ残す
        Zc = (pts_w @ R[2].reshape(3,1)).reshape(-1) + float(t[2])
        mask = Zc > 0
        pts_use = pts_w[mask]
    else:
        pts_use = pts_w

    if len(pts_use) < 2:
        return image_bgr  # 点が足りなければ描かない

    # 射影
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(pts_use.astype(np.float32), rvec, t, K.astype(np.float32), dist)
    uv = pts2d.reshape(-1,2).astype(int)

    # ポリライン描画
    cv2.polylines(image_bgr, [uv], isClosed=False, color=color, thickness=thickness)
    # 始点/終点を強調（任意）
    cv2.circle(image_bgr, tuple(uv[0]), 4, color, -1)
    cv2.circle(image_bgr, tuple(uv[-1]), 5, color, -1)
    return image_bgr


def integrate_traj_xytheta_world(v_seq, start_xyz, freq):
    """
    v_seq: (N,3) = [vx, vy, omega] in world frame (m/s, rad/s)
    start_xyz: (3,) world
    """
    dt = 1.0/float(freq)
    N  = int(v_seq.shape[0])
    xs = np.empty(N, np.float32); ys = np.empty(N, np.float32); zs = np.empty(N, np.float32)
    x,y,z = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])
    theta = 0.0
    for k in range(N):
        vx,vy,om = [float(x) for x in v_seq[k]]
        x += vx*dt*1000; y += vy*dt*1000; theta += om*dt
        xs[k]=x; ys[k]=y; zs[k]=z
    return np.stack([xs,ys,zs], axis=1)


# --- 1) (vx, vy, omega) を積分してワールド軌跡を作る ---
def integrate_traj_xytheta(
    v_seq,             # shape (N,3) → [vx, vy, omega] ; 単位: m/s, rad/s
    start_xyz,         # (x0, y0, z0) ワールド 初期位置（zは固定で使う）
    theta0=0.0,        # 初期方位（rad）。速度がワールド基準なら使われない
    freq=30.0,         # Hz
    vel_frame="world"  # "world" | "body"
):
    dt = 1.0 / float(freq)
    N  = v_seq.shape[0]
    xs = np.empty(N, np.float32)
    ys = np.empty(N, np.float32)
    thetas = np.empty(N, np.float32)

    x, y = float(start_xyz[0]), float(start_xyz[1])
    th   = float(theta0)

    for k in range(N):
        vx, vy, omega = [float(v) for v in v_seq[k]]
        if vel_frame == "body":
            # ボディ速度 → ワールドへ回転
            c, s = np.cos(th), np.sin(th)
            vx_w =  c*vx - s*vy
            vy_w =  s*vx + c*vy
        else:
            vx_w, vy_w = vx, vy  # そのまま（ワールド基準）

        x  += vx_w * dt
        y  += vy_w * dt
        th += omega * dt

        xs[k], ys[k], thetas[k] = x, y, th

    # zは固定（x0,y0の高さ維持）
    zs = np.full(N, float(start_xyz[2]), np.float32)
    P_w = np.stack([xs, ys, zs], axis=1)  # (N,3)
    return P_w, thetas  # thetasは必要なら使用

# --- 2) ワールド点列を各フレームの画像へ射影して描く ---
def overlay_world_traj_on_frame(
    image_bgr, K, dist,
    T_c2w_k,      # shape (6,) または (4x4)。このフレームの C→W
    pts_w,        # shape (M,3) ワールド軌跡点（0..現在まで）
    color=(0,200,255), thickness=2, skip_behind=True
):
    # C→W から W→C
    if T_c2w_k.shape == (6,):
        T_c2w = row6_to_T(T_c2w_k)
    else:
        T_c2w = T_c2w_k.astype(np.float32)
    T_w2c = np.linalg.inv(T_c2w).astype(np.float32)

    R = T_w2c[:3,:3].astype(np.float32)
    t = T_w2c[:3, 3].astype(np.float32).reshape(3,1)

    # 後方除外（任意）
    if skip_behind:
        # Zc = R @ Pw + t のZ > 0 のものだけ残す
        Zc = (pts_w @ R[2].reshape(3,1)).reshape(-1) + float(t[2])
        mask = Zc > 0
        pts_use = pts_w[mask]
    else:
        pts_use = pts_w

    if len(pts_use) < 2:
        return image_bgr  # 点が足りなければ描かない

    # 射影
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(pts_use.astype(np.float32), rvec, t, K.astype(np.float32), dist)
    uv = pts2d.reshape(-1,2).astype(int)

    # ポリライン描画
    cv2.polylines(image_bgr, [uv], isClosed=False, color=color, thickness=thickness)
    # 始点/終点を強調（任意）
    cv2.circle(image_bgr, tuple(uv[0]), 4, color, -1)
    cv2.circle(image_bgr, tuple(uv[-1]), 5, color, -1)
    return image_bgr

def row6_to_T_w2c(row6, degrees=True):
    # あなたの row6_to_T は C2W でも W2C でも使える一般関数
    # ここでは「この row6 は W→C だ」と明示するだけ（数式は同じ）
    x,y,z,r,p,yw = [float(v) for v in row6]
    # RPY(XYZ intrinsic)
    if degrees:
        r,p,yw = np.deg2rad([r,p,yw])
    c,s = np.cos, np.sin
    Rx = np.array([[1,0,0],[0,c(r),-s(r)],[0,s(r),c(r)]], np.float32)
    Ry = np.array([[ c(p),0,s(p)],[0,1,0],[-s(p),0,c(p)]], np.float32)
    Rz = np.array([[c(yw),-s(yw),0],[s(yw),c(yw),0],[0,0,1]], np.float32)
    R = (Rz @ Ry @ Rx).astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = np.array([x,y,z], np.float32)
    return T

def overlay_traj_world_with_w2c(
    image_bgr, K, dist, T_w2c_k, pts_w, *,
    color=(0,200,255), thickness=2, draw_points=True,
    clip_behind=False, clip_outside=False, margin=10
):
    """
    T_w2c_k : (6,) または (4,4) の W→C
    pts_w   : (M,3) 世界座標の軌跡点（0..kまで）
    """
    # 1) 取り出し
    if isinstance(T_w2c_k, np.ndarray) and T_w2c_k.shape == (4,4):
        T_w2c = T_w2c_k.astype(np.float32)
    else:
        T_w2c = row6_to_T_w2c(T_w2c_k).astype(np.float32)

    R = T_w2c[:3,:3].astype(np.float32)
    t = T_w2c[:3, 3].astype(np.float32).reshape(3,1)

    if pts_w.size == 0:
        return image_bgr

    # 2) 後方/画面外クリップ（切り分けのためデフォルトOFF）
    Pw = pts_w.astype(np.float32)  # (M,3)
    Zc = Pw @ R[2].reshape(3,1) + float(t[2])  # (M,1)
    mask = np.ones((Pw.shape[0],), dtype=bool)
    if clip_behind:
        mask &= (Zc.reshape(-1) > 0)

    # 射影
    rvec,_ = cv2.Rodrigues(R)
    pts2d,_= cv2.projectPoints(Pw[mask], rvec, t, K.astype(np.float32), dist)
    uv = pts2d.reshape(-1,2).astype(int)

    # 3) 画面外マスク（任意）
    H,W = image_bgr.shape[:2]
    if clip_outside:
        inside = (uv[:,0] >= -margin) & (uv[:,0] < W+margin) & (uv[:,1] >= -margin) & (uv[:,1] < H+margin)
        uv = uv[inside]

    # 4) 描画（点＋ポリライン）
    if uv.shape[0] >= 2:
        cv2.polylines(image_bgr, [uv], isClosed=False, color=color, thickness=thickness)
    if draw_points and uv.shape[0] >= 1:
        for p in uv:
            cv2.circle(image_bgr, tuple(p), 2, color, -1)
        cv2.circle(image_bgr, tuple(uv[0]), 5, (0,0,255), -1)    # 始点赤
        cv2.circle(image_bgr, tuple(uv[-1]), 5, (0,255,0), -1)   # 終点緑

    # デバッグ：何点投影できたか左上に表示
    cv2.putText(image_bgr, f"proj:{uv.shape[0]}", (8,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return image_bgr





key   = "index"          # 区間キー列（数値/日時どれでもOK）
start, end = 0, 400 # 抜きたい行の区間
cols_head  = "observation.head_pose.absolute"  # 取り出す列（Noneなら全部）
cols_hand  = "observation.end_effector_pose.absolute"  # 取り出す列（Noneなら全部）

mask_head = (data[key] >= start) & (data[key] < end)
poses_w2c = np.stack(data.loc[mask_head, cols_head].to_numpy(), axis=0).astype(np.float32)

mask_hand = (data[key] >= start) & (data[key] < end)
poses_w2o = np.stack(data.loc[mask_hand, cols_hand].to_numpy(), axis=0).astype(np.float32)



img = cv2.cvtColor(frames[150], cv2.COLOR_RGB2BGR)
res = overlay_pose_list_flexible(
    img, K,
    poses_w2o=poses_w2o,     # (N,6)
    poses_w2c=poses_w2c,     # (N,6) ← World→Camera があるならこれだけでOK
    axis="y", length=0.10, label=True,
    cull_back=False,         # まず False で出ること確認
    dist=None
)

cv2.imwrite("overlay_fixed_cam.png", res["image"])







cols_base = "action.base"
mask_base = (data[key] >= start) & (data[key] < end)
vel_world = np.stack(data.loc[mask_base, cols_base].to_numpy(), axis=0).astype(np.float32)

ee0 = data[cols_hand][0]  # (x,y,z, r,p,yaw)
start_xyz = np.array(ee0[:3], dtype=np.float32)
freq = 30
# 1) 軌跡点列（ワールド）を生成
P_w, _ = integrate_traj_xytheta(
    v_seq=vel_world.astype(np.float32),   # (N,3)
    start_xyz=start_xyz,                  # (x0,y0,z0)
    theta0=0.0,                           # 不要なら0
    freq=freq,
    vel_frame="world"                     # <- ここが今回の前提
)  # P_w.shape=(N,3)

# 2) 例：フレーム k 上に「そこまでの軌跡」を重畳
img = cv2.cvtColor(frames[start], cv2.COLOR_RGB2BGR)


print(P_w[start:end])

# poses_c2w は (N,6) の C→W。k番目を渡す
img = overlay_world_traj_on_frame(
    img, K, dist=None,
    T_c2w_k=data["observation.head_pose.absolute"][start],  # C→W
    pts_w=P_w[start:end],                                    # 0..k の軌跡
    color=(0,200,255), thickness=2, skip_behind=True
)

cv2.imwrite("traj_k{:03d}.png".format(start), img)

N = end - start
#t = np.arange(N)/freq
#v_seq = np.c_[ 0.02*np.cos(0.5*t), 0.02*np.sin(0.5*t), 0.1*np.ones_like(t) ].astype(np.float32)  # (vx,vy,ω)

P_w = integrate_traj_xytheta_world(vel_world, start_xyz, freq)   # (N,3)

k = start
img = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR)

# W→C（headの absolute が本当に W→C であることが前提）
T_w2c_k = data["observation.head_pose.absolute"][k]

# 射影／描画（カリングOFF）
out = overlay_traj_world_with_w2c(
    img, K, dist=None,
    T_w2c_k=T_w2c_k,
    pts_w=P_w[:k+1],
    color=(0,200,255), thickness=2,
    draw_points=True,
    clip_behind=False, clip_outside=False
)
cv2.imwrite("traj_debug.png", out)


# Pw_world, _ = integrate_traj_xytheta(
#     v_seq=v_seq,              # (N,3) = [vx,vy,omega]
#     start_xyz=start_xyz,      # 最初のエンドエフェクタ位置 (x0,y0,z0) in world
#     theta0=0.0,               # 未使用（world解釈）
#     freq=freq,
#     vel_frame="world"
# )

# # 2) ボディ解釈（Yawで回してから積分）
# # 初期方位は「ロボット／手先のYaw」を使う（データがdegならradに）
# theta0 = np.deg2rad(float(ee0[5]))  # 例：手先yawを使う（適宜ベースYawに差し替え）
# Pw_body, _ = integrate_traj_xytheta(
#     v_seq=v_seq,
#     start_xyz=start_xyz,
#     theta0=theta0,
#     freq=freq,
#     vel_frame="body"          # ← ここがポイント
# )

# # 3) 同じフレーム k 上に両方描く（W→Cしか分からない前提）
# k = min(len(frames)-1, len(v_seq)-1)
# img = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR)
# Tw2c_k = data["observation.head_pose.absolute"][k]   # W→C（ここはあなたの列に合わせる）

# img = overlay_traj_world_with_w2c(img, K, None, Tw2c_k, Pw_world[:k+1],
#                                   color=(0,255,255), thickness=2, draw_points=True,
#                                   clip_behind=False, clip_outside=False)
# img = overlay_traj_world_with_w2c(img, K, None, Tw2c_k, Pw_body[:k+1],
#                                   color=(255,0,255), thickness=2, draw_points=False,
#                                   clip_behind=False, clip_outside=False)
# cv2.putText(img, "world=cyan, body=magenta", (8,22), 0, 0.6, (255,255,255), 2, cv2.LINE_AA)
# cv2.imwrite("traj_compare.png", img)