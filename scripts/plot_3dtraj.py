import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl

# ---- 視線軸(±x/±y/±z)に対する平面基底を作る ----
def plane_basis(view_axis='z', sign=+1):
    """
    view_axis: 'x'|'y'|'z'（視線方向 = その軸）
    sign     : +1 or -1（+は+軸方向を見る、-は-軸方向を見る）
    戻り: n,u,v （すべて (3,) の単位ベクトル）
          n: 視線方向の単位ベクトル
          u,v: 画像平面上の2軸（右手系を保つ並び）
    """
    view_axis = view_axis.lower()
    s = 1.0 if sign >= 0 else -1.0
    if view_axis == 'x':
        n = np.array([s,0,0], np.float32)  # 視線
        u = np.array([0,1,0], np.float32)  # 画面の+uをYに
        v = np.array([0,0,1], np.float32)  # 画面の+vをZに
    elif view_axis == 'y':
        n = np.array([0,s,0], np.float32)
        u = np.array([0,0,1], np.float32)  # +u = Z
        v = np.array([1,0,0], np.float32)  # +v = X
    else:  # 'z'
        n = np.array([0,0,s], np.float32)
        u = np.array([1,0,0], np.float32)  # +u = X
        v = np.array([0,1,0], np.float32)  # +v = Y
    # 念のため正規直交化（ここでは既に直交・単位）
    return n/np.linalg.norm(n), u/np.linalg.norm(u), v/np.linalg.norm(v)

# ---- 3D点群を指定平面へ射影し、深度(=視線方向成分)を得る ----
def project_to_plane(P_w, view_axis='z', sign=+1):
    """
    P_w : (N,3) world座標の点列
    戻り: U : (N,2) 平面座標 [u,v]
          D : (N,)   深度（視線nに沿った符号付き距離）→ ヒートマップ用
    """
    n,u,v = plane_basis(view_axis, sign)
    U = np.stack([P_w @ u, P_w @ v], axis=1).astype(np.float32)
    D = (P_w @ n).astype(np.float32)
    return U, D

# ---- 線分ごとに色を変える（深度で着色された軌跡）----
def plot_colored_trajectory(ax, U, D, cmap='viridis', lw=2.0, label=None):
    """
    U: (N,2), D:(N,), 連続軌跡を深度カラーマップで描く
    """
    if len(U) < 2:
        return
    # セグメント化
    segs = np.stack([U[:-1], U[1:]], axis=1)  # (N-1, 2, 2)
    # セグメントの色は両端の平均深度に
    Dc  = 0.5*(D[:-1] + D[1:])
    lc = LineCollection(segs, cmap=cmap, norm=None, linewidths=lw)
    lc.set_array(Dc)
    ax.add_collection(lc)
    # 始点/終点
    ax.scatter(U[0,0], U[0,1], c='k', s=20, zorder=3)
    ax.scatter(U[-1,0],U[-1,1], c='w', edgecolors='k', s=30, zorder=3)
    if label:
        ax.plot([],[], color='none', label=label)  # 凡例用ダミー

# ---- 俯瞰（平面）への重畳描画メイン ----
def plot_topdown_with_depth(P_list, labels=None, view_axis='z', sign=+1,
                            cmap='viridis', equal=True, show_colorbar=True,
                            title=None):
    """
    P_list: [P1, P2, ...] 各 (N,3)
    labels: 各軌跡のラベル（凡例用）
    """
    if labels is None:
        labels = [None]*len(P_list)

    fig, ax = plt.subplots(figsize=(6,6))
    # 各軌跡を投影＆描画
    mappable_for_cb = None
    for P, lab in zip(P_list, labels):
        U, D = project_to_plane(P, view_axis=view_axis, sign=sign)
        plot_colored_trajectory(ax, U, D, cmap=cmap, lw=2.0, label=lab)
        # 最後に追加した LineCollection をカラーバーに使う
        mappable_for_cb = ax.collections[-1]

    # 軸ラベル
    _, u, v = plane_basis(view_axis, sign)
    def axis_name(vec):
        # 近いワールド軸名を表示（見やすさ用）
        names = ['X','Y','Z']; axes = np.eye(3, dtype=np.float32)
        idx = int(np.argmax(np.abs(vec @ axes.T)))
        sgn = '+' if vec[idx] >= 0 else '-'
        return f"{sgn}{names[idx]}"
    ax.set_xlabel(f"u-axis ~ {axis_name(u)}")
    ax.set_ylabel(f"v-axis ~ {axis_name(v)}")
    if title is None:
        title = f"Top-down along {('+' if sign>=0 else '-')}{view_axis.upper()} (color = depth)"
    ax.set_title(title)
    ax.grid(True)

    # スケール揃え（等尺）
    if equal:
        # 2DなのでX,Y範囲を合わせる
        allU = []
        for P in P_list:
            U,_ = project_to_plane(P, view_axis=view_axis, sign=sign)
            allU.append(U)
        Ucat = np.vstack(allU)
        min2 = Ucat.min(axis=0); max2 = Ucat.max(axis=0)
        ctr  = 0.5*(min2+max2)
        half = 0.5*float((max2-min2).max()) * 1.05
        ax.set_xlim(ctr[0]-half, ctr[0]+half)
        ax.set_ylim(ctr[1]-half, ctr[1]+half)
        ax.set_aspect('equal', adjustable='box')

    # カラーバー（深度）
    if show_colorbar and mappable_for_cb is not None:
        cb = plt.colorbar(mappable_for_cb, ax=ax)
        cb.set_label("depth along view axis (m)")

    # 原点（Base0/World原点）を平面に投影して目印（常に (0,0)）
    ax.scatter(0,0, c='red', s=30, marker='+', zorder=4)
    ax.text(0,0, "  origin(Base0)", color='red')

    if any([lab for lab in labels]):
        ax.legend()
    plt.tight_layout()
    plt.savefig("traj_3d_projected.png")


# ---- 回転/行列ユーティリティ（あなたの定義と整合） ----
def rpy_to_R_XYZ_intrinsic(roll, pitch, yaw, degrees=True):
    if degrees: roll, pitch, yaw = np.deg2rad([roll, pitch, yaw])
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

# ---- 1) body系速度 (vx,vy,ω) を積分して B→W を作る ----
def integrate_base_body_vel(v_seq, freq=30.0, theta0=0.0, omega_unit="rad"):
    """
    v_seq: (N,3) = [vx, vy, omega] in BASE frame (m/s, m/s, {rad|deg}/s)
    返り値:
      t_bw: (N,3)  ベース原点のワールド座標（x,y,z=0）
      R_bw: (N,3,3) ベース→ワールドの回転（Yawのみ反映, X/Y軸はworldと一致想定）
      yaw:  (N,)   ワールドに対するベースのyaw
    """
    dt = 1.0/float(freq)
    N  = int(v_seq.shape[0])
    t_bw = np.zeros((N,3), np.float32)   # world座標中のベース原点
    R_bw = np.zeros((N,3,3), np.float32)
    yaw  = np.zeros((N,), np.float32)

    x=y=0.0
    th  = float(theta0)
    for k in range(N):
        vx, vy, om = map(float, v_seq[k])
        if omega_unit == "deg": om = np.deg2rad(om)
        # body→world 速度変換
        c, s = np.cos(th), np.sin(th)
        vx_w =  c*vx - s*vy
        vy_w =  s*vx + c*vy

        x  += vx_w * dt
        y  += vy_w * dt
        th += om   * dt

        t_bw[k] = (x, y, 0.0)
        R_bw[k] = rpy_to_R_XYZ_intrinsic(0.0, 0.0, np.rad2deg(th), degrees=True)  # yawのみ反映
        yaw[k]  = th
    return t_bw, R_bw, yaw

# ---- 2) B→EEF (各時刻) をワールド座標へ位置変換 ----
def eef_world_positions_from_base(ee_b6, t_bw, R_bw, degrees=True):
    """
    ee_b6: (N,6) 各時刻の B→EEF（ベース基準のEEF姿勢）
    t_bw:  (N,3) ベース原点のワールド座標
    R_bw:  (N,3,3) ベース→ワールドの回転
    返り値:
      P_e_w: (N,3) ワールド座標でのEEF原点位置
    """
    N = ee_b6.shape[0]
    P_e_w = np.empty((N,3), np.float32)
    for k in range(N):
        T_b2e = row6_to_T(ee_b6[k], degrees=degrees)
        t_be  = T_b2e[:3, 3]                # EEF原点のベース内位置
        P_e_w[k] = R_bw[k] @ t_be + t_bw[k] # worldへ
    return P_e_w

def set_axes_equal_3d(ax, *point_arrays, pad_ratio=0.05,
                      include_origin=True,  # 原点(0,0,0)を必ず含める
                      origin_center=False   # 原点を立方体の中心にする
                      ):
    """
    3D軸のスケールを揃える（等尺）。
    - include_origin=True: 原点もバウンディングに入れる
    - origin_center=True : 原点を中心に立方体を作り、全点を内包する半径にする
    """
    import numpy as np
    pts = []
    for p in point_arrays:
        if p is not None and len(p) > 0:
            pts.append(p.reshape(-1,3))
    if not pts:
        pts = [np.zeros((1,3), dtype=np.float32)]
    P = np.vstack(pts)

    if include_origin:
        P = np.vstack([P, np.zeros((1,3), dtype=np.float32)])

    if origin_center:
        # 原点を中心に、全点を内包する半径 r を決める
        r = float(np.abs(P).max())  # 各軸の最大絶対値
        r *= (1.0 + pad_ratio)
        ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    else:
        xyz_min = P.min(axis=0); xyz_max = P.max(axis=0)
        center  = (xyz_min + xyz_max) / 2.0
        max_range = float((xyz_max - xyz_min).max())
        half = max_range / 2.0 * (1.0 + pad_ratio)
        ax.set_xlim(center[0]-half, center[0]+half)
        ax.set_ylim(center[1]-half, center[1]+half)
        ax.set_zlim(center[2]-half, center[2]+half)

    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass




def draw_frame_triad(ax, origin, R=np.eye(3, dtype=np.float32), length=0.15, lw=2.0, label="Base0 / World"):
    """
    origin: (3,)  原点
    R     : (3,3) そのフレームの回転（列ベクトルが x,y,z の向き）
    length: 軸の長さ（m）
    """
    o = np.asarray(origin, np.float32).reshape(3)
    x = o + R[:,0] * length
    y = o + R[:,1] * length
    z = o + R[:,2] * length

    # 軸（X=red, Y=green, Z=blue）
    ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], color='r', lw=lw)
    ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], color='g', lw=lw)
    ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], color='b', lw=lw)

    # 矢印先に少しだけマーカー（見やすさ）
    ax.scatter([x[0]],[x[1]],[x[2]], c='r', s=16)
    ax.scatter([y[0]],[y[1]],[y[2]], c='g', s=16)
    ax.scatter([z[0]],[z[1]],[z[2]], c='b', s=16)

    # 原点マーク＆ラベル
    ax.scatter([o[0]],[o[1]],[o[2]], c='k', s=30, marker='o')
    ax.text(o[0], o[1], o[2], f"  {label}", fontsize=10, color='k')



# ---- 3) 可視化 ----
def plot_trajs_3d(P_base_w=None, P_eef_w=None, title="Trajectories (world=first base frame)"):
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    # Base
    if P_base_w is not None:
        ax.plot(P_base_w[:,0], P_base_w[:,1], P_base_w[:,2], color='tab:gray', lw=2, label='Base (origin)')
        ax.scatter(P_base_w[0,0], P_base_w[0,1], P_base_w[0,2], c='k', s=40, marker='o')
        ax.scatter(P_base_w[-1,0],P_base_w[-1,1],P_base_w[-1,2], c='k', s=50, marker='^')
    # EEF
    ax.plot(P_eef_w[:,0], P_eef_w[:,1], P_eef_w[:,2], color='tab:blue', lw=2, label='EEF')
    ax.scatter(P_eef_w[0,0], P_eef_w[0,1], P_eef_w[0,2], c='b', s=40, marker='o')
    ax.scatter(P_eef_w[-1,0],P_eef_w[-1,1],P_eef_w[-1,2], c='b', s=50, marker='^')

    ax.legend(); ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(title); ax.set_box_aspect([1,1,1])
    #set_axes_equal_3d(ax, P_base_w, P_eef_w, pad_ratio=0.05)
    set_axes_equal_3d(ax, P_base_w, P_eef_w, pad_ratio=0.05,
                  include_origin=True, origin_center=True)
    draw_frame_triad(ax, origin=np.zeros(3, dtype=np.float32), R=np.eye(3, dtype=np.float32),
                     length=0.15, lw=2.0, label="Base0 / World")
    # 目で見やすいようgrid/等尺など
    ax.grid(True)
    plt.savefig("traj_3d.png")

def eef_world_positions_from_E2B(ee_e2b, t_bw, R_bw, degrees=True):
    """
    ee_e2b: (N,6)  EEF→Base
    t_bw : (N,3)   ベース原点のワールド位置
    R_bw : (N,3,3) ベース→ワールドの回転
    戻り: P_e_w (N,3)  EEF原点のワールド位置
    """
    N = ee_e2b.shape[0]
    P = np.empty((N,3), np.float32)
    for k in range(N):
        T_e2b = row6_to_T(ee_e2b[k], degrees=degrees)  # E→B
        T_b2e = np.linalg.inv(T_e2b)                   # B→E
        t_be  = T_b2e[:3,3]                            # ベース座標でのEEF位置
        P[k]  = R_bw[k] @ t_be + t_bw[k]               # ワールドへ
    return P


import pandas as pd
data = pd.read_parquet("/groups/gag51454/workspace_oh/hsr/2025-06-v3.0-success-only/data/chunk-000/episode_000000.parquet")

# print(data["observation.head_pose.absolute"][0])
# print(data["observation.end_effector_pose.absolute"][0])

# import torchvision
# torchvision.set_video_backend("pyav")
# reader = torchvision.io.VideoReader("/groups/gag51454/workspace_oh/hsr/2025-06-v3.0-success-only/videos/chunk-000/observation.image.head/episode_000000.mp4", "video")
# frames = []
# for frame in reader:
#      frames.append(frame["data"].numpy())
# frames = np.array(frames)
# frames = frames.transpose(0, 2, 3, 1)
# print(frames[0])

# K = np.array([
#     [607.3814086914062,   0.0,                 315.9123840332031],
#     [  0.0,               607.2514038085938,   233.77308654785156],
#     [  0.0,                 0.0,                 1.0]
# ], dtype=np.float32)

# fx, fy = K[0,0], K[1,1]
# cx, cy = K[0,2], K[1,2]
# print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")

key   = "index"          # 区間キー列（数値/日時どれでもOK）
start, end = 0, 400 # 抜きたい行の区間
cols_head  = "observation.head_pose.absolute"  # 取り出す列（Noneなら全部）
cols_hand  = "observation.end_effector_pose.absolute"  # 取り出す列（Noneなら全部）
cols_base = "action.base"

mask_head = (data[key] >= start) & (data[key] < end)
poses_w2c = np.stack(data.loc[mask_head, cols_head].to_numpy(), axis=0).astype(np.float32)

mask_hand = (data[key] >= start) & (data[key] < end)
poses_w2o = np.stack(data.loc[mask_hand, cols_hand].to_numpy(), axis=0).astype(np.float32)

mask_base = (data[key] >= start) & (data[key] < end)
vel_world = np.stack(data.loc[mask_base, cols_base].to_numpy(), axis=0).astype(np.float32)

print(poses_w2o)
# 入力:
#   v_base: (N,3) = [vx, vy, omega]  ベース座標系の速度（m/s, m/s, rad/s or deg/s）
#   ee_b6 : (N,6) = [x,y,z, r,p,yaw] ベース→EEF の6DoF（単位: m, deg なら degrees=True）
#   freq  : サンプリング周波数（例: 30）

freq = 30.0
v_base = vel_world
ee_b6  = poses_w2o

# 1) baseのB→Wを積分で作る（最初フレームのベースが原点＝ワールド）
t_bw, R_bw, yaw = integrate_base_body_vel(v_base, freq=freq, theta0=0.0, omega_unit="rad")  # "deg"なら切替

# 2) EEFのワールド位置列
# P_eef_w = eef_world_positions_from_base(ee_b6, t_bw, R_bw, degrees=True)

# # 3) ベース原点のワールド位置（そのままプロット）
# P_base_w = t_bw

# 4) “最初のフレームを原点”にした相対表示にしたければ以下を有効化
# P_base_w = P_base_w - P_base_w[0]
# P_eef_w  = P_eef_w  - P_eef_w[0]




# (2)(3) EEF→Base を反転して B→EEF にし、EEFのワールド位置列を得る
P_eef_w = eef_world_positions_from_E2B(ee_b6, t_bw, R_bw, degrees=True)

# （任意）相対化：最初フレームを原点に平行移動
P_base_w = t_bw - t_bw[0]
# P_eef_w  = P_eef_w - P_eef_w[0]

print(P_eef_w[0])


plot_trajs_3d(P_base_w=None, P_eef_w=P_eef_w, title="Base & EEF trajectories (world = first base frame)")


plot_topdown_with_depth(
    [P_base_w, P_eef_w],
    labels=["Base","EEF"],
    view_axis='z',   # ← 'x','y','z' から選択
    sign=+1,         # +1: +Zを見る（上から下を見るイメージ） / -1: 逆向き
    cmap='viridis',  # 好みで 'plasma','turbo' など
    equal=True,
    show_colorbar=True,
    title="Top-down (view +Z) with depth colormap"
)





