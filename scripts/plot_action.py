# plot_action_resampling.py
# 可視化はオフスクリーン（PNG保存のみ）

import os
import json
from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 重要：オフスクリーン
import matplotlib.pyplot as plt
import tyro

from gr00t.data.dataset import LeRobotSingleDataset, LeRobotMixtureDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING

@dataclass
class Args:
    # === 入力 ===
    dataset_path: List[str]
    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["decord", "torchvision_av"] = "decord"

    # エピソード指定（どちらか）
    include_json_path: str = ""      # 例: "/path/to/microwave-aist.json"
    pa_instruction: str = ""         # 上記JSONのキー名
    include_episodes_csv: str = ""   # "1511,1520,..." のようなCSV文字列（上のJSONを使わないならこちら）

    # === 間引き設定 ===
    sample_every_n: int = None
    target_fps: float = None

    # === 可視化対象 ===
    episode_index: int = -1          # 可視化する1本
    action_key: str = "action.relative"
    dims_to_show: Sequence[int] = (0, 1, 2, 5)  # 図に重ねる次元

    # === 出力 ===
    out_dir: str = "./debug_plots"
    fig_prefix: str = "traj"

# ---------- ユーティリティ ----------
def resolve_action_column(ds: LeRobotSingleDataset, action_key: str) -> str:
    sub = action_key.split(".", 1)[1]
    le_key = getattr(ds.lerobot_modality_meta, "action")[sub].original_key
    return le_key or sub

def get_action_series(ds: LeRobotSingleDataset, ep: int, action_key: str):
    df = ds.get_trajectory_data(ep)
    col = resolve_action_column(ds, action_key)
    assert col in df.columns, f"{col=} not in parquet columns"
    A = np.stack(df[col].to_numpy())                    # (T, D)
    ts = np.asarray(df["timestamp"], dtype=np.float64)  # (T,)
    return A, ts

def compose_sum_with_abs6(window: np.ndarray) -> np.ndarray:
    """6次元目(index=5)は最後の値、他は総和"""
    s = window.sum(axis=0)
    if s.shape[0] >= 6:
        s[5] = window[-1, 5]
    return s

def resample_stride(actions: np.ndarray, stride: int) -> np.ndarray:
    out = []
    T = actions.shape[0]
    i = 0
    while i < T:
        j = min(i + stride, T)
        out.append(compose_sum_with_abs6(actions[i:j]))
        i = j
    return np.stack(out, axis=0)

def pick_indices_by_fps(timestamps: np.ndarray, target_fps: float):
    period = 1.0 / float(target_fps)
    picked = []
    last_t = -np.inf
    for i, t in enumerate(timestamps):
        if t - last_t >= period or i == 0:
            picked.append(i)
            last_t = t
    return np.asarray(picked, dtype=int)

def resample_target_fps(actions: np.ndarray, timestamps: np.ndarray, target_fps: float):
    picked = pick_indices_by_fps(timestamps, target_fps)
    T = actions.shape[0]
    out = []
    for k, start in enumerate(picked):
        end = (picked[k+1]-1) if (k+1 < len(picked)) else (T-1)
        out.append(compose_sum_with_abs6(actions[start:end+1]))
    return np.stack(out, axis=0), picked

def to_cumulative_path(actions: np.ndarray):
    """相対→累積。6次元目(index=5)は絶対なのでそのまま"""
    cum = actions.copy()
    D = cum.shape[1]
    if D >= 6:
        abs6 = cum[:, 5].copy()
        idx = [i for i in range(D) if i != 5]
        cum[:, idx] = np.cumsum(cum[:, idx], axis=0)
        cum[:, 5] = abs6
    else:
        cum = np.cumsum(cum, axis=0)
    return cum

def rmse_at_indices(ref: np.ndarray, sub: np.ndarray, idx_ref: np.ndarray):
    """ref[idx_ref] と sub を比較して per-dim RMSE"""
    K = min(len(idx_ref), len(sub))
    diff = ref[idx_ref[:K]] - sub[:K]
    return np.sqrt((diff**2).mean(axis=0))

# ---------- メイン ----------
def main(args: Args):
    os.makedirs(args.out_dir, exist_ok=True)

    # include_episodes を決定
    include_episodes = None
    if args.include_json_path and args.pa_instruction:
        include_episodes = set(json.load(open(args.include_json_path, "rb"))[args.pa_instruction])
    elif args.include_episodes_csv:
        include_episodes = set(int(x) for x in args.include_episodes_csv.split(",") if x.strip())
    assert include_episodes is not None and len(include_episodes) > 0, "エピソード指定が必要です"

    # dataset 構築
    embodiment_tag = EmbodimentTag(args.embodiment_tag)
    data_cfg_cls = DATA_CONFIG_MAP[args.data_config]
    modality_configs = data_cfg_cls.modality_config()
    transforms = data_cfg_cls.transform()

    if len(args.dataset_path) == 1:
        ds = LeRobotSingleDataset(
            dataset_path=args.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=args.video_backend,
            include_episodes=include_episodes,
            sample_every_n=args.sample_every_n,
            target_fps=args.target_fps,
        )
    else:
        # 必要なら Mixture でもOK（可視化は1本を選ぶ）
        singles = []
        for p in args.dataset_path:
            singles.append(
                LeRobotSingleDataset(
                    dataset_path=p,
                    modality_configs=modality_configs,
                    transforms=transforms,
                    embodiment_tag=embodiment_tag,
                    video_backend=args.video_backend,
                    include_episodes=include_episodes,
                    sample_every_n=args.sample_every_n,
                    target_fps=args.target_fps,
                )
            )
        ds = LeRobotMixtureDataset(
            data_mixture=[(d, 1.0) for d in singles],
            mode="train",
            balance_dataset_weights=True,
            balance_trajectory_weights=True,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )
        # Mixture の場合、可視化する ep がどの Single にあるかはユーザ責務

    # 可視化対象エピソード
    ep = args.episode_index if args.episode_index >= 0 else sorted(list(include_episodes))[0]
    print(f"[INFO] visualize episode_index={ep}")

    # 元の系列（相対）
    A, ts = get_action_series(ds if isinstance(ds, LeRobotSingleDataset) else singles[0], ep, args.action_key)
    cum_full = to_cumulative_path(A)
    T, D = A.shape

    # sample_every_n
    cum_stride = None
    idx_stride = None
    rmse_s = None
    if args.sample_every_n and args.sample_every_n > 1:
        A_stride = resample_stride(A, args.sample_every_n)
        cum_stride = to_cumulative_path(A_stride)
        idx_stride = np.arange(0, T, args.sample_every_n)
        idx_stride[-1] = min(idx_stride[-1], T-1)
        rmse_s = rmse_at_indices(cum_full, cum_stride, idx_stride)

    # target_fps
    cum_fps = None
    picked = None
    rmse_f = None
    if args.target_fps and args.target_fps > 0:
        A_fps, picked = resample_target_fps(A, ts, args.target_fps)
        cum_fps = to_cumulative_path(A_fps)
        rmse_f = rmse_at_indices(cum_full, cum_fps, picked)

    # 保存：RMSE要約
    with open(os.path.join(args.out_dir, f"{args.fig_prefix}_ep{ep}_rmse.txt"), "w") as f:
        if rmse_s is not None:
            f.write(f"RMSE vs Full (sample_every_n={args.sample_every_n}):\n{rmse_s.tolist()}\n")
        if rmse_f is not None:
            f.write(f"RMSE vs Full (target_fps={args.target_fps}):\n{rmse_f.tolist()}\n")
    print("[INFO] RMSE written.")

    # 図保存：各次元ごと
    t_full = np.arange(T)
    if idx_stride is None: idx_stride = np.array([], dtype=int)
    if picked is None: picked = np.array([], dtype=int)

    for d in args.dims_to_show:
        plt.figure()
        plt.plot(t_full, cum_full[:, d], label="full cum")
        if cum_stride is not None:
            plt.plot(idx_stride[:len(cum_stride)], cum_stride[:, d], marker='o', linestyle='--',
                     label=f"stride={args.sample_every_n}")
        if cum_fps is not None:
            plt.plot(picked[:len(cum_fps)], cum_fps[:, d], marker='x', linestyle=':',
                     label=f"{args.target_fps} fps")
        plt.title(f"Episode {ep} - cumulative action dim {d}")
        plt.xlabel("frame index")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"{args.fig_prefix}_ep{ep}_dim{d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"[SAVE] {out_path}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
