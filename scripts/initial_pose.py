import numpy as np
from pathlib import Path
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag

# 例: 使うデータ設定（あなたの学習時と同じものを指定）
data_config_cls = DATA_CONFIG_MAP["hsr_v2"]
modality_configs = data_config_cls.modality_config()

# データセットを用意（学習と同じ引数に合わせてください）
ds = LeRobotSingleDataset(
    dataset_path="/home/group_25b505/dataset/hsr/processed/2025-07-v3.0-success-only",
    modality_configs=modality_configs,
    embodiment_tag=EmbodimentTag("new_embodiment"),  # 学習時の指定に合わせる
    video_backend="torchvision_av",                           # 使っているバックエンドに合わせる
)

# 抜き出したいエピソード ID（episodes.jsonl の episode_index）
EP = 1511

# 1) まず、このエピソードの軌跡表（低次元の元データ）を取り出せます
#traj_df = ds.get_trajectory_data(EP)  # pandas.DataFrame
# print(traj_df.head())  # 中身を確認したい時

# 2) 「最初の state（t=0）」を取り出す
#    データ設定で有効な state のキー一覧は ds.modality_keys["state"] に入っています
ds.curr_traj_data = ds.get_trajectory_data(EP)
ds.curr_traj_id = EP

first_state_dict = {}
for key in ds.modality_keys["state"]:
    arr = ds.get_state_or_action(trajectory_id=EP, modality="state", key=key, base_index=0)
    first_state_dict[key] = arr[0]

# 3) もし 1 本のベクトルに連結したい場合
first_state_vec = np.concatenate([first_state_dict[k] for k in ds.modality_keys["state"]], axis=-1)

# 確認
print("keys:", ds.modality_keys["state"])
for k, v in first_state_dict.items():
    print(k, v.shape)          # 各サブ state の次元
print("concat state:", first_state_vec)  # 連結後の次元

#microwave, open the door
#[ 0.28899965  0.00471395 -0.05098654 -1.42599928 -0.0487077   1.23381805 -0.03549254 -0.28434199]
#pick the item
#[ 0.38799953 -0.45471406  0.10700347 -1.12311625 -0.0467857  -0.87099898 -0.03546154 -0.28433797]
#place the item
#[ 0.41799724 -0.89841706  0.03200746 -0.78795028 -0.0487077  -0.114045 -0.03544654 -0.28433299]
#close the door
#[ 0.34099635 -0.38184807  0.01601346 -0.87800825 -0.0490887   1.23795402 -0.03544654 -0.28433797]


cfg = ds.lerobot_modality_meta.action["relative"]
orig_key = cfg.original_key or "relative"  # Parquet の列名

# 3) 全時刻 T × 全次元 から、relative の [start:end] だけ取り出す
#    df[orig_key] は各行が list/np.ndarray の列になっている前提
full = np.stack(ds.curr_traj_data[orig_key].to_list())          # 形状: (T, D_total)
idx  = np.arange(cfg.start, cfg.end)             # relative の有効インデックス
actions_rel = full[:, idx]                       # 形状: (T, D_relative)

# (オプション) タイムスタンプも欲しければ
timestamps = np.asarray(ds.curr_traj_data["timestamp"])

print(EP, actions_rel.shape)

out = Path(f"episode_{EP}_action_relative.npz")

np.savez_compressed(
    out,
    episode_index=EP,
    actions=actions_rel,          # 形状: (T, D_relative)
    timestamps=timestamps,        # 形状: (T,)
    shape=np.array(actions_rel.shape, dtype=np.int64)
)
print("saved ->", out)
