# debug_instruction_probe.py
from pprint import pprint

# あなたのデータセット/ローダ作成コードに合わせて import を調整
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.embodiment_tags import EmbodimentTag

# === ここを環境に合わせて ===
DATASET_ROOT = "/home/group_25b505/group_6/workspace/user_00094_25b505/Isaac-GR00T/demo_data/043653_curated"  # meta/tasks.jsonl と episodes.jsonl がある場所
DATA_CONFIG = "hsr_v2"  # 実際に使っている設定名
EMB = EmbodimentTag.NEW_EMBODIMENT            # 実際に使っているタグ

ds = LeRobotSingleDataset(
    dataset_path=DATASET_ROOT,
    modality_configs=DATA_CONFIG_MAP[DATA_CONFIG].modality_config(),
    video_backend="torchvision_av",
    transforms=None,                    # 変換前の生データを見たい
    embodiment_tag=EMB,
)
print(len(ds))
print("=== sample -> (episode_index, chosen_instruction) ===")
for i in range(min(1000, len(ds))):
    sample = ds[i]
    meta = sample.get("meta", {})
    ep = meta.get("episode_index", None)

    # GR00T のローダは languageキー(=annotation系)にタスク文を入れる実装。
    # キー名は環境で異なる可能性があるので「annotation を含むキー」を拾う。
    lang_keys = [k for k in sample.keys() if "annotation" in k or k == "language"]
    if not lang_keys:
        print(i, "(ep:", ep, ")  <no language key>")
        continue
    lang_key = lang_keys[0]
    lang_val = sample[lang_key]
    # list の場合はローダが1タスクを選んで dict/str化しているか確認
    if isinstance(lang_val, list) and lang_val and isinstance(lang_val[0], (str, dict)):
        use_lang = lang_val[0]["task"] if isinstance(lang_val[0], dict) else lang_val[0]
    elif isinstance(lang_val, dict) and "task" in lang_val:
        use_lang = lang_val["task"]
    else:
        use_lang = lang_val  # str 想定

    print(f"{i:03d} -> (ep:{ep})  {use_lang}")
