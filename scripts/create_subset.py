#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Dict, Iterable
import pandas as pd
from tqdm import tqdm
import glob

def load_selected_episodes(arg: str) -> List[int]:
    """
    1) JSON/JSONL ファイル:  episode_index の配列 or {"your_key":[...]} 形式
    2) テキスト: カンマ区切り "1,2,3"
    3) JSONL の場合は各行に episode_index を含むとみなす
    """
    p = Path(arg)
    if p.exists():
        # try json
        try:
            obj = json.loads(p.read_text())
            if isinstance(obj, list):
                return sorted({int(x) for x in obj})
            if isinstance(obj, dict):
                # 最初の配列を拾う
                for v in obj.values():
                    if isinstance(v, list):
                        return sorted({int(x) for x in v})
                raise ValueError("JSON dict に配列が見つかりませんでした")
        except json.JSONDecodeError:
            # maybe jsonl
            eps = []
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "episode_index" in rec:
                            eps.append(int(rec["episode_index"]))
                    except Exception:
                        continue
            if eps:
                return sorted(set(eps))
            raise
    # comma separated
    return sorted({int(x) for x in arg.split(",") if x.strip()})

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def link_or_copy(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:
        raise ValueError(f"unknown mode: {mode}")

def read_info(src_root: Path) -> dict:
    info = json.loads((src_root/"meta/info.json").read_text())
    return info

def read_modality(src_root: Path) -> dict:
    #return json.loads((src_root/"meta/modality.json").read_text())
    return json.load(open("/home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T/modality.json","rb"))

def read_episodes(src_root: Path) -> List[dict]:
    out = []
    with (src_root/"meta/episodes.jsonl").open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_episodes(dst_root: Path, rows: Iterable[dict]):
    ensure_dir(dst_root/"meta")
    with (dst_root/"meta/episodes.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def write_json(dst_root: Path, rel: str, obj: dict):
    ensure_dir((dst_root/rel).parent)
    (dst_root/rel).write_text(json.dumps(obj, indent=2, ensure_ascii=False))

import os, json, shutil
from pathlib import Path
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag

def export_subset(ds: LeRobotSingleDataset, episode_ids: list[int], dst: str):
    dst = Path(dst)
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "data").mkdir(parents=True, exist_ok=True)
    (dst / "video").mkdir(parents=True, exist_ok=True)

    # --- 1. episodes.jsonl フィルタ
    ep_path = ds.dataset_path / "meta" / "episodes.jsonl"
    with open(ep_path) as f:
        episodes = [json.loads(line) for line in f]
    episodes_subset = [ep for ep in episodes if int(ep["episode_index"]) in episode_ids]
    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for ep in episodes_subset:
            f.write(json.dumps(ep) + "\n")

    # --- 2. meta ファイルをコピー（tasks.jsonl, modality.json, info.json, stats.json）
    for fname in ["tasks.jsonl", "modality.json", "info.json", "stats.json"]:
        src = ds.dataset_path / "meta" / fname
        if src.exists():
            shutil.copy2(src, dst / "meta" / fname)

    # --- 3. parquet コピー
    for ep in episodes_subset:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // ds.chunk_size
        parquet_src = ds.dataset_path / ds.data_path_pattern.format(
            episode_chunk=chunk, episode_index=ep_idx
        )
        parquet_dst = dst / ds.data_path_pattern.format(
            episode_chunk=chunk, episode_index=ep_idx
        )
        parquet_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parquet_src, parquet_dst)

    # --- 4. video コピー
    for ep in episodes_subset:
        ep_idx = ep["episode_index"]
        chunk = ep_idx // ds.chunk_size
        for video_key in ds.lerobot_modality_meta.video.keys():
            orig_key = ds.lerobot_modality_meta.video[video_key].original_key or video_key
            video_src = ds.dataset_path / ds.video_path_pattern.format(
                episode_chunk=chunk, episode_index=ep_idx, video_key=orig_key
            )
            video_dst = dst / ds.video_path_pattern.format(
                episode_chunk=chunk, episode_index=ep_idx, video_key=orig_key
            )
            if video_src.exists():
                video_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(video_src, video_dst)

    print(f"✅ Exported {len(episodes_subset)} episodes to {dst}")


def main():
    dataset_path = "/home/group_25b505/dataset/hsr/processed/2025-07-v3.0-success-only"
    data_config_cls = DATA_CONFIG_MAP["hsr_v2"]
    modality_configs = data_config_cls.modality_config()

    ds = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag("new_embodiment"),
    )

    # 例: 1511番と2117番だけ export
    export_subset(ds, episode_ids=[1511], dst="./subset_dataset")

if __name__ == "__main__":
    main()
