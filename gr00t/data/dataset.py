# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.utils.video import get_all_frames, get_frames_by_timestamps

from .embodiment_tags import EmbodimentTag
from .schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from .transform import ComposedModalityTransform


from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import os
import re
import shutil
import tempfile


LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in all_low_dim_data.columns:
        print(f"Computing statistics for {le_modality}...")
        # check if the data is the modality is actually a list of numbers
        # skip if it is a string
        if isinstance(all_low_dim_data[le_modality].iloc[0], str):
            print(f"Skipping {le_modality} because it is a string")
            continue

        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics

# ---------- ユーティリティ ----------

from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import heapq, math, os, re, tempfile, shutil

# ========= ユーティリティ =========

def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def _infer_dim_from_arrow_array(arr: pa.Array) -> int | None:
    # 変更前:
    # for i in range(len(arr)):
    #     if arr.is_valid(i):
    #         v = arr[i].as_py()
    #         ...

    # 変更後:
    for scalar in arr:              # scalar: pa.Scalar
        v = scalar.as_py()          # None なら欠損
        if v is None or isinstance(v, (str, bytes)):
            continue
        a = np.asarray(v)
        return 1 if a.ndim == 0 else int(np.prod(a.shape))
    return None

def _flatten_to_length(v, dim: int) -> np.ndarray | None:
    """v -> float32[dim]（形が合わなければ None）。"""
    if v is None or isinstance(v, (str, bytes)):
        return None
    a = np.asarray(v)
    # object や不正は弾く
    try:
        a = a.astype(np.float32, copy=False)
    except Exception:
        return None
    a = a.reshape(-1) if a.ndim > 0 else a.reshape(1)
    if a.shape[0] != dim:
        return None
    return a

class WelfordVec:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.M2 = np.zeros(dim, dtype=np.float64)
        self.vmin = np.full(dim, np.inf, dtype=np.float64)
        self.vmax = np.full(dim, -np.inf, dtype=np.float64)
    def update(self, X: np.ndarray):  # X: (B,dim) float32/64
        X = X.astype(np.float64, copy=False)
        self.vmin = np.minimum(self.vmin, np.min(X, axis=0))
        self.vmax = np.maximum(self.vmax, np.max(X, axis=0))
        b = X.shape[0]
        n0 = self.n
        self.n = n0 + b
        delta = X - self.mean
        self.mean = self.mean + np.sum(delta, axis=0) / self.n
        delta2 = X - self.mean
        self.M2 = self.M2 + np.sum(delta * delta2, axis=0)
    def finalize(self):
        if self.n < 2:
            std = np.zeros(self.dim, dtype=np.float64)
        else:
            var = self.M2 / (self.n - 1)
            std = np.sqrt(np.maximum(var, 0.0))
        return (self.mean.astype(np.float32),
                std.astype(np.float32),
                self.vmin.astype(np.float32),
                self.vmax.astype(np.float32),
                int(self.n))

# ========= 外部ソートで分位（線形補間） =========

def _write_sorted_runs(values: Iterable[float], run_size: int, tmpdir: Path) -> List[Path]:
    """values を run_size ごとに集めて in-memory sort → .npy（昇順）として保存。"""
    runs = []
    buf = []
    for v in values:
        buf.append(v)
        if len(buf) >= run_size:
            arr = np.asarray(buf, dtype=np.float32)
            arr.sort(kind="quicksort")
            run_path = tmpdir / f"run_{len(runs):06d}.npy"
            np.save(run_path, arr)
            runs.append(run_path)
            buf.clear()
    if buf:
        arr = np.asarray(buf, dtype=np.float32)
        arr.sort(kind="quicksort")
        run_path = tmpdir / f"run_{len(runs):06d}.npy"
        np.save(run_path, arr)
        runs.append(run_path)
    return runs

def _kth_from_sorted_runs(run_paths: List[Path], k: int) -> float:
    """昇順ラン群の k 番目（0-index）の値を k-way merge で取得。"""
    # 各 run をメモリに丸ごと載せず、必要になったらチャンク読みでもOK。
    # ここでは各 run を mmap で読む（OS に任せる）。
    arrays = [np.load(p, mmap_mode="r") for p in run_paths]
    # ポインタと値で min-heap を作る
    heap = []
    for i, arr in enumerate(arrays):
        if arr.size > 0:
            heap.append((arr[0], i, 0))  # (value, run_id, index_in_run)
    heapq.heapify(heap)
    popped = -1
    while heap:
        val, rid, idx = heapq.heappop(heap)
        popped += 1
        if popped == k:
            return float(val)
        nxt = idx + 1
        if nxt < arrays[rid].size:
            heapq.heappush(heap, (arrays[rid][nxt], rid, nxt))
    raise IndexError("k is out of range")

def _quantile_linear_from_runs(run_paths: List[Path], q: float, N: int) -> float:
    """NumPy の method='linear' と一致（pos=q*(N-1), 線形補間）。"""
    if N == 0:
        return float("nan")
    if q <= 0.0:
        return float(np.load(run_paths[0], mmap_mode="r")[0])
    if q >= 1.0:
        # 最大は各 run の末尾の最大値
        mx = -np.inf
        for p in run_paths:
            a = np.load(p, mmap_mode="r")
            if a.size:
                mx = max(mx, float(a[-1]))
        return mx
    pos = q * (N - 1)
    k = int(math.floor(pos))
    r = pos - k
    vk  = _kth_from_sorted_runs(run_paths, k)
    if r == 0.0:
        return vk
    vk1 = _kth_from_sorted_runs(run_paths, k + 1)
    return (1.0 - r) * vk + r * vk1

# ========= 本体：厳密＆低RAM =========

def calculate_dataset_statistics_exact_streaming(
    parquet_paths: List[Path],
    *,
    tmpdir: str | None = None,
    keep_tmp: bool = False,
    read_columns: List[str] | None = None,
    batch_rows: int = 100_000,     # Arrowの record batch サイズ（RAM に合わせて）
    run_size: int = 5_000_000      # 1 ランの要素数（RAM に合わせて調整）
) -> Dict[str, dict]:
    """
    - 読み込みは Arrow の iter_batches で本当のストリーミング
    - mean/std/min/max は Welford（厳密）
    - q01/q99 は外部ソート＋k-way merge で厳密（NumPy の線形補間一致）
    - 列×次元を “順番に” 処理（同時保持しない）
    """
    if not parquet_paths:
        return {}

    # 作業ディレクトリ
    own_tmp = False
    if tmpdir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="parq_stats_extsort_"))
        own_tmp = True
    else:
        work_dir = Path(tmpdir); work_dir.mkdir(parents=True, exist_ok=True)

    # 列一覧（スキーマは最初のファイルから）
    first_pq = pq.ParquetFile(str(parquet_paths[0]))
    cols_all = [m.name for m in first_pq.schema_arrow]
    target_cols = read_columns if read_columns is not None else cols_all

    results: Dict[str, dict] = {}

    # 列を1つずつ処理
    for col in target_cols:
        # 次元推定（最初のファイルの先頭の有効セルから）
        dim = None
        for pf_path in parquet_paths:
            pf = pq.ParquetFile(str(pf_path))
            for batch in pf.iter_batches(columns=[col], batch_size=batch_rows):
                arr: pa.Array = batch.column(0)
                dim = _infer_dim_from_arrow_array(arr)
                if dim is not None:
                    break
            if dim is not None:
                break
        if dim is None:
            # 数値でなさそう（文字列など）→ スキップ
            continue

        # 統計器
        w = WelfordVec(dim)
        N_total = 0

        # --- パス1: Welford 集計 + 件数 N の把握 ---
        for pf_path in tqdm(parquet_paths, desc=f"[Pass1] {col}"):
            pf = pq.ParquetFile(str(pf_path))
            for batch in pf.iter_batches(columns=[col], batch_size=batch_rows):
                arr: pa.Array = batch.column(0)
                mats = []
                # Arrow Array を Python 値で取り出す（ゼロコピーではないが安全）
                for i in range(len(arr)):
                    v = arr[i].as_py()          # ← is_valid(i) を使わない
                    a = _flatten_to_length(v, dim)
                    if a is not None:
                        mats.append(a)
                if mats:
                    X = np.vstack(mats)  # チャンク内のみ
                    w.update(X)
                    N_total += X.shape[0]

        if N_total == 0:
            # 空列
            results[col] = {
                "mean": [0.0]*dim, "std": [0.0]*dim, "min": [np.inf]*dim, "max": [-np.inf]*dim,
                "count": 0, "q01": [], "q99": [],
            }
            continue

        mean, std, vmin, vmax, _ = w.finalize()

        # --- パス2: 各次元ごとに外部ソートの「ラン」を作る ---
        q01 = np.empty(dim, dtype=np.float32)
        q99 = np.empty(dim, dtype=np.float32)

        for d in range(dim):
            dim_dir = work_dir / f"{_safe_name(col)}__d{d}"
            dim_dir.mkdir(parents=True, exist_ok=True)

            def value_stream() -> Iterable[float]:
                for pf_path in parquet_paths:
                    pf = pq.ParquetFile(str(pf_path))
                    for batch in pf.iter_batches(columns=[col], batch_size=batch_rows):
                        arr: pa.Array = batch.column(0)
                        for i in range(len(arr)):
                            v = arr[i].as_py()
                            a = _flatten_to_length(v, dim)
                            if a is not None:
                                yield float(a[d])

            run_paths = _write_sorted_runs(value_stream(), run_size=run_size, tmpdir=dim_dir)
            # 厳密分位（線形補間）
            q01[d] = _quantile_linear_from_runs(run_paths, 0.01, N_total)
            q99[d] = _quantile_linear_from_runs(run_paths, 0.99, N_total)

            # ランは次元ごとに削除（中間ファイルを増やし過ぎない）
            if not keep_tmp:
                shutil.rmtree(dim_dir, ignore_errors=True)

        results[col] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": vmin.tolist(),
            "max": vmax.tolist(),
            "count": int(N_total),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }

    if own_tmp and not keep_tmp:
        shutil.rmtree(work_dir, ignore_errors=True)

    return results




class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        include_episodes: set[int] | None = None,
        sample_every_n: int | None = None,
        target_fps: float | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self.target_fps = target_fps
        self.sample_every_n = sample_every_n

        
        self.include_episodes = include_episodes
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._all_steps = self._get_all_steps()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._picked_indices_per_traj: dict[int, list[int]] = {}

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        #modality_meta_path = "/home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T/modality.json"
        #assert (
        #    modality_meta_path.exists()
        #), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"

        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
                #fps = le_video_meta["info"]["video.fps"]
            except (ValueError, KeyError):
                channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                #channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }
        
        self.fps = fps

        # 2. Dataset statistics
        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        #stats_path = "/home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T/stats.json"
        try:
            with open(stats_path, "r") as f:
                le_statistics = json.load(f)
            for stat in le_statistics.values():
                DatasetStatisticalValues.model_validate(stat)
        except (FileNotFoundError, ValidationError) as e:
            print(f"Failed to load dataset statistics: {e}")
            print(f"Calculating dataset statistics for {self.dataset_name}")
            # Get all parquet files in the dataset paths
            parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
            #le_statistics = calculate_dataset_statistics(parquet_files)
            le_statistics = calculate_dataset_statistics_exact_streaming(parquet_files)
            with open(stats_path, "w") as f:
                json.dump(le_statistics, f, indent=4)
        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    # def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
    #     """Get the trajectories in the dataset."""
    #     # Get trajectory lengths, IDs, and whitelist from dataset metadata
    #     episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
    #     with open(episode_path, "r") as f:
    #         episode_metadata = [json.loads(line) for line in f]
    #     trajectory_ids = []
    #     trajectory_lengths = []
    #     for episode in episode_metadata:
    #         trajectory_ids.append(episode["episode_index"])
    #         trajectory_lengths.append(episode["length"])
    #     return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        episode_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
        trajectory_ids = []
        trajectory_lengths = []
        with open(episode_path, "r") as f:
            for line in f:
                ep = json.loads(line)
                ep_idx = int(ep["episode_index"])
                if self.include_episodes is not None and ep_idx not in self.include_episodes:
                    continue
                trajectory_ids.append(ep_idx)
                trajectory_lengths.append(int(ep["length"]))
        return np.array(trajectory_ids), np.array(trajectory_lengths)

    # def _get_all_steps(self) -> list[tuple[int, int]]:
    #     """Get the trajectory IDs and base indices for all steps in the dataset.

    #     Returns:
    #         list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.

    #     Example:
    #         self.trajectory_ids: [0, 1, 2]
    #         self.trajectory_lengths: [3, 2, 4]
    #         return: [
    #             ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
    #             ("traj_1", 0), ("traj_1", 1),
    #             ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
    #         ]
    #     """
    #     all_steps: list[tuple[int, int]] = []
    #     for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
    #         for base_index in range(trajectory_length):
    #             all_steps.append((trajectory_id, base_index))
    #     return all_steps

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """(trajectory_id, base_index) の列挙を、必要なら間引いて返す"""
        all_steps: list[tuple[int, int]] = []
        self._picked_indices_per_traj = {}

        # 1) 等間引きが指定されていれば速い
        if self.sample_every_n is not None and self.sample_every_n > 1:
            for trajectory_id, traj_len in zip(self.trajectory_ids, self.trajectory_lengths):
                picked = list(range(0, int(traj_len), int(self.sample_every_n)))
                self._picked_indices_per_traj[int(trajectory_id)] = picked
                for base_index in range(0, int(traj_len), self.sample_every_n):
                    all_steps.append((int(trajectory_id), int(base_index)))
            return all_steps

        # 2) target_fps に基づく時間間引き
        if self.target_fps is not None and self.target_fps > 0:
            # 各トラジェクトリの timestamp からインデックスを選ぶ
            for trajectory_id, traj_len in zip(self.trajectory_ids, self.trajectory_lengths):
                # parquet を1本読むが、キャッシュが効くのでOK
                df = self.get_trajectory_data(int(trajectory_id))
                assert "timestamp" in df.columns, "timestamp カラムが必要です"
                ts = np.asarray(df["timestamp"], dtype=np.float64)
                # 秒系で来ている想定。ナノ秒等なら適宜スケール調整。
                period = 1.0 / float(self.target_fps)
                picked = []
                last_t = -np.inf
                for idx, t in enumerate(ts):
                    if t - last_t >= period or idx == 0:
                        picked.append(idx)
                        last_t = t
                self._picked_indices_per_traj[int(trajectory_id)] = picked
                for base_index in picked:
                    all_steps.append((int(trajectory_id), int(base_index)))
            return all_steps

        # 3) 間引きなし（従来動作）
        for trajectory_id, trajectory_length in zip(self.trajectory_ids, self.trajectory_lengths):
            picked = list(range(int(trajectory_length)))
            self._picked_indices_per_traj[int(trajectory_id)] = picked
            for base_index in range(int(trajectory_length)):
                all_steps.append((int(trajectory_id), int(base_index)))

        
        return all_steps

    

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                ind = np.array(config.delta_indices)
                if self.sample_every_n is not None:
                    delta_indices[key] = self.dilate_delta_indices(ind, stride=self.sample_every_n)
                else:
                    delta_indices[key] = ind
        return delta_indices

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        #modality_meta_path = "/home/group_25b505/group_6/workspace/user_00031_25b505/Isaac-GR00T/modality.json"
        #assert (
        #    modality_meta_path.exists()
        #), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
        with open(tasks_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        df = pd.DataFrame(tasks)
        return df.set_index("task_index")

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        out = self.transforms(self.get_step_data(trajectory_id, base_index))
        #out["episode_index"] = trajectory_id
        return out

    def dilate_delta_indices(self, delta_indices, stride: int):
        return np.array([x * stride for x in delta_indices])

    def _debug_get_episode_index(self, trajectory_id: int) -> int:
        for attr in (
            "_trajectory_id_to_episode_index",
            "_trajectory_to_episode",
            "trajectory_id_to_episode_index",
            "trajectory_to_episode",
        ):
            mapping = getattr(self, attr, None)
            if isinstance(mapping, (list, tuple)):
                try:
                    return int(mapping[trajectory_id])
                except Exception:
                    pass
            if isinstance(mapping, dict):
                if trajectory_id in mapping:
                    return int(mapping[trajectory_id])

        trajs = getattr(self, "_trajectories", None)
        if isinstance(trajs, list) and 0 <= trajectory_id < len(trajs):
            tr = trajs[trajectory_id]
            if isinstance(tr, dict):
                for k in ("episode_index", "episode_id", "episode"):
                    if k in tr:
                        return int(tr[k])

        # 3) LeRobot系：trajectory-task のペア配列がある場合
        pairs = getattr(self, "_trajectory_task_pairs", None)
        if isinstance(pairs, list) and 0 <= trajectory_id < len(pairs):
            try:
                ep_idx = pairs[trajectory_id][0]
                return int(ep_idx)
            except Exception:
                pass

        # 4) それでも無ければ、最終手段：trajectory_id をそのまま返す
        return int(trajectory_id)

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)

        #import sys; sys.exit()


        #print("gggggggggggg")
        #data["episode_index"] = int(self._trajectories[trajectory_id].get("episode_index", trajectory_id))
        #data["meta"] = {
        #    "episode_index": int(trajectory_id),
        #    "__debug_frame_start": int(base_index),
        #    "__debug_window_len": int(getattr(self, "state_horizon", 0)),
        #}
        # 互換性のため top-level も置いておく（既存の sample["episode_index"] を壊さない）
        #data["episode_index"] = int(trajectory_id)
        #data["episode_index"] = self._debug_get_episode_index(trajectory_id)
        #data["__debug_frame_start"] = int(base_index)
        #data["__debug_window_len"] = int(self.state_horizon if hasattr(self, "state_horizon") else 0)
        return data

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else:
            chunk_index = self.get_episode_chunk(trajectory_id)
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            return pd.read_parquet(parquet_path)

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        video_filename = self.video_path_pattern.format(
            episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
        )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = np.array(self.delta_indices[key]) + base_index
        #print(f"{key}:{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = np.array(self.delta_indices[key]) + base_index
        #print(f"{key}:{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_state_or_action_concat(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = np.array(self.delta_indices[key]) + base_index
        #print(f"{key}:{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]
        #is_absolute = state_or_action_cfg.absolute
        #rotation_type = state_or_action_cfg.rotation_type  # "quaternion" など

        # ここから「取り出しインデックス」を作る（既存）
        #step_indices = np.array(self.delta_indices[key]) + base_index
        #trajectory_index = self.get_trajectory_index(trajectory_id)
        #max_length = self.trajectory_lengths[trajectory_index]
        # 端処理はこの後 retrieve_data_and_pad に任せる

        # ---- 重要：relative action を間引いているときの合成処理 ----
        if modality == "action" and (
            (self.sample_every_n is not None and self.sample_every_n > 1) or
            (self.target_fps is not None and self.target_fps > 0)
        ):
            # 1. まず「大きい刻み」の基準インデックス列を得る（既存の step_indices）
            # 2. 各インデックス i について、「次の採用点まで」の全deltaを合成して1本にする
            #    例えば sample_every_n = s のとき、i..i+(s-1) を合成
            picked = self._picked_indices_per_traj.get(int(trajectory_id), None)

            composed_list = []
            for idx in step_indices:
                # 合成区間の幅を決める
                if self.sample_every_n is not None and self.sample_every_n > 1:
                    stride = int(self.sample_every_n)
                    start = int(np.clip(idx, 0, max_length - 1))
                    end   = int(np.clip(idx + stride - 1, 0, max_length - 1))
                else:
                    # target_fps の場合：次の採用点まで
                    # base の idx から picked 内で「次の採用点」を探す
                    # idx 自体が picked に居ないこともあるので、>=idx の位置を探す
                    import bisect
                    if picked is None or len(picked) == 0:
                        start = int(np.clip(idx, 0, max_length - 1))
                        end   = start
                    else:
                        pos = bisect.bisect_left(picked, int(idx))
                        # 現在区間の開始は idx、終了は「次の picked の直前」
                        start = int(np.clip(idx, 0, max_length - 1))
                        if pos + 1 < len(picked):
                            end = int(np.clip(picked[pos + 1] - 1, 0, max_length - 1))
                        else:
                            end = max_length - 1

                window = data_array[start:end + 1]          # (K, D)
                composed = self._compose_sum_with_abs6(window)
                composed_list.append(composed)

            composed = np.stack(composed_list, axis=0)       # (T, D)`

            # relative を区間合成済みなので padding は 0 でOK
            return self.retrieve_data_and_pad(
                array=composed,
                step_indices=np.arange(len(composed)),
                max_length=len(composed),
                padding_strategy="zero",
            )

        # ---- それ以外（従来の1ステップ取り出し） ----
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = np.array(self.delta_indices[key]) + base_index
        #print(f"{key}:{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):
            task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            #return self.get_state_or_action(trajectory_id, modality, key, base_index)
            return self.get_state_or_action_concat(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _compose_relative_actions(self, arr_window: np.ndarray, rotation_type: str | None):
        """
        arr_window: 形状 (K, D) の相対アクション列（Kは区間内のフレーム数）
        rotation_type: "quaternion" | "axis_angle" | "euler" | None
        戻り値: 形状 (D,) の1本に合成された相対アクション
        """
        vec = arr_window.copy()  # (K, D)

        # 例: 配列の先頭から [pos, rot, gripper] のように並ぶ想定なら
        # メタデータで各サブキーの範囲が取れるのが理想だけど、
        # ここでは簡単のために rotation_type がある＝回転成分が含まれる前提で分岐。
        if rotation_type in ("quaternion", "axis_angle", "euler"):
            # --- 並進成分は総和 ---
            # 例) 並進3要素が先頭にあると仮定
            # 必要ならメタの start/end で厳密に切り出して
            trans = vec[:, :3].sum(axis=0)

            # --- 回転成分は合成 ---
            if rotation_type == "quaternion":
                # vec[:, 3:7] が dq（相対回転クォータニオン）と仮定
                q = np.array([1.0, 0.0, 0.0, 0.0])  # 単位Quat(w,x,y,z)
                for dq in vec[:, 3:7]:
                    # 正規化（数値安定）
                    dq = dq / (np.linalg.norm(dq) + 1e-12)
                    # 合成 q <- q * dq
                    w1,x1,y1,z1 = q
                    w2,x2,y2,z2 = dq
                    q = np.array([
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2
                    ])
                q = q / (np.linalg.norm(q) + 1e-12)
                rot = q
                tail = vec[:, 7:]  # 残り（グリッパ等）
            elif rotation_type == "axis_angle":
                # 近似: 小角なら単純加算でも大抵OK。厳密にはexp/logでSE(3)合成。
                rot = vec[:, 3:6].sum(axis=0)
                tail = vec[:, 6:]
            elif rotation_type == "euler":
                # オイラーは順序依存＆合成が不安定。小角想定で総和か、クォータニオン化が無難。
                rot = vec[:, 3:6].sum(axis=0)
                tail = vec[:, 6:]

            # tail の扱い（例：グリッパは最後の値）
            if tail.size > 0:
                last = vec[-1, -tail.shape[1]:]
                out = np.concatenate([trans, rot, last], axis=0)
            else:
                out = np.concatenate([trans, rot], axis=0)
            return out
        else:
            # 回転なし：並進は総和、離散は最後、レート量は平均など
            # ここはプロジェクト仕様に合わせて調整
            return vec.sum(axis=0)  # とりあえず総和

    def _compose_sum_with_abs6(self, window: np.ndarray) -> np.ndarray:
        """
        window: 形状 (K, D) の連続相対アクション列
        戻り値: 形状 (D,) の1本に合成されたアクション
        ルール: index!=5 は総和、index==5 は window の最後の値
        """
        s = window.sum(axis=0)              # 全次元をまず総和
        if s.shape[0] >= 6:
            s[5] = window[-1, 5]            # 6次元目だけ絶対（最後を採用）
        return s



class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = np.array(self.delta_indices[key]) + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)
        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])

        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for dataset in self.datasets:
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths
            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            raise ValueError(
                "No primary dataset found, please at least set one dataset's weight to 1.0"
            )

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = rng.choice(
            len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        dataset, trajectory_name, step = self.sample_step(index)
        return dataset.transforms(dataset.get_step_data(trajectory_name, step))

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        return int(
            (self.dataset_lengths / self.dataset_sampling_weights)[
                self.primary_dataset_indices
            ].max()
        )

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(configs) == 1
            ), f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict) -> None:
        """Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

from dataclasses import dataclass
from typing import Any

# Import from your tree
# from .dataset import LeRobotSingleDataset, DatasetMetadata


@dataclass
class _MixtureItem:
    ds: "LeRobotSingleDataset"
    weight: float


class LeRobotMultiEmbodimentMixtureDataset(Dataset):
    """
    Mixture dataset that **does not merge** stats/modality metadata across datasets
    (i.e., supports *multi‑embodiment* cleanly). Each underlying dataset keeps its
    own `metadata` (stats, modality.json) and `embodiment_tag`.

    Key differences vs. LeRobotMixtureDataset:
    - No aggregation of statistics or modalities. No `update_metadata()`.
    - `__getitem__` returns a sample **plus** `embodiment_tag` and `dataset_name` so
      downstream code can branch on embodiment when needed (e.g., per‑embodiment heads,
      different normalizers, etc.).
    - Sampling and epoch behavior are unchanged from your original mixture (stratified by
      dataset weights and within‑dataset trajectory lengths if desired).

    Assumptions:
    - Each `LeRobotSingleDataset` instance already called `set_transforms_metadata(dataset.metadata)`
      inside its constructor (as in your current implementation). Thus per‑dataset transforms
      see the correct per‑embodiment stats and modality config.
    - Your model/collator can accept heterogeneous modality key‑sets. If you need padding/union
      behavior, see `multiembodiment_collate` below.
    """

    def __init__(
        self,
        data_mixture: Sequence[Tuple["LeRobotSingleDataset", float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict | None = None,
    ) -> None:
        super().__init__()
        assert mode in {"train", "val", "test"}

        self._mixture: List[_MixtureItem] = [
            _MixtureItem(ds=ds, weight=float(w)) for (ds, w) in data_mixture
        ]
        self.mode = mode
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed

        # 1) dataset lengths (number of *steps*, not number of trajectories)
        self._dataset_lengths = np.array([len(m.ds) for m in self._mixture], dtype=np.int64)

        # 2) sampling weights across datasets
        self._dataset_sampling_weights = np.array([m.weight for m in self._mixture], dtype=np.float64)
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        sw = self._dataset_sampling_weights.sum()
        if sw <= 0:
            raise ValueError("All dataset weights are zero.")
        self._dataset_sampling_weights /= sw

        # 3) trajectory sampling weights per dataset
        self._traj_sampling_weights: List[np.ndarray] = []
        for m in self._mixture:
            traj_w = np.ones(len(m.ds.trajectory_lengths), dtype=np.float64)
            if self.balance_trajectory_weights:
                traj_w *= m.ds.trajectory_lengths
            s = traj_w.sum()
            if s <= 0:
                raise ValueError(f"Dataset {m.ds.dataset_name} has no trajectories.")
            traj_w /= s
            self._traj_sampling_weights.append(traj_w)

        # 4) choose a primary length to define __len__ (like original impl)
        #    We emulate the same behavior: epochs are as long as the *largest*
        #    effective dataset once divided by its sampling weight.
        mask = self._dataset_sampling_weights > 0
        if not np.any(mask):
            raise ValueError("All dataset sampling weights are zero after normalization.")
        eff = self._dataset_lengths[mask] / self._dataset_sampling_weights[mask]
        self._epoch_len = int(np.ceil(eff.max()))

        self.set_epoch(0)

        self.merged_metadata: dict[str, DatasetMetadata] = {}
        seen: set[str] = set()
        for ds in self._mixture:
            tag = ds.ds.metadata.embodiment_tag
            key = tag.value if hasattr(tag, "value") else str(tag)
            if key not in seen:
                self.merged_metadata[key] = ds.ds.metadata
                seen.add(key)

    # ---------- Public helpers ----------
    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights
    @property
    def datasets(self):
        # TrainRunner 互換: List[LeRobotSingleDataset] を返す
        return [m.ds for m in self._mixture]

    def per_embodiment_metadatas(self) -> Dict[str, List["DatasetMetadata"]]:
        """Return *all* metadatas grouped by embodiment_tag (no merging).
        Note: multiple datasets can share the same tag but have different modality.json;
        we therefore return a LIST per tag.
        """
        out: Dict[str, List["DatasetMetadata"]] = {}
        for m in self._mixture:
            tag = m.ds.metadata.embodiment_tag
            out.setdefault(tag.value, []).append(m.ds.metadata)
        return out

    def per_dataset_metadatas(self) -> Dict[str, "DatasetMetadata"]:
        """Return metadata keyed by dataset name (useful if multiple datasets share a tag)."""
        out: Dict[str, "DatasetMetadata"] = {}
        for m in self._mixture:
            out[m.ds.dataset_name] = m.ds.metadata
        return out

    # ---------- Epoch / sampling ----------
    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._epoch_len

    def _rng_for_index(self, index: int) -> np.random.Generator:
        seed = index if self.mode != "train" else _safe_hash((self.epoch, index, self.seed))
        return np.random.default_rng(seed)

    def _sample_step(self, index: int) -> tuple["LeRobotSingleDataset", int, int]:
        rng = self._rng_for_index(index)
        # choose dataset
        ds_idx = rng.choice(len(self._mixture), p=self._dataset_sampling_weights)
        m = self._mixture[int(ds_idx)]
        ds = m.ds
        # choose trajectory inside ds
        traj_idx = rng.choice(len(ds.trajectory_ids), p=self._traj_sampling_weights[int(ds_idx)])
        traj_id = int(ds.trajectory_ids[int(traj_idx)])
        # choose base step inside that trajectory
        base = int(rng.choice(ds.trajectory_lengths[int(traj_idx)]))
        return ds, traj_id, base

    # def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
    #     rng = self._rng_for_index(index)
    #     # dataset
    #     ds_idx = int(rng.choice(len(self.datasets), p=self.dataset_sampling_weights))
    #     ds = self.datasets[ds_idx]
    #     # trajectory
    #     traj_idx = int(rng.choice(len(ds.trajectory_ids), p=self.trajectory_sampling_weights[ds_idx]))
    #     traj_id = int(ds.trajectory_ids[traj_idx])
    #     # step
    #     base_index = int(rng.choice(ds.trajectory_lengths[traj_idx]))
    #     return ds, traj_id, base_index

    # def __getitem__(self, index: int) -> Dict[str, Any]:
    #     ds, traj_id, base = self._sample_step(index)
    #     sample = ds.get_step_data(traj_id, base)
    #     # apply per‑dataset transforms (already owning correct per‑embodiment metadata)
    #     sample = ds.transforms(sample)
    #     # Attach identification for downstream routing
    #     sample["embodiment_tag"] = str(ds.metadata.embodiment_tag)
    #     sample["dataset_name"] = ds.dataset_name
    #     sample["episode_index"] = int(traj_id)  # helpful for logging/debug
    #     sample["__base_index"] = int(base)
    #     return sample
    
    def __getitem__(self, index: int) -> dict:
        ds, traj_id, base = self._sample_step(index)
        sample = ds.get_step_data(traj_id, base)
        sample = ds.transforms(sample)  # ← 各DSが自前のmetadataを見て動く
        # 下流でルーティングしやすい識別情報を付与（数値のままでもOK）
        #tag = ds.metadata.embodiment_tag
        #sample["embodiment_tag"] = tag.value if hasattr(tag, "value") else str(tag)
        #sample["dataset_name"]   = ds.dataset_name
        #sample["episode_index"]  = int(traj_id)
        #sample["__base_index"]   = int(base)
        return sample

    # def per_embodiment_metadatas(self) -> dict[str, list[DatasetMetadata]]:
    #     out: dict[str, list[DatasetMetadata]] = {}
    #     for ds in self.datasets:
    #         tag = ds.metadata.embodiment_tag
    #         key = tag.value if hasattr(tag, "value") else str(tag)
    #         out.setdefault(key, []).append(ds.metadata)
    #     return out

    # def per_dataset_metadatas(self) -> dict[str, DatasetMetadata]:
    #     return {ds.dataset_name: ds.metadata for ds in self.datasets}

    


# ---------- Optional: tolerant collate for heterogeneous modalities ----------
# This keeps per‑key tensors that can stack cleanly; for keys with mismatched shapes
# or non‑ndarray types (e.g., lists of strings), it keeps them as a Python list.
# Replace with your project’s collator if you already have one.

def multiembodiment_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    import torch
    out: Dict[str, Any] = {}
    keys = set().union(*(b.keys() for b in batch))
    for k in keys:
        vals = [b.get(k) for b in batch]
        # Try to stack numpy/torch arrays of identical shape
        try:
            if isinstance(vals[0], torch.Tensor):
                if all((v is not None) and isinstance(v, torch.Tensor) and v.shape == vals[0].shape for v in vals):
                    out[k] = torch.stack(vals, dim=0)
                else:
                    out[k] = vals
            elif hasattr(vals[0], "shape"):
                arrs = [torch.as_tensor(v) for v in vals]
                if all(a.shape == arrs[0].shape for a in arrs):
                    out[k] = torch.stack(arrs, dim=0)
                else:
                    out[k] = vals
            else:
                out[k] = vals
        except Exception:
            out[k] = vals
    return out


# ---------- small utility ----------

def _safe_hash(tup) -> int:
    import hashlib
    s = repr(tup).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest(), 16) & 0xFFFFFFFF
