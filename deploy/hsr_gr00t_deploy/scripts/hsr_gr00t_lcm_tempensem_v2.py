#!/usr/bin/env python3
from collections import deque
from typing import Any, List

import cv2
import numpy as np

# gr00t 関連
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

import lcm
from lcm_msgs import RequestMsg, ResponseMsg


# ====== Utilities ======
def compressedimage_to_array_lcm(msg):
    # Convert signed bytes (-128~127) to unsigned (0~255)
    signed = np.array(msg.data[: msg.data_length], dtype=np.int8)
    unsigned = signed.astype(np.uint8)
    # NOTE: 元コードの都合で BGR->RGB 変換は行わない（学習時も同じ前提）
    image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, :]
    # 正しく RGB にしたい場合は下記に変更（学習に合わせること）
    # image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, ::-1]
    return image


# ====== LCM Server ======
class HSRLcmServer:
    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.2  # グリッパーを閉じる閾値

    def __init__(self, policy, traj_hz=10.0):
        self.traj_hz = float(traj_hz)
        self._lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        self._lc.subscribe("REQUEST_CHANNEL", self._handle_request)
        self.policy = policy

        self.joint_state_names: List[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

    def handle(self):
        self._lc.handle()

    def _handle_request(self, channel, data):
        request = RequestMsg.decode(data)
        head_rgb = compressedimage_to_array_lcm(request.head_rgb)
        hand_rgb = compressedimage_to_array_lcm(request.hand_rgb)

        try:
            # self.joint_state_names の順に並び替え
            joint_msg = request.joint_state
            name_list = joint_msg.name[: joint_msg.num_joints]
            position_list = joint_msg.position[: joint_msg.num_joints]
            joint_state = [position_list[name_list.index(name)] for name in self.joint_state_names]

            observation = {
                "head_rgb": head_rgb,
                "hand_rgb": hand_rgb,
                "joint_state": np.array(joint_state, dtype=np.float32),
                "instruction": request.instruction,
            }

            action = self.policy.act(observation)

            action_joint_names = self.joint_state_names + ["base_x", "base_y", "base_t"]
            if action.shape[1] != len(action_joint_names):
                raise RuntimeError(f"Expected (T, {len(action_joint_names)}), got {action.shape}")

        except Exception as e:
            print(f"Error processing data: {e}")
            # フォールバック：現在位置 + 台車ゼロ
            action = np.array([joint_state + [0.0, 0.0, 0.0]])

        rows, cols = action.shape
        response = ResponseMsg()
        response.hz = self.traj_hz
        response.num_joints = cols
        response.joint_names = action_joint_names
        response.rows = rows
        response.cols = cols
        response.result = action.tolist()
        self._lc.publish("ACTION_RESPONSE", response.encode())


# ====== Policy Wrapper ======
class Gr00tHSRPolicy:
    """
    - グロットの get_action() はチャンク（H ステップ）を返す。
    - act() では、キューを先頭から消費しつつ、低水位で早め補充（ウォーターマーク）する。
    - EMA で出力の時間平滑（任意）。
    """

    def __init__(
        self,
        model_path: str,
        adopted_action_chunks: int = None,  # 省略時はモデルの horizon を使用
        num_traj: int = 1,                  # 1 呼び出しあたり返すステップ数（= ストライド）
        low_watermark: int = None,          # 低水位の閾値（省略時は num_traj）
        use_temp_ensem: bool = True,
        temporal_half_life: float = 24.0,   # EMA 半減期（フレーム数）
        traj_hz: float = 10.0,
        device: str = "cuda",
    ):
        # データ設定と Gr00tPolicy 構築
        data_config = DATA_CONFIG_MAP["hsr_v2"]
        self.modality_config = data_config.modality_config()
        transforms = data_config.transform()
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=self.modality_config,
            modality_transform=transforms,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            device=device,
        )

        # モデルの出力ホライゾン
        model_H = self.policy.model.action_head.action_horizon

        # 採用するチャンク長
        self.adopted_action_chunks = (
            int(adopted_action_chunks) if adopted_action_chunks is not None else int(model_H)
        )
        self.adopted_action_chunks = max(1, min(self.adopted_action_chunks, model_H))

        # 1 呼び出しあたり返すフレーム数（＝ストライド）
        self.num_traj = int(num_traj)
        assert 1 <= self.num_traj <= self.adopted_action_chunks, \
            "num_traj must be in [1, adopted_action_chunks]"

        # 低水位：この長さを下回ったら補充
        self.low_watermark = int(low_watermark) if low_watermark is not None else self.num_traj
        self.low_watermark = max(1, min(self.low_watermark, self.adopted_action_chunks))

        # 出力バッファ
        self.action_queue = {
            "action.relative": deque(maxlen=self.adopted_action_chunks)
        }

        # EMA 設定
        self.use_temp_ensem = bool(use_temp_ensem)
        self.temporal_half_life = float(temporal_half_life)
        self._ema_action = None  # 直近の出力（絶対値 11 次元）

        # 表示用など
        self.traj_hz = float(traj_hz)

        # ベース変換などのための並び
        # 出力 11 次元の並び（学習の relative の並びに合わせて後で整列）
        self.action_names_11 = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "gripper",
            "head_pan_joint",
            "head_tilt_joint",
            "base_x",
            "base_y",
            "base_t",
        ]

    # ---- internal helpers ----
    def _maybe_refill(self, obs: dict[str, Any]) -> None:
        """
        低水位なら 1 チャンク推論して、**先頭から** adopted_action_chunks までをキューに詰める。
        """
        q = self.action_queue["action.relative"]
        if len(q) >= self.low_watermark:
            return

        # ---- 入力組み立て（Gr00t の transform が面倒を見る）----
        # video.* は [B=1, ...] で渡す
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm  = np.expand_dims(obs["joint_state"][:5], axis=0)
        state_hand = np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0)
        state_head = np.expand_dims(obs["joint_state"][6:8], axis=0)
        instruction = [obs["instruction"]]

        policy_input = {
            "video.head": video_head,
            "video.hand": video_hand,
            "state.arm": state_arm,
            "state.gripper": state_hand,
            "state.head": state_head,
            "annotation.human.task_description": instruction,
        }

        print("=== Gr00tHSRPolicy: Getting action from policy (refill) ===")
        out = self.policy.get_action(policy_input)
        # out["action.relative"] は [H, D] の numpy を想定
        rel = out["action.relative"]

        # 先頭から adopted_action_chunks までをキューへ（←ココが重要）
        rel_slice = rel[: self.adopted_action_chunks]
        q.extend(rel_slice)

    def _relative_to_absolute_11d(self, action_relative: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        """
        relative から 11 次元の絶対値へ変換。
        - 関節5軸 + グリッパ + ヘッド2軸 + 台車3軸（台車は差分のまま足さずに出す場合はここで調整）
        """
        #action_relative[2] = 0.0
        #action_relative[5] = 0.0

        # 学習の relative の並びが [0..10]=[arm(5), gripper(1), head(2), base(3)] だと仮定
        a = np.concatenate(
            [
                action_relative[0:5],          # arm 5
                [action_relative[5]],          # gripper
                action_relative[6:8],          # head 2
                #[0.0, 0.0],
                action_relative[8:11],         # base 3
            ]
        )

        # 差分→絶対値（arm/head は現在角度を足す。台車はここでは 0 を足す＝速度/差分として出す）
        abs_action = a + np.concatenate(
            [
                joint_state[:5],               # arm 5
                np.array([0.0]),               # gripper はそのまま（ここでは差分として扱うなら 0）
                joint_state[6:8],              # head 2
                np.array([0.0, 0.0, 0.0]),     # base 3 は差分出力を維持
            ]
        )
        return abs_action

    def _apply_ema(self, action_abs_11d: np.ndarray) -> np.ndarray:
        if not self.use_temp_ensem:
            return action_abs_11d

        # 半減期 -> α
        alpha = 1.0 - np.exp(-np.log(2.0) / max(1e-6, self.temporal_half_life))
        if self._ema_action is None:
            self._ema_action = action_abs_11d.astype(np.float64)
        else:
            self._ema_action = alpha * action_abs_11d + (1.0 - alpha) * self._ema_action
        return self._ema_action

    # ---- public ----
    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        1 回の呼び出しで num_traj ステップ返す（= ストライド）。
        内部で低水位なら補充（推論）してから、**先頭から**消費。
        """
        # 低水位なら補充（推論はここでだけ発生）
        self._maybe_refill(obs)

        q = self.action_queue["action.relative"]
        if len(q) < self.num_traj:
            # それでも足りなければもう一度補充を試みる（初回など）
            self._maybe_refill(obs)

        # まだ足りない場合はある分だけ返す（フォールバック）
        take = min(self.num_traj, len(q)) if len(q) > 0 else 0
        if take == 0:
            # 何もない場合のフォールバック（現在姿勢＋台車停止）
            js = obs["joint_state"]
            return np.array([np.concatenate([js[:8], [0.0, 0.0, 0.0]])])

        actions_abs = []
        for _ in range(take):
            rel = q.popleft()  # 先頭から消費
            # relative(11に整列) -> absolute(11)
            action_abs = self._relative_to_absolute_11d(rel, obs["joint_state"])
            # EMA
            action_abs = self._apply_ema(action_abs)
            actions_abs.append(action_abs)

        return np.stack(actions_abs, axis=0)


# ====== main ======
def main():
    print("Start Issac-GR00T")

    # 例：chunk32 のチェックポイント
    #checkpoint_dir = "/home/veluga-g3/airoa/gr00t-chunk32-10hz_v2"
    checkpoint_dir = "/home/veluga-g3/airoa/gr00t-chunk32-5hz_v2"

    # 採用するチャンク長（省略時はモデルの H を使う）
    #adopted_action_chunks = 31
    adopted_action_chunks = 31

    # ストライド（= act() 1 回で返すフレーム数）
    num_traj = 1

    # 低水位（この長さ未満なら補充）
    low_watermark = 1

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}, num_traj: {num_traj}, low_watermark: {low_watermark}")

    policy = Gr00tHSRPolicy(
        model_path=checkpoint_dir,
        adopted_action_chunks=adopted_action_chunks,
        num_traj=num_traj,
        low_watermark=low_watermark,
        use_temp_ensem=True,
        temporal_half_life=16.0,   # 大きいほどなめらか（動き小さめ）。必要に応じて下げて調整
        traj_hz=10.0,
        device="cuda",
    )

    # --- 動作テスト（任意）
    # for _ in range(20):
    #     rand_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    #     policy_input = {
    #         "head_rgb": rand_img,
    #         "hand_rgb": rand_img,
    #         "joint_state": np.array([0.0 for _ in range(8)], dtype=np.float32),
    #         "instruction": "Test prompt. Do not move.",
    #     }
    #     action = policy.act(policy_input)
    #     print(action)

    # --- LCM サーバ起動 ---
    lcm_hsr_server = HSRLcmServer(policy)
    print("start server...")
    try:
        while True:
            lcm_hsr_server.handle()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
