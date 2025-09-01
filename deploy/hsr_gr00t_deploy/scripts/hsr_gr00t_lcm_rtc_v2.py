#!/home/openpi/.venv/bin/python3
from collections import deque

#!/usr/bin/env python3
from typing import Any
import copy

import cv2
import numpy as np

# gr00t関連
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

from gr00t.model.policy import Gr00tPolicy

import lcm
from lcm_msgs import RequestMsg, ResponseMsg


def compressedimage_to_array_lcm(msg):
    # Convert signed bytes (-128~127) to unsigned (0~255)
    signed = np.array(msg.data[: msg.data_length], dtype=np.int8)
    unsigned = signed.astype(np.uint8)

    # TODO: 元コードにバグがあり，rgbに変換されずに使われていたのでそのままにしてある
    # 学習はバグありのままされていた？
    # そうならそのまま使って，そうでないなら以下の正しいコードを使う
    image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
    # image = cv2.imdecode(unsigned, cv2.IMREAD_COLOR)[:, :, ::-1]  # bgr -> rgb
    return image


class HSRLcmServer:
    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.9  # グリッパーを閉じる閾値

    def __init__(self, policy, traj_hz=10.0):
        self.traj_hz = float(traj_hz)

        self._lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        self._lc.subscribe("REQUEST_CHANNEL", self._handle_request)

        self.policy = policy

        self.joint_state_names: list[str] = [
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
            # self.joint_state_namesの順に並び替え
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
            #print(observation)

            action = self.policy.act(observation)

            action_joint_names = self.joint_state_names + [
                "base_x",
                "base_y",
                "base_t",
            ]

            if action.shape[1] != len(action_joint_names):
                raise RuntimeError(f"Expected (T, {len(action_joint_names)}), got {action.shape}")

        except Exception as e:
            print(f"Error processing data: {e}")
            # 現在位置 + 台車速度なし（仮）で対応
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


class Gr00tHSRPolicy:
    """
    Gr00tPolicyをHSRロボットに適用するためのクラスです.
    HSREnvからセンサ情報を取得し, policyの計算後にアクションを環境に反映させます.
    """

    def __init__(
        self,
        model_path: str = "/home/veluga-g3/airoa/gr00t-microwave", 
        adopted_action_chunks: int = 15,
        num_traj: int = 4
    ):
        assert num_traj <= adopted_action_chunks, "num_traj must be <= adopted_action_chunks"
        self.dagtconfig = data_config = DATA_CONFIG_MAP["hsr_v2"]
        self.modality_config = data_config.modality_config()
        transforms = data_config.transform()
        self.policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=self.modality_config,
            modality_transform=transforms,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,  # HSRのembodiment tag
            device="cuda",
        )
        self.adopted_action_chunks = adopted_action_chunks
        self.action_queue = {
            "action.relative": deque(maxlen=self.adopted_action_chunks),  
        }
        #print(self.action_queue)
        self.num_traj = num_traj
        rand_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        policy_input = {
            "head_rgb": rand_img,
            "hand_rgb": rand_img,
            "joint_state": np.array([0.0 for _ in range(8)]),
            "instruction": "Test prompt. Do not move.",
        }
        #self.reset_buffer()

        replay_episode = np.load("/home/veluga-g3/Downloads/episode_1511_action_relative.npz", allow_pickle=False)
        self.actions_rel = replay_episode["actions"]
        self.frame_num = 0

        self.use_temp_ensem = True          # 無効にしたい時は False
        self.temporal_half_life = 8        # フレーム半減期(=約10ステップで重み半減)
        self._ema_action = None

        self.prev_chunk_world = None     # 前チャンク（非正規化/ロボット座標系）
        self.exec_since_prev = 0        # 前チャンクのうち既に実行したステップ数 d
        self.s_min_ratio = 0.5           # 末尾自由区間の下限比
        self.mask_lambda = 0.3
        self.rtc_beta = 5.0
        self.rtc_guidance_clip = 1.0
    
    def reset_buffer(self):
        self.action_queue.clear()
        

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        # 1) 入力の整形はそのまま
        video_head = np.expand_dims(obs["head_rgb"], axis=0)
        video_hand = np.expand_dims(obs["hand_rgb"], axis=0)
        state_arm  = np.expand_dims(obs["joint_state"][:5], axis=0)
        state_hand = np.expand_dims(np.expand_dims(obs["joint_state"][5], axis=0), axis=0)
        state_head = np.expand_dims(obs["joint_state"][6:8], axis=0)
        instruction = [obs["instruction"]]
        policy_input = {
            "video.head": video_head,
            "video.hand": video_hand,
            "state.arm" : state_arm,
            "state.gripper": state_hand,
            "state.head": state_head,
            "annotation.human.task_description": instruction,
        }

        # 2) RTC パラメータ（H, D）
        H = self.policy.model.action_head.action_horizon
        D = self.policy.model.action_head.action_dim   # 32（Head内部）
        B = 1
        device = self.policy.model.device if hasattr(self.policy.model, "device") else "cuda"
        dtype  = (self.policy.model.action_head.dtype
                if hasattr(self.policy.model.action_head, "dtype") else torch.float32)

        # 3) 遅延 d と 末尾自由 s を決める
        d_steps = min(self.exec_since_prev, H-1) if (self.prev_chunk_world is not None) else 0
        s_min   = max(1, int(H * self.s_min_ratio))
        s_steps = max(d_steps, s_min)

        # 4) 時間マスク（先頭=1, 中間=exp減衰, 末尾=0）
        W_time = self.policy.build_rtc_weight_mask(
            H=H, d=d_steps, s=s_steps, lam=self.mask_lambda, B=B, D=D, device=device, dtype=dtype
        )

        # 5) RTC 入力セット（前チャンクは None ならガイダンス無し）
        rtc = {
            "rtc_prev_action": self.prev_chunk_world,   # shape [H, 11]（Policy 側で Transform を通して32に）
            "rtc_weight_mask": W_time,                  # [1,H,32] or [1,H,1]
            "rtc_beta": self.rtc_beta,
            "rtc_guidance_clip": self.rtc_guidance_clip,
            # "rtc_angle_indices": [10],  # 角度DoFがあれば
        }

        # 6) 毎サイクル再推論（RTCつき）
        action_chunk = self.policy.get_action_rtc(policy_input, rtc)

        # 7) 出力＆状態更新
        #   - action_chunk["action.relative"] は [H,11]（非正規化, 相対指令）想定
        chunk_rel = action_chunk["action.relative"]   # numpy [H,11] を想定
        # このサイクルで吐き出すステップ数 = stride
        stride = self.num_traj

        # 7a) 11次元 → ロボット absolute へ復元（今までどおり）
        out_list = []
        for t in range(stride):
            a_rel = chunk_rel[t]  # 形状 [11]
            a_11 = np.concatenate([
                a_rel[0:5],
                [a_rel[5]],
                a_rel[6:8],
                a_rel[8:11],
            ])
            a_abs = a_11 + np.concatenate([
                obs["joint_state"][:5],
                np.array([0]),
                obs["joint_state"][6:8],
                np.array([0,0,0]),
            ])

            if self.use_temp_ensem:
                alpha = 1.0 - np.exp(-np.log(2) / max(1e-6, self.temporal_half_life))
                if self._ema_action is None:
                    self._ema_action = a_abs.astype(np.float64)
                else:
                    self._ema_action = alpha * a_abs + (1.0 - alpha) * self._ema_action
                a_abs = self._ema_action
            out_list.append(a_abs)

        # 7b) 次サイクル用に「前チャンク」と「既実行カウンタ」を更新
        self.prev_chunk_world = chunk_rel.copy()  # 次回の Y として渡す（Policy 内で Transform→32次元化）
        self.exec_since_prev = stride            # 次回の d になる

        return np.stack(out_list, axis=0)



def main():
    print("Start Issac-GR00T")

    # TODO: 引数でいい感じに処理するようにする
    checkpoint_dir = "/home/veluga-g3/airoa/gr00t-chunk16-10hz"
    adopted_action_chunks = 15

    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"adopted_action_chunks: {adopted_action_chunks}")

    policy = Gr00tHSRPolicy(model_path=checkpoint_dir,adopted_action_chunks=adopted_action_chunks)

    for i in range(100):
        rand_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        policy_input = {
            "head_rgb": rand_img,
            "hand_rgb": rand_img,
            "joint_state": np.array([0.0 for _ in range(8)]),
            "instruction": "Test prompt. Do not move.",
        }
        action = policy.act(policy_input)
        print(action)

    import sys; sys.exit()
    lcm_hsr_server = HSRLcmServer(policy)

    print("start server...")
    try:
        while True:
            lcm_hsr_server.handle()
    except KeyboardInterrupt:
        pass

    return


if __name__ == "__main__":
    main()
