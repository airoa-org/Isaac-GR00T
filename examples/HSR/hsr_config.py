from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


so100_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["head", "hand"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "arm",
            "gripper",
            "head",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ],
        modality_keys=[
            "relative",
        ],
        # action_configs=[
        #     ActionConfig(
        #         rep=ActionRepresentation.ABSOLUTE,
        #         type=ActionType.NON_EEF,
        #         format=ActionFormat.DEFAULT,
        #     ),
        # ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(so100_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
