import numpy as np


# Slices used by behavior-1k-rl's B1kInputs after reading raw proprioception.
# Keep these configurable from callers if a dataset revision changes the raw
# proprio layout.
DEFAULT_R1PRO_PROPRIOCEPTION_INDICES = {
    "base_qvel": list(range(0, 3)),
    "trunk_qpos": list(range(3, 7)),
    "arm_left_qpos": list(range(7, 14)),
    "gripper_left_qpos": list(range(14, 16)),
    "arm_right_qpos": list(range(16, 23)),
    "gripper_right_qpos": list(range(23, 25)),
}


def _take(values, indices):
    return values[..., indices]


def extract_state_from_proprio(proprio_data, indices=None):
    indices = indices or DEFAULT_R1PRO_PROPRIOCEPTION_INDICES
    proprio_data = np.asarray(proprio_data)

    base_qvel = _take(proprio_data, indices["base_qvel"])
    trunk_qpos = _take(proprio_data, indices["trunk_qpos"])
    arm_left_qpos = _take(proprio_data, indices["arm_left_qpos"])
    arm_right_qpos = _take(proprio_data, indices["arm_right_qpos"])

    max_gripper_width = 0.1
    left_gripper_raw = _take(proprio_data, indices["gripper_left_qpos"]).sum(axis=-1, keepdims=True)
    right_gripper_raw = _take(proprio_data, indices["gripper_right_qpos"]).sum(axis=-1, keepdims=True)
    left_gripper_width = 2.0 * (left_gripper_raw / max_gripper_width) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / max_gripper_width) - 1.0

    return np.concatenate(
        [
            base_qvel,
            trunk_qpos,
            arm_left_qpos,
            left_gripper_width,
            arm_right_qpos,
            right_gripper_width,
        ],
        axis=-1,
    )


def parse_b1k_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    elif image.ndim == 4 and image.shape[1] == 3:
        image = np.moveaxis(image, 1, -1)
    return image


class B1kInputs:
    def __init__(self, proprioception_indices=None):
        self.proprioception_indices = proprioception_indices

    def __call__(self, data):
        inputs = {
            "state": extract_state_from_proprio(
                data["observation/state"],
                indices=self.proprioception_indices,
            ),
            "image": {
                "base_0_rgb": parse_b1k_image(data["observation/egocentric_camera"]),
                "left_wrist_0_rgb": parse_b1k_image(data["observation/wrist_image_left"]),
                "right_wrist_0_rgb": parse_b1k_image(data["observation/wrist_image_right"]),
            },
        }
        for key in (
            "task_index",
            "timestamp",
            "episode_index",
            "index",
            "q_score",
            "raw_ttc_steps",
            "ep_idx",
            "frame_idx",
            "reward",
            "progress_target",
            "progress_mask",
            "progress_loss_weight",
            "progress_margin_target",
            "stage_token_index",
        ):
            if key in data:
                inputs[key] = data[key]
        return inputs
