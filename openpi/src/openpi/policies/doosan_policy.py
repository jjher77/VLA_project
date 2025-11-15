import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_doosan_example() -> dict:
    """Creates a random input example for the Doosan policy bridge."""
    return {
        "observation/top_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/front_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "청소해 줘",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DoosanInputs(transforms.DataTransformFn):
    """Packs Doosan ROS bridge data into the Observation schema expected by OpenPI."""

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        joint_pos = np.asarray(data["observation/joint_position"])
        gripper = np.asarray(data.get("observation/gripper_position", []))
        if gripper.ndim == 0 and gripper.size > 0:
            gripper = gripper[np.newaxis]
        state = np.concatenate([joint_pos, gripper]) if gripper.size else joint_pos

        top_image = _parse_image(data["observation/top_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        front_image = _parse_image(data["observation/front_image"])

        images = {
            "base_0_rgb": top_image,
            "left_wrist_0_rgb": wrist_image,
            "right_wrist_0_rgb": front_image,
        }
        image_masks = {k: np.True_ for k in images}

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DoosanOutputs(transforms.DataTransformFn):
    """Returns only the first N dims (default: 6 joints) from policy outputs."""

    dims: int = 6

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        actions = actions[..., : self.dims]
        return {"actions": actions}
