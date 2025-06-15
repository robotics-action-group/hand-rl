from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def distance_to_goal(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg(name="object"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg(name="goal_post_center"),
    std: float = 0.1
) -> torch.Tensor:
    """Reward based on the distance to the goal.

    Args:
        env: The environment instance.
        object_cfg: Configuration for the object being pushed.
        goal_frame_cfg: Configuration for the goal frame.
        std: Standard deviation for the reward scaling.

    Returns:
        A tensor containing reward values based on the distance to the goal.
    """
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]# Get the object and goal frame transforms

    # Combine the transforms to get the goal position in the object's frame
    cube_pos_w = object.data.root_pos_w
    goal_pos_w = goal.data.root_pos_w

    # Calculate the distance to the goal
    distance = torch.norm(cube_pos_w - goal_pos_w, dim=1)

    return 1 - torch.tanh(distance / std)  


