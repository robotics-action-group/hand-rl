import gymnasium as gym
import os

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Raga-Franka-Push-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_push_env_cfg:FrankaCubePushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PushCubePPORunnerCfg",
    },
    disable_env_checker=True,
)