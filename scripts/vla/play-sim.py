from datetime import datetime
import wandb
import argparse
from dataclasses import dataclass, field


from isaaclab.app import AppLauncher

# local imports
# import c3po.utils.cli_args  # isort: skip
# from c3po.utils import cli_args
from c3po_utils.load_yaml import load_yaml_into_obj
from c3po_utils import cli_args
# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli=load_yaml_into_obj("scripts/sample.yaml", args_cli)

# always enable cameras to record video
# if args_cli.video:
args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import cv2

from data_config import NeuraDataConfig
from rldata import RLData as OnlineBuffer
# Import extensions to set up environment tasks
import c3po.tasks  # noqa: F401


class Simulation():

    def __init__(self):
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        agent_cfg = load_yaml_into_obj("scripts/sample.yaml", agent_cfg)
        env_cfg = load_yaml_into_obj("scripts/sample.yaml", env_cfg)
        self.num_steps = agent_cfg.max_iterations
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.log_root_path = os.path.abspath(log_root_path)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(self.log_root_path, "overview_watch"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap around environment for rsl-rl
        self.env = RslRlVecEnvWrapper(env)
        # obs, obs_dict = self.env.get_observations()
        # for aix in range(10):
        #     action = env.action_space.sample()
        #     action = torch.from_numpy(action)
        #     obs, _, _, obs_dict = env.step(action)
        #     # save obs_dict into a object file, for each time step
        #     obs_dict_path = os.path.join(log_root_path, f"obs_dict_{aix}.pt")
        #     torch.save(obs_dict, obs_dict_path)
        #     # wait till the file is saved
        #     print(f"Saved obs_dict at {obs_dict_path}")
        #     print("saved an obs obj")
        # print(f"[INFO] Loaded environment: {args_cli.task} with {self.env.num_envs} environments.")
        data_config = NeuraDataConfig()
        self.manager = OnlineBuffer(data_config)
        self.manager.initialize()
        print("buffer initialized")
        self.manager.init_distributed("simulator")
        print("[INFO] Initialized OnlineBuffer for data management.")
        # self.manager.loadEnv(self.env)

    def sim_episode(self):
        obs, obs_dict = self.env.get_observations()

        print(f"Starting simulation episode with {self.num_steps} steps.")
        for timestep in range(self.num_steps):
            print(f"Step {timestep + 1}/{self.num_steps}")
            self.manager.add_to_replay_buffer(obs_dict)
            observation = self.manager.toGR00T()
            action = self.manager.sendDictOneByOne(observation)
            obs, _, dones, obs_dict = self.env.step(action)

            # if episode is done, reset the environment
            if dones[0]:
                print(f"Episode {timestep} done. Resetting environment.")
                torch.save(obs_dict, os.path.join(self.log_root_path, f"obs_dict_{timestep}.pt"))
                torch.save(dones, os.path.join(self.log_root_path, f"dones_{timestep}.pt"))
                # obs, obs_dict = self.env.reset()
                self.manager.makeGif(key="eye_camera", gif_dir=os.path.join(self.log_root_path, f"gifs_{timestep}"))

        # create a dummy observation for testing purposes, with 
        # dummy_observation = {
        #     "state": {
        #         "policy": torch.zeros((1, 19), device="cuda" if torch.cuda.is_available() else "cpu"),
        #     },
        #     "video": {
        #         "camera": torch.zeros((3, 224, 224), device="cuda" if torch.cuda.is_available() else "cpu"),
        #     }
        # }
        # dummy = {"observations": dummy_observation}
        print("Starting simulation episode with dummy observation.")
        self.manager.add_to_replay_buffer(dummy)
        observation = self.manager.toGR00T()
        self.manager.sendDictOneByOne(observation)
        print("Dummy observation sent to manager.")

    def session(self):
        # while True:
        try:
            # print(f"Starting simulation session with {self.num_steps} steps.") 
            self.sim_episode()
        except KeyboardInterrupt:
            print("Simulation interrupted. Closing environment.")
        # finally:
            self.close()
            # break

    def close(self):
        self.env.close()
        # self.manager.shutdown_rpc()
        # simulation_app.close()


if __name__ == "__main__":
    # run the main function
    simulation = Simulation()
    simulation.session()
    # while True:
    # close sim app
    # simulation_app.close()