from datetime import datetime
from time import sleep
import wandb
import argparse
from dataclasses import dataclass, field


# from isaaclab.app import AppLauncher

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
# AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli=load_yaml_into_obj("scripts/sample.yaml", args_cli)

# always enable cameras to record video
# if args_cli.video:
args_cli.enable_cameras = True


# launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch

# from rsl_rl.runners import OnPolicyRunner

# from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
# from isaaclab.utils.dict import print_dict
# from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
# from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import cv2

from data_config import NeuraDataConfig
from rldata import RLDataDistributed
# Import extensions to set up environment tasks
# import c3po.tasks  # noqa: F401
import torch.distributed as dist

import sys
import torch

def pt_file_contents(file_path):
    try:
        data = torch.load(file_path, map_location='cpu')
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")

class Simulation():

    def __init__(self):
        data_config = NeuraDataConfig()
        self.manager = RLDataDistributed(name = "simulator", data_config=data_config, buffer_size=1000)
        print("buffer initialized")
        # dist.init_process_group("gloo", rank=1, world_size=2, init_method='tcp://10.5.0.2:8000')
        print("[INFO] Initialized OnlineBuffer for data management.")
        # dummy_tensor = torch.ones((1, 20))
        # dummy_tensor = dummy_tensor * 4
        # dummy_observation = {
        #     "state": {
        #         "policy": torch.ones((1, 19))
        #     },
        #     "video": {
        #         "camera": torch.ones((3, 224, 224)),
        #     }
        # }
        # dummy = {"observations": dummy_observation}
        # dist.send(tensor=dummy_tensor, dst=0)
        # dist.send_object_list([dummy], dst=0)
        # self.manager.sendObject(dummy, dst=0)
        data_dict = pt_file_contents("/workspace/c3po/obs_dict_5.pt")
        print(data_dict)
        for _ in range(15):
            processed_obs = self.manager.process_obs_from_isaac(data_dict)
            self.manager.add_to_replay_buffer(processed_obs)
        
        print("[INFO] Dummy data added to replay buffer.")
        self.manager.sendBufferToAgent()
        print("[INFO] Dummy tensor sent out")
        

    def sim_episode(self):
        # obs, obs_dict = self.env.get_observations()

        # print(f"Starting simulation episode with {self.num_steps} steps.")
        # for timestep in range(self.num_steps):
        #     print(f"Step {timestep + 1}/{self.num_steps}")
        #     self.manager.add_to_replay_buffer(obs_dict)
        #     observation = self.manager.toGR00T()
        #     action = self.manager.sendDictOneByOne(observation)
        #     obs, _, dones, obs_dict = self.env.step(action)

        #     # if episode is done, reset the environment
        #     if dones[0]:
        #         print(f"Episode {timestep} done. Resetting environment.")
        #         torch.save(obs_dict, os.path.join(self.log_root_path, f"obs_dict_{timestep}.pt"))
        #         torch.save(dones, os.path.join(self.log_root_path, f"dones_{timestep}.pt"))
        #         # obs, obs_dict = self.env.reset()
        #         self.manager.makeGif(key="eye_camera", gif_dir=os.path.join(self.log_root_path, f"gifs_{timestep}"))

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
        dummy_tensor = torch.zeros((1, 19))
        print("Starting simulation episode with dummy observation.")
        # self.manager.add_to_replay_buffer(dummy)
        # observation = self.manager.toGR00T()
        # self.manager.sendDictOneByOne(observation)
        self.manager.sendData(dummy_tensor, dst=0)
        print("Dummy observation sent to manager.")

    def session(self):
        # while True:
        try:
            # print(f"Starting simulation session with {self.num_steps} steps.") 
            self.sim_episode()
            sleep(30)
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
    # simulation.session()
    # while True:
    # close sim app
    # simulation_app.close()