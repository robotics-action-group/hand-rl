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

# wandb_project = load_yaml_into_obj("scripts/sample.yaml", "wandb_project")
# # Initialize a run
# entity = "gg-robotics"
# run_id = load_yaml_into_obj("scripts/sample.yaml", "run_id")


# api = wandb.Api()
# run = api.run(f"{entity}/{wandb_project}/{run_id}")

# highest_number = 0
# for file in run.files():
#     # select the file with model_number.pt, where number is the highest.
#     if file.name.endswith(".pt") and "model" in file.name:
#         # check if the number is the highest
#         number = int(file.name.split("_")[-1].split(".")[0])
#         # check if the number is higher than the current highest number
#         if number > highest_number:
#             highest_number = number
#             latest_file_name = file.name

# import os
# model_path = run.file(latest_file_name).download(replace=True)
# while not os.path.exists(latest_file_name):
#     pass
    
# # run.file.download
# print(model_path)
# model_path = latest_file_name

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.factory import make_policy
from lerobot.scripts.eval import *
from huggingface_hub import hf_hub_download
from lerobot.configs.types import *
from lerobot.common.constants import *
import yaml

import cv2

# Import extensions to set up environment tasks
import c3po.tasks  # noqa: F401

class FuckingConfig(PI0Config):
    type = "pi0"
    pretrained_path = "lerobot/pi0"

class StateFeature(PolicyFeature):
    type: FeatureType = FeatureType.STATE
    shape: tuple = (105,)

class ImageFeature(PolicyFeature):
    type: FeatureType = FeatureType.VISUAL
    shape: tuple = (256, 256, 3)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg = load_yaml_into_obj("scripts/sample.yaml", agent_cfg)
    env_cfg = load_yaml_into_obj("scripts/sample.yaml", env_cfg)
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    # log_dir = os.path.dirname(resume_path)

    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, "inferr_play"),
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
    env = RslRlVecEnvWrapper(env)

    # print(f"[INFO]: Loading model checkpoint from: {model_path}")
    # load previously trained model
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner.load(model_path)

    # # obtain the trained policy for inference
    # policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # --- PATCH: Load config without 'type' field ---
    # config_path = hf_hub_download(repo_id="lerobot/pi0", filename="config.yaml")
    # with open(config_path, "r") as f:
    #     config_dict = yaml.safe_load(f)
    # config_dict.pop("type", None)  # Remove 'type' if present
    # config = PI0Config(**config_dict)
    # --- END PATCH ---

    policy = PI0Policy.from_pretrained("lerobot/pi0")
    # policy.config.features
    # config = PI0Config()
    # config.type = "pi0"
    # state = StateFeature()
    # image_feature = ImageFeature()
    policy.config.input_features = {
        "policy": PolicyFeature(type=FeatureType.STATE, shape=(19, )),
        "camera": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
        "table_camera": PolicyFeature(type=FeatureType.VISUAL, shape=(256, 256, 3)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(19, 1)),
    }
    # policy.config.max_state_dim = 105
    env_cfg.features_map = {
        "camera": OBS_IMAGE,
        "policy": OBS_ROBOT,
        "action": ACTION
   }
    
    # policy = make_policy(cfg=FuckingConfig,env_cfg=env_cfg)
    # policy.eval()

    device = args_cli.device
    # cfg = EvalPipelineConfig()
    # eval_main()

    # reset environment
    obs, obs_dict = env.get_observations()
    action = env.action_space.sample()
    action = torch.from_numpy(action)
    obs, _, _, obs_dict = env.step(action)

    # FIXME also good to have image as input
    timestep = 0

    # Create a directory for the video if it doesn't exist
    video_dir = os.path.join(log_root_path, "wrist_infer_play")
    os.makedirs(video_dir, exist_ok=True)
    video_dir_2 = os.path.join(log_root_path, "table_infer_play")
    os.makedirs(video_dir_2, exist_ok=True)
    # video_path = os.path.join(video_dir, "env_video.mp4")

    while simulation_app.is_running():
        # run everything in inference mode
        state = obs
        image = obs_dict["observations"]["camera"] # image shape: torch.Size([1, 80, 80, 3])
        image_table = obs_dict["observations"]["table_camera"] 
        # save the image in the video_dir
        # if timestep % 50 == 0:
            # cv2.imwrite(os.path.join(video_dir, f"image_{timestep}.png"), image[0].cpu().numpy())

        # Remove the first dimension of the image tensor
        image = image.squeeze(0)
        image_table = image_table.squeeze(0)

        # # Initialize video writer only once
        if timestep % 50 == 0:
        #     height, width, _ = image.shape
        #     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #     video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

        # # Convert image tensor to numpy array and BGR format for OpenCV
            frame = (image.cpu().numpy() * 255).astype('uint8')
            frame_table = (image_table.cpu().numpy() * 255).astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_table = cv2.cvtColor(frame_table, cv2.COLOR_RGB2BGR)
        # video_writer.write(frame)

        # #save the image
            image_path = os.path.join(video_dir, f"image_{timestep}.png")
            cv2.imwrite(image_path, frame)

            image_path_table = os.path.join(video_dir_2, f"image_{timestep}.png")
            cv2.imwrite(image_path_table, frame_table)

        # # Release video writer at the end of recording
        # if timestep + 1 == 150:
        #     video_writer.release()

        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        # print(f"image shape: {image.shape}")
        image = image.permute(2, 0, 1)

        # Send data tensors from CPU to GPU
        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        # Add extra (empty) batch dimension, required to forward the policy
        # state = state.unsqueeze(0)
        image = image.unsqueeze(0)

        task_strings = ["pick up and lift the cylinder on the table"]

        # print(f"state shape: {state.shape}")
        # print(f"image shape: {image.shape}")
        observation = {
            "observation.state": state,
            "camera": image,
            "task": task_strings,
        }
        

        with torch.inference_mode():
            action = policy.select_action(observation)
            # action = env.unwrapped.action_manager.sample_action()
            # action = env.action_space.sample()
            # convert action to torch tensor
            # action = torch.from_numpy(action)
            # env stepping
            obs, _, _, obs_dict = env.step(action)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
    # video_writer.release()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()