# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
# import rsl_rl.cli_args  # isort: skip
from scripts.rsl_rl import cli_args

from load_yaml import load_yaml_into_obj

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Gut-Velocity-Flat-G0-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
args_cli=load_yaml_into_obj("scripts/sample.yaml", args_cli)
# launch omniverse app
# print(args_cli)
# print(type(args_cli))
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import c3po.tasks  # noqa: F401

from c3po.wrappers.support import SupportWrapper

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
import wandb
from moviepy.editor import VideoFileClip

# from gymnasium.utils.save_video import save_video


# 1. Next door python class with inheritance, but solo no merging, fuck hydra
# 2. YAML hybrid file, manually update objs inside main and before app launch. But scalable, yes and yes
#['pelvis', 'left_hip_pitch_link', 'pelvis_contour_link', 'right_hip_pitch_link', 'torso_link', 'left_hip_roll_link', 'right_hip_roll_link', 'head_link', 'imu_link', 'left_shoulder_pitch_link', 'logo_link', 'right_shoulder_pitch_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_elbow_pitch_link', 'right_elbow_pitch_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_elbow_roll_link', 'right_elbow_roll_link', 'left_palm_link', 'right_palm_link', 'left_five_link', 'left_three_link', 'left_zero_link', 'right_five_link', 'right_three_link', 'right_zero_link', 'left_six_link', 'left_four_link', 'left_one_link', 'right_six_link', 'right_four_link', 'right_one_link', 'left_two_link', 'right_two_link']


# ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'torso_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_elbow_pitch_joint', 'right_elbow_pitch_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_elbow_roll_joint', 'right_elbow_roll_joint', 'left_five_joint', 'left_three_joint', 'left_zero_joint', 'right_five_joint', 'right_three_joint', 'right_zero_joint', 'left_six_joint', 'left_four_joint', 'left_one_joint', 'right_six_joint', 'right_four_joint', 'right_one_joint', 'left_two_joint', 'right_two_joint']
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    agent_cfg = load_yaml_into_obj("scripts/sample.yaml", agent_cfg)
    env_cfg = load_yaml_into_obj("scripts/sample.yaml", env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # robot =  env.unwrapped.scene.articulations["robot"]
    # # print(type(robot))
    # print(robot.data.body_names)
    # print(robot.data.joint_names)
    # print("jhfsjshfgsjhgshjfgjhfsjshgf")
    env = SupportWrapper(env, env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    # runner.writer.add_video()
    # frames = []
    #create a empty tensor to store frames
    # frames = []
    # # env.metadata["
    # with torch.inference_mode():
    #     obs, extras = env.reset()
    #     obs = runner.obs_normalizer(obs)
    #     for i in range(50):
    #         obs, rewards, dones, infos = env.step(runner.alg.act(obs, obs))
    #         obs = runner.obs_normalizer(obs)
    #         # frames.append(env.unwrapped.render())
    #         # print(infos)
    #         frame = env.unwrapped.render()
    #         print(frame.shape)
    #         # add frame to the tensor
    #         frames.append(frame)

    # convert the list of frames to a tensor
    # out = torch.tensor(frames)
    
    # print(out.shape)
    # runner.writer.add_video("train", frame, fps=30, global_step=runner.current_learning_iteration)
    # close the simulator

    if not args_cli.video:
        print("[INFO] Videos not recorded, skipping video logging.")
        if hasattr(runner.writer, "stop"):
            runner.writer.stop()
        env.close()
        return
    video_folder = video_kwargs["video_folder"]
    # iterate through the video folder and get all the video files
    video_files = []
    #wait till videos are saved
    imgio_kargs = {'fps': 1, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    step_fake = 0
    # for root, dirs, files in os.walk(video_folder):
    #     for file in files:
    #         # print(file)
    #         if file.endswith(".mp4"):
    #             video_files.append(os.path.join(root, file))
    #             step = int(file.split("-")[-1].split(".")[0])
    #             step_fake = step_fake + 1
    #             # wait till file exists
    #             # TODO: add a timeout
    #             while not os.path.exists(os.path.join(root, file)):
    #                 pass
    #                 # wait for 1 minute and then exit, skipping the video
    #             # log the video to wandb
    #             # print(os.path.join(root, file))
                
    #             video_path = os.path.join(root, file)
    #             # videoClip = VideoFileClip(video_path)
                
    #             # videoClip.write_gif(os.path.join(root, file.replace(".mp4", ".gif")))
    #             # gif_path = os.path.join(root, file.replace(".mp4", ".gif"))
    #             wandb.log({"Video": wandb.Video(video_path, format="mp4")})
    #             print(f"[INFO] Video saved at: {os.path.join(root, file)} with step {step_fake} but the original step is {step}")
    #         # if file.endswith(".json"):
    #         #     # log the json file to wandb
    #         #     json_path = os.path.join(root, file)
    #         #     wandb.log
    # from c3po.utils.wandb_video import check_videos_are_uploaded
    # check_videos_are_uploaded(log_dir = log_dir, step=agent_cfg.max_iterations)
    runner.writer.stop()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
