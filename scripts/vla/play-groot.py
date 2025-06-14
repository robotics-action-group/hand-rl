from datetime import datetime
import time
import wandb
import argparse
from dataclasses import dataclass, field

from c3po_utils.load_yaml import load_yaml_into_obj
from c3po_utils import cli_args

import gymnasium as gym
import os
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

from data_config import NeuraDataConfig
from c3po_utils.online_data import OnlineBuffer

class Dummy:
    def get_action(self, observation):
        # Dummy action for testing purposes
        print("Dummy action taken for observation:", observation)
        # return a tensor of zeros as a dummy action
        return torch.zeros((1, 19), device="cuda" if torch.cuda.is_available() else "cpu")
    
class Inference():

    def __init__(self):
        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "groot", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        # resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        # log_dir = os.path.dirname(resume_path)
        
        data_config = NeuraDataConfig()
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        # policy: BasePolicy = Gr00tPolicy(
        #         model_path="nvidia/GR00T-N1-2B",
        #         modality_config=modality_config,
        #         modality_transform=modality_transform,
        #         embodiment_tag="new_embodiment",
        #         denoising_steps=4,
        #         device="cuda" if torch.cuda.is_available() else "cpu",
        #     )
        self.policy = Dummy()

        self.manager = OnlineBuffer(data_config)
        self.manager.initialize()
        self.manager.init_distributed("agent")
        # self.manager.loadPolicy(self.policy)
        task_strings = ["pick up and lift the cylinder on the table"]
        print("Finished loading policy, starting inference with task:", task_strings[0])
    def run(self):
        try:
            while True:
                data = self.manager.recvDictOneByOne()
                print("Received data:", data.keys())
        except KeyboardInterrupt:
            pass
        # finally:
        # observation = manager.waitForObs()
        # print("Received observation:", observation.shape)
        # action = policy.get_action(observation)
        # manager.sendAction(action)
            # self.manager.shutdown_rpc()

if __name__ == "__main__":
    # run the main function
    service = Inference()
    service.run()