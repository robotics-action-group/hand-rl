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

# from gr00t.data.dataset import LeRobotSingleDataset
# from gr00t.eval.robot import RobotInferenceClient
# from gr00t.experiment.data_config import DATA_CONFIG_MAP
# from gr00t.model.policy import BasePolicy, Gr00tPolicy
# from gr00t.utils.eval import calc_mse_for_single_trajectory

from data_config import NeuraDataConfig
from rldata import RLDataDistributed
import torch.distributed as dist
# Import extensions to set up environment tasks

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
        self.policy = Dummy()
        data_config = NeuraDataConfig()
        self.manager = RLDataDistributed(name="agent", data_config=data_config, buffer_size=1000)
        # dist.init_process_group("gloo", rank=0, world_size=2, init_method='tcp://10.5.0.2:8000')
        # self.manager.loadPolicy(self.policy)
        task_strings = ["pick up and lift the cylinder on the table"]
        print("Finished loading policy, starting inference with task")
        data = torch.zeros((1, 20))
        dummy_observation = {
            "state": {
                "policy": None
            },
            "video": {
                "camera": torch.zeros((3, 224, 224)),
            }
        }
        dummy = None
        # out = dist.recv(tensor=data, src=1)
        dump = [None]
        # out = dist.recv_object_list(dump, src=1)
        # dummy = self.manager.recvObject(src=1)
        self.manager.recvBufferFromSim()
        print("Received data")
        # dummy = dump[0]
        dummy = self.manager.replay_buffer 
        print("Received data:", dummy["state"]["policy"].shape)
        print(dummy["state"]["policy"])
        print("Received data:", dummy["video"]["camera"].shape)
        print(dummy["video"]["camera"])
        # print(ou;t)

    def run(self):
        try:
            while True:
                data = torch.zeros((1, 20))
                data = self.manager.recvData(src=1, trgt_tensor=data)
                print("Received data:", data.shape)
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
    # service.run()