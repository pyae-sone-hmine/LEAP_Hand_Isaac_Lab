#!/usr/bin/env python3

import time
import torch
import xml.etree.ElementTree as ET
import os
import pathlib
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from collections import deque
import math
from pathlib import Path
import random
import yaml
import argparse

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from leap_hand.srv import LeapPosition, LeapPosVelEff
import threading

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder

from load_model import RLGamesPolicy
import pickle

class HardwarePlayer(Node):
    def __init__(self, cfg_path, ckpt_path):
        super().__init__('hardware_player_node')
        
        self.action_scale = 1 / 24
        self.action_type="relative"
        self.actions_num = 16
        self.num_proprio_obs = 16
        self.hist_len = 3
        self.device = 'cuda:0'
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.hz = 30
        self.control_dt = 1 / self.hz

        self.store_cur_actions = True
        if self.store_cur_actions:
            self.num_proprio_obs = 32 #16 for hand position, 16 for action target

        # hand setting
        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()
        self.setup_ros_communication()

    def setup_ros_communication(self):
        self.pub_hand = self.create_publisher(JointState, '/cmd_ones', 10)
        self.cli = self.create_client(LeapPosVelEff, '/leap_pos_vel_eff')
        
        # Wait for service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        self.req = LeapPosVelEff.Request()
        self.get_logger().info('ROS communication setup complete')

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()
        if values.dim() == 1:
            return values[self.real_to_sim_indices]
        else:
            return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        if values.dim() == 1:
            return values[self.sim_to_real_indices]
        else:
            return values[:, self.sim_to_real_indices]

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        
        return sim_min, sim_max

    def LEAPhand_to_LEAPsim(self, joints):
        ret_joints = joints - 3.14
        return ret_joints

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)
        
        return joints

    def construct_sim_to_real_transformation(self):
        #received from sim, run self.sim_real_indices() in the sim env file to figure out indices
        self.sim_to_real_indices = torch.tensor([ 4,  0,  8, 12,  6,  2, 10, 14,  7,  3, 11, 15,  1,  5,  9, 13], device=self.device)
        self.real_to_sim_indices = torch.tensor([ 1, 12,  5,  9,  0, 13,  4,  8,  2, 14,  6, 10,  3, 15,  7, 11], device=self.device)

    def get_dof_limits(self):
        self.leap_dof_lower, self.leap_dof_upper = self.get_leap_hand_joint_limits()
        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)

    def get_leap_hand_joint_limits(self):

        # received from sim
        upper_limits = [2.2300, 2.0940, 2.2300, 2.2300, 1.0470, 2.4430, 1.0470, 1.0470, 1.8850,
        1.9000, 1.8850, 1.8850, 2.0420, 1.8800, 2.0420, 2.0420]
        lower_limits = [-0.3140, -0.3490, -0.3140, -0.3140, -1.0470, -0.4700, -1.0470, -1.0470,
        -0.5060, -1.2000, -0.5060, -0.5060, -0.3660, -1.3400, -0.3660, -0.3660]
        
        return lower_limits, upper_limits

    def fetch_grasp_state(self, s=1.0):
        return torch.tensor([[0.000, 0.500, 0.000, 0.000, 
                             -0.750, 1.300, 0.000, 0.750, 
                              1.750, 1.500, 1.750, 1.750, 
                              0.00, 1.0000, 0.0000, 0.00]], device=self.device)

    def command_joint_position(self, desired_pose):
        desired_pose = (2 * desired_pose - self.leap_dof_lower - self.leap_dof_upper) / (self.leap_dof_upper - self.leap_dof_lower)
        desired_pose = self.sim_to_real(desired_pose) 
        desired_pose = desired_pose.detach().cpu().numpy().astype(float).flatten().tolist()

        # Create and publish JointState message
        joint_msg = JointState()
        joint_msg.position = desired_pose
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        
        self.pub_hand.publish(joint_msg)

    def poll_joint_position(self):

        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            joint_position = np.array(response.position)
            joint_position = torch.from_numpy(joint_position).to(device=self.device)

            joint_position = self.LEAPhand_to_sim_ones(joint_position)
            joint_position = self.real_to_sim(joint_position)
            joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower

            return {'position': joint_position}
        else:
            self.get_logger().error('Service call failed')
            return {'position': torch.zeros(16, device=self.device)}

    def deploy(self):       
        print("command to the initial position")
        for _ in range(self.hz * 4):
            self.command_joint_position(self.init_pose)
            robot_state = self.poll_joint_position()
            obses = robot_state['position']
            time.sleep(self.control_dt)
        print("Initial position reached!")
       
        # Get current state
        self.command_joint_position(self.init_pose)
        robot_state = self.poll_joint_position()
        obses = robot_state['position']

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)
        
        obs_hist_buf = torch.zeros((1, 32, self.hist_len), device=self.device, dtype=torch.double)
        prev_target = obses.clone()
        
        unscaled_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)
        frame = torch.cat([unscaled_pos, prev_target], dim=-1).double()
        
        # Fill history buffer 
        for i in range(self.hist_len):
            obs_hist_buf[0, :, i] = frame
        obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float()

        counter = 0
        print("Starting policy execution:")
        while True:
            counter += 1
            start_time = time.time()
            
            # Get action from policy
            action = self.forward_network(obs_buf)
            action = action.squeeze(0)

            if self.action_type=="relative":
                action = torch.clamp(action, -1.0, 1.0)
                target = prev_target + self.action_scale * action
            elif self.action_type=="absolute":
                action = unscale(action, self.leap_dof_lower, self.leap_dof_upper)
                target = self.action_scale * action + (1.0 - self.action_scale) * prev_target 
            else:
                raise ValueError(f"Unsupported action type: {self.action_type}. Must be relative or absolute.")

            target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            prev_target = target.clone()
        
            print(f"Sending command: {target}")
            self.command_joint_position(target)
            
            robot_state = self.poll_joint_position()
            print(f"Received state: {robot_state['position']}")

            obses = robot_state['position']
            unscaled_pos = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)

            frame = torch.cat([unscaled_pos, target], dim=-1).double()
            obs_hist_buf[:, :, :-1] = obs_hist_buf[:, :, 1:]
            obs_hist_buf[:, :, -1] = frame
            obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float()

            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.control_dt - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def forward_network(self, obs):
        return self.player.step(obs)["selected_action"]

    def restore(self):
        self.player = RLGamesPolicy(
            cfg_path=self.cfg_path,
            ckpt_path=self.ckpt_path,
            num_proprio_obs=self.num_proprio_obs*self.hist_len,
            action_space=self.actions_num
        )
        self.player.reset_hidden_state()
        print("Model restored!")

def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower)/(upper - lower)

def main():
    rclpy.init()
    
    parent_path = root_path = os.path.normpath(os.path.join(os.path.abspath(__file__),"../../")) 
    root_path = os.path.normpath(os.path.join(os.path.abspath(__file__),"../../../../../")) 

    cfg_path = f"{parent_path}/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
    ckpt_path = f"{root_path}/logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth"
    
    agent = HardwarePlayer(cfg_path=cfg_path, ckpt_path=ckpt_path)
    agent.restore()
    
    try:
        agent.deploy()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__=="__main__":
    main()