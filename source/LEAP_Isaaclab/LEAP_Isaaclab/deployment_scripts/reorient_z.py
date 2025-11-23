#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
from typing import Dict
from collections import deque
import traceback
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from utils.leap_hand_utils.dynamixel_client import DynamixelClient
from utils.leap_hand_utils import leap_hand_utils as lhu
from load_model import RLGamesPolicy

class LEAPHandController:
    def __init__(self, cfg_path, ckpt_path):

        self.action_scale = 1 / 24
        self.action_type="relative"
        self.actions_num = 16
        self.hist_len = 3
        self.device = 'cuda:0'
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.hz = 30
        self.control_dt = 1 / self.hz
        
        self.kP = 800.0
        self.kI = 0.0
        self.kD = 200.0
        self.curr_lim = 500.0
        self.ema_amount = 0.2
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        
        self.motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        
        for port_num in range(3):
            try:
                port = f'/dev/ttyUSB{port_num}'
                self.dxl_client = DynamixelClient(self.motors, port, 4000000)
                self.dxl_client.connect()
                print(f"Connected to LEAP hand on {port}")
                break
            except Exception as e:
                if port_num == 2:  # Last attempt
                    raise Exception(f"Failed to connect to LEAP hand on any USB port: {e}")
                continue

        # Enables position-current control mode and the default parameters
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)  # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)  # Igain
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)  # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)  # Dgain damping for side to side should be a bit less
        # Max at current (in unit 1ma) so don't overheat and grip too hard
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        # self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()

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

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = torch.tensor([ 4,  0,  8, 12,  6,  2, 10, 14,  7,  3, 11, 15,  1,  5,  9, 13], device=self.device)
        self.real_to_sim_indices= torch.tensor([ 1, 12,  5,  9,  0, 13,  4,  8,  2, 14,  6, 10,  3, 15,  7, 11], device=self.device)

    def LEAPsim_limits(self):
        sim_min = self.sim_to_real(self.leap_dof_lower)
        sim_max = self.sim_to_real(self.leap_dof_upper)
        return sim_min, sim_max
    
    def LEAPsim_to_LEAPhand(self, joints):
        # joints = np.array(joints)
        ret_joints = joints + 3.14159
        return ret_joints

    def LEAPhand_to_LEAPsim(self, joints):
        ret_joints = joints - 3.14
        return ret_joints

    def LEAPhand_to_sim_ones(self, joints):
        joints = self.LEAPhand_to_LEAPsim(joints)
        sim_min, sim_max = self.LEAPsim_limits()
        joints = unscale_np(joints, sim_min, sim_max)
        return joints

    def get_dof_limits(self):
        self.leap_dof_lower, self.leap_dof_upper = self.get_leap_hand_joint_limits()
        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)

    def get_leap_hand_joint_limits(self):
        upper_limits = [2.2300, 2.0940, 2.2300, 2.2300, 1.0470, 2.4430, 1.0470, 1.0470, 1.8850,
                       1.9000, 1.8850, 1.8850, 2.0420, 1.8800, 2.0420, 2.0420]
        lower_limits = [-0.3140, -0.3490, -0.3140, -0.3140, -1.0470, -0.4700, -1.0470, -1.0470,
                       -0.5060, -1.2000, -0.5060, -0.5060, -0.3660, -1.3400, -0.3660, -0.3660]
        return lower_limits, upper_limits

    def fetch_grasp_state(self):
        return torch.tensor([[0.000, 0.500, 0.000, 0.000, 
                             -0.750, 1.300, 0.000, 0.750, 
                              1.750, 1.500, 1.750, 1.750, 
                              0.00, 1.0000, 0.0000, 0.00]], device=self.device)

    def command_joint_position(self, desired_pose):
        desired_pose = self.LEAPsim_to_LEAPhand(desired_pose)
        # desired_pose = (2 * desired_pose - self.leap_dof_lower - self.leap_dof_upper) / (self.leap_dof_upper - self.leap_dof_lower)
        desired_pose = self.sim_to_real(desired_pose) 
        desired_pose = desired_pose.detach().cpu().numpy().astype(float).flatten()
        
        # send command to motors
        self.dxl_client.write_desired_pos(self.motors, desired_pose)

    def poll_joint_position(self):
        # read position from hardware
        joint_position = self.dxl_client.read_pos()
        joint_position = torch.from_numpy(joint_position).to(device=self.device)
        
        joint_position = self.LEAPhand_to_sim_ones(joint_position)
        joint_position = self.real_to_sim(joint_position)
        joint_position = (self.leap_dof_upper - self.leap_dof_lower) * (joint_position + 1) / 2 + self.leap_dof_lower
        
        return {'position': joint_position}

    def deploy(self):
        print("Command to the initial position")
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
        obs_hist_buf[0, :, -1] = frame
        obs_buf = obs_hist_buf.transpose(1, 2).reshape(1, -1).float() 

        counter = 0
        print("Starting policy execution:")
        try:
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
                # self.command_joint_position(self.init_pose)

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
                    
        except KeyboardInterrupt:
            print("Stopping policy execution...")
        except Exception as e:
            print(f"Error during execution: {e}")
        finally:
            # Disable motors safely
            self.dxl_client.set_torque_enabled(self.motors, False)
            print("Motors disabled.")

    def forward_network(self, obs):
        return self.player.step(obs)["selected_action"]
    
    def restore_policy(self):
        self.player = RLGamesPolicy(
            cfg_path=self.cfg_path,
            ckpt_path=self.ckpt_path,
            num_proprio_obs=96,
            action_space=self.actions_num
        )
        self.player.reset_hidden_state()
        print("Model restored!")

    def manual_control(self, joint_positions):
        joint_positions = np.array(joint_positions)
        self.prev_pos = self.curr_pos
        self.curr_pos = joint_positions
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def read_hand_state(self):
        output = self.dxl_client.read_pos_vel_cur()
        return {
            'position': output[0].tolist(),
            'velocity': output[1].tolist(), 
            'effort': output[2].tolist()
        }
    
def unscale_np(x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

def main():
    parent_path = root_path = os.path.normpath(os.path.join(os.path.abspath(__file__),"../../")) 
    root_path = os.path.normpath(os.path.join(os.path.abspath(__file__),"../../../../../")) 

    cfg_path = f"{parent_path}/tasks/leap_hand_reorient/agents/rl_games_ppo_cfg.yaml"
    ckpt_path = f"{root_path}/logs/rl_games/leap_hand_reorient/pretrained/nn/leap_hand_reorient.pth"
    
    controller = LEAPHandController(cfg_path=cfg_path, ckpt_path=ckpt_path)
    controller.restore_policy()
    controller.deploy()

if __name__ == "__main__":
    main()