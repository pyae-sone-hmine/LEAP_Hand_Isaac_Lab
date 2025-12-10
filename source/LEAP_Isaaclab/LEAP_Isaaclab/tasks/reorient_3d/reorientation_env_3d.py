from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import sys
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    saturate,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
)
import time

# Try to import random_orientation, provide fallback if not available
try:
    from isaaclab.utils.math import random_orientation
except ImportError:
    def random_orientation(num: int, device: torch.device) -> torch.Tensor:
        """Generate random unit quaternions using the standard method.
        
        Returns quaternions in (w, x, y, z) format.
        """
        # Sample from standard normal distribution
        q = torch.randn(num, 4, device=device)
        # Normalize to unit quaternions
        q = q / torch.norm(q, dim=-1, keepdim=True)
        # Ensure w is positive (canonical form)
        q = torch.where(q[:, 0:1] < 0, -q, q)
        return q

if TYPE_CHECKING:
    from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_env_cfg_3d import LeapHandEnvCfg3D

from LEAP_Isaaclab.utils import adr_utils, obs_utils
from LEAP_Isaaclab.utils.adr import LeapHandADR


class ReorientationEnv3D(DirectRLEnv):
    """Full 3D in-hand cube reorientation environment."""
    
    cfg: LeapHandEnvCfg3D

    def __init__(self, cfg: LeapHandEnvCfg3D, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = [self.hand.joint_names.index(j) for j in self.cfg.actuated_joint_names]
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets (used only inside reward for success counting / logging)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Multi-goal per episode tracking
        self.has_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.goals_completed_this_episode = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Checkpoint reward tracking for 3D
        # Instead of tracking Z-angle progress, we track rotation distance reduction
        self.initial_rot_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.num_checkpoints = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.checkpoints_reached = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.prev_rot_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # used to compare object position (keep cube in hand)
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] += 0.01

        # default goal positions and rotations
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0  # Identity quaternion
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)

        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes for logging / ADR
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.override_default_joint_pos = torch.tensor(
            [[
                0.000, 0.500, 0.000, 0.000,
               -0.750, 1.300, 0.000, 0.750,
                1.750, 1.500, 1.750, 1.750,
                0.000, 1.000, 0.000, 0.000
            ]],
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_linvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_angvel = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_rot[:, 0] = 1.0

        # initialize history tensor
        self.obs_hist_buf = torch.zeros(
            (self.num_envs, self.cfg.observation_space // self.cfg.hist_len, self.cfg.hist_len),
            device=self.device,
            dtype=torch.double,
        )
        self.output_obs_hist_buf = torch.zeros(
            self.cfg.scene.num_envs,
            self.cfg.observation_space // self.cfg.hist_len,
            self.cfg.hist_len,
            device=self.cfg.sim.device,
            dtype=torch.double,
        )

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        min_steps, max_steps = self._episode_length_bounds()
        self._max_episode_length_steps = max_steps
        self.randomized_episode_lengths = torch.randint(
            min_steps,
            max_steps + 1,
            (self.num_envs,),
            dtype=torch.int32,
            device=self.device,
        )

        # set adr up
        if self.cfg.enable_adr:
            self.leap_adr = LeapHandADR(
                self.event_manager,
                self.cfg.adr_cfg_dict,
                self.cfg.adr_custom_cfg_dict,
            )
            self.step_since_last_dr_change = 0
            self.leap_adr.set_num_increments(self.cfg.starting_adr_increments)
            adr_utils.init_adr_obs_act_noise(self)

            self.obs_hist_buf = torch.zeros(
                self.num_envs,
                self.cfg.observation_space // self.cfg.hist_len,
                self.cfg.hist_len + self.cfg.obs_max_latency,
                device=cfg.sim.device,
                dtype=torch.float,
            )
            self.obs_latency = torch.empty((self.num_envs, self.cfg.obs_per_timestep), device=self.cfg.sim.device)
            self.act_latency = torch.empty((self.num_envs, self.cfg.action_space), device=self.cfg.sim.device)
            self.act_hist_buf = torch.zeros(
                self.num_envs,
                self.cfg.action_space,
                self.cfg.act_max_latency + 1,
                device=self.cfg.sim.device,
                dtype=torch.float,
            )

            print("starting ranges: ")
            print(self.leap_adr.print_params())

        # Initialize extras if not already present
        if not hasattr(self, "extras") or self.extras is None:
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}

        self.sim_real_indices()

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        if self.cfg.enable_adr:
            hand_noise = self.leap_adr.get_custom_param_value("robot_action_noise", "hand_noise")
            if hand_noise > 0:
                noise = torch.randn_like(actions) * hand_noise
                self.actions = actions + noise
            self.actions = obs_utils.create_action_latency(self, self.actions)

        self.actions = torch.clamp(self.actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # Clone to avoid modifying original actions
        actions = self.actions.clone()

        if self.cfg.action_type == "relative":
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.cfg.act_moving_average * actions
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                targets,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        elif self.cfg.action_type == "absolute":
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                actions,
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = saturate(
                self.cur_targets[:, self.actuated_dof_indices],
                self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                self.hand_dof_upper_limits[:, self.actuated_dof_indices],
            )
        else:
            raise ValueError(f"Unsupported action type: {self.cfg.action_type}. Must be relative or absolute.")

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        if self.cfg.enable_adr:
            adr_utils.apply_object_wrench(self, self.object, "object")

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    def _get_observations(self) -> dict:
        """
        Observation per timestep includes:
        - normalized hand DOF positions
        - current joint targets
        - object world position (env-origin offset)
        - object orientation (quat)
        - goal orientation (quat)
        """
        # normalize joint positions to [-1, 1]
        frame = unscale(
            self.hand_dof_pos,
            self.hand_dof_lower_limits,
            self.hand_dof_upper_limits,
        )

        per_timestep_features = [frame]

        if self.cfg.store_cur_actions:
            per_timestep_features.append(self.cur_targets[:])

        # object world position (relative to env origin)
        per_timestep_features.append(self.object_pos)

        # condition on current object orientation (w, x, y, z)
        per_timestep_features.append(self.object_rot)

        # condition on goal orientation
        per_timestep_features.append(self.goal_rot)

        frame = torch.cat(per_timestep_features, dim=-1)

        # history buffer
        self.obs_hist_buf[:, :, :-1] = self.obs_hist_buf[:, :, 1:]
        self.obs_hist_buf[:, :, -1] = frame

        obs = self.obs_hist_buf.transpose(1, 2).reshape(self.num_envs, -1)
        return {"policy": obs.float()}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for 3D reorientation with checkpoint bonuses."""
        
        # Compute penalties
        pose_diff_penalty = ((self.cur_targets[:, self.actuated_dof_indices] - self.override_default_joint_pos) ** 2).sum(-1)
        torque_penalty = (self.hand.data.computed_torque ** 2).sum(-1)

        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
            new_success,
            rot_dist,
            rot_progress,
            rot_regress_penalty,
            align_reward,
            off_axis_penalty,
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.has_succeeded,
            self._max_episode_length_steps,
            self.fingertip_pos,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.object_linvel,
            self.object_angvel,
            self.actions,
            pose_diff_penalty,
            torque_penalty,
            self.prev_rot_dist,
            # Reward scales
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_exp_decay_scale,
            self.cfg.rot_linear_penalty_scale,
            self.cfg.rot_progress_scale,
            self.cfg.rot_regress_penalty_scale,
            self.cfg.rot_eps,
            self.cfg.action_penalty_scale,
            self.cfg.pose_diff_penalty_scale,
            self.cfg.torque_penalty_scale,
            self.cfg.angvel_penalty_scale,
            self.cfg.align_angvel_scale,
            self.cfg.off_axis_penalty_scale,
            self.cfg.align_gate_dist,
            self.cfg.align_angvel_cap,
            # Success/failure params
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.success_tolerance,
            self.cfg.av_factor,
            self.cfg.fingertip_dist_penalty_scale,
            self.cfg.centering_sigma,
            self.cfg.centering_reward_scale,
        )

        # === 3D CHECKPOINT BONUS LOGIC ===
        # Progress is measured by how much we've reduced the rotation distance
        # from initial_rot_dist toward 0 (the goal)
        progress = self.initial_rot_dist - rot_dist
        progress = torch.clamp(progress, min=0)  # Only count forward progress
        
        # Number of checkpoints reached = floor(progress / checkpoint_step)
        checkpoints_now = torch.floor(progress / self.cfg.checkpoint_step_rad).to(torch.int32)
        checkpoints_now = torch.clamp(checkpoints_now, min=0)
        checkpoints_now = torch.minimum(checkpoints_now, self.num_checkpoints - 1)
        
        # Calculate newly crossed checkpoints
        new_checkpoints = checkpoints_now - self.checkpoints_reached
        new_checkpoints = torch.clamp(new_checkpoints, min=0)
        
        # Apply checkpoint bonuses
        progress_fraction = torch.where(
            self.initial_rot_dist > 1e-6,
            progress / torch.clamp(self.initial_rot_dist, min=1e-6),
            torch.zeros_like(progress),
        )
        progress_fraction = torch.clamp(progress_fraction, 0.0, 1.0)
        checkpoint_reward = new_checkpoints.float() * self.cfg.checkpoint_bonus * progress_fraction
        total_reward = total_reward + checkpoint_reward
        
        # Update checkpoints reached
        self.checkpoints_reached = torch.maximum(self.checkpoints_reached, checkpoints_now)
        
        # === FINAL GOAL BONUS ===
        total_reward = torch.where(new_success, total_reward + self.cfg.reach_goal_bonus, total_reward)

        # Update has_succeeded flag when new success occurs
        self.has_succeeded = torch.logical_or(self.has_succeeded, new_success)

        # Goal switching logic
        self._update_goals(new_success)

        # Logging
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["goals_per_episode"] = self.goals_completed_this_episode.float().mean()
        self.extras["log"]["avg_hold_remaining"] = self.hold_counter.float().mean()
        self.extras["log"]["rot_dist"] = rot_dist.mean()
        self.extras["log"]["rot_progress"] = rot_progress.mean()
        self.extras["log"]["rot_regress_penalty"] = rot_regress_penalty.mean()
        self.extras["log"]["align_angvel_reward"] = align_reward.mean()
        self.extras["log"]["off_axis_penalty"] = off_axis_penalty.mean()
        self.extras["log"]["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["log"]["torque_info"] = torque_penalty.mean()
        self.extras["log"]['object_linvel'] = torch.norm(self.object_linvel, p=1, dim=-1).mean()
        self.extras["log"]['roll'] = self.object_angvel[:, 0].mean()
        self.extras["log"]['pitch'] = self.object_angvel[:, 1].mean()
        self.extras["log"]['yaw'] = self.object_angvel[:, 2].mean()
        self.extras["log"]["checkpoints_reached"] = self.checkpoints_reached.float().mean()
        self.extras["log"]["checkpoint_reward"] = checkpoint_reward.mean()
        self.extras["log"]["initial_rot_dist"] = self.initial_rot_dist.mean()
        self.prev_rot_dist = rot_dist.detach()

        # Log episode length statistics
        self.extras["log"]["avg_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).mean()
        self.extras["log"]["min_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).min()
        self.extras["log"]["max_episode_length_s"] = (
            self.randomized_episode_lengths.float() * self.cfg.sim.dt * self.cfg.decimation
        ).max()

        if self.cfg.enable_adr:
            adr_criteria = (
                self.consecutive_successes.float().mean()
                / (self.randomized_episode_lengths.float().mean() * self.cfg.sim.dt * self.cfg.decimation)
            ).float().mean()
            self.extras["log"]["adr_criteria"] = adr_criteria

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.randomized_episode_lengths - 1

        # For 3D reorientation, we don't terminate on "flipped" cube
        # since any orientation is a valid goal

        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        assert env_ids is not None

        if self.cfg.enable_adr:
            adr_criteria = (
                self.consecutive_successes.float().mean()
                / (self.randomized_episode_lengths.float().mean() * self.cfg.sim.dt * self.cfg.decimation)
            ).float().mean()

        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        min_steps, max_steps = self._episode_length_bounds()
        self._max_episode_length_steps = max_steps
        self.randomized_episode_lengths[env_ids] = torch.randint(
            min_steps,
            max_steps + 1,
            (len(env_ids),),
            dtype=torch.int32,
            device=self.device,
        )

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        dof_pos = self.override_default_joint_pos[env_ids]
        dof_vel = self.hand.data.default_joint_vel[env_ids]

        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])

        if self.cfg.enable_adr:
            x_width = self.leap_adr.get_custom_param_value("object_spawn", "x_width_spawn")
            y_width = self.leap_adr.get_custom_param_value("object_spawn", "y_width_spawn")
            x_rot = self.leap_adr.get_custom_param_value("object_spawn", "x_rotation")
            y_rot = self.leap_adr.get_custom_param_value("object_spawn", "y_rotation")
            z_rot = self.leap_adr.get_custom_param_value("object_spawn", "z_rotation")

            # Apply randomization
            if x_width > 0 or y_width > 0:
                pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), self.device)
                object_default_state[:, 0] += pos_noise[:, 0] * x_width
                object_default_state[:, 1] += pos_noise[:, 1] * y_width

            if x_rot > 0:
                x_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), self.device)
                x_rot_quat = quat_from_angle_axis(x_rot_noise * x_rot, self.x_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(x_rot_quat, object_default_state[:, 3:7])

            if y_rot > 0:
                y_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), self.device)
                y_rot_quat = quat_from_angle_axis(y_rot_noise * y_rot, self.y_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(y_rot_quat, object_default_state[:, 3:7])

            if z_rot > 0:
                z_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids),), self.device)
                z_rot_quat = quat_from_angle_axis(z_rot_noise * z_rot, self.z_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(z_rot_quat, object_default_state[:, 3:7])

            joint_pos_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_pos_noise")
            joint_vel_noise_width = self.leap_adr.get_custom_param_value("robot_spawn", "joint_vel_noise")

            if joint_pos_noise_width > 0:
                joint_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), self.device)
                dof_pos += joint_pos_noise * joint_pos_noise_width

            if joint_vel_noise_width > 0:
                joint_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), self.device)
                dof_vel += joint_vel_noise * joint_vel_noise_width

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # reset hand
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.successes[env_ids] = 0
        self.reset_goal_buf[env_ids] = False

        # Reset multi-goal tracking
        self.has_succeeded[env_ids] = False
        self.hold_counter[env_ids] = 0
        self.goals_completed_this_episode[env_ids] = 0

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        if self.cfg.enable_adr and len(env_ids) > 0:
            adr_utils.update_adr_obs_act_noise(self, env_ids)

            obs_latency_resets = self.leap_adr.get_custom_param_value("obs_latency", "latency") - torch.randint(
                0, self.cfg.obs_latency_rand + 1, (len(env_ids), 1), device=self.cfg.sim.device
            )
            obs_latency_resets = torch.maximum(obs_latency_resets, torch.tensor(0))
            self.obs_latency[env_ids, :] = obs_latency_resets.expand(-1, self.cfg.obs_per_timestep)

            act_latency_resets = self.leap_adr.get_custom_param_value("action_latency", "hand_latency") - torch.randint(
                0, self.cfg.act_latency_rand + 1, (len(env_ids), 1), device=self.cfg.sim.device
            )
            act_latency_resets = torch.maximum(act_latency_resets, torch.tensor(0))
            self.act_latency[env_ids, :] = act_latency_resets.expand(-1, self.cfg.action_space)

            self.extras["log"]["num_adr_increases"] = self.leap_adr.num_increments()

            if (
                self.step_since_last_dr_change >= self.cfg.min_steps_for_dr_change
                and (adr_criteria >= self.cfg.min_rot_adr_coeff)
            ):
                self.step_since_last_dr_change = 0
                self.leap_adr.increase_ranges()
                self.leap_adr.print_params()
                self.consecutive_successes.fill_(0.0)
            else:
                self.step_since_last_dr_change += 1

            # update whether to apply wrench for the episode
            self.object_mass = self.object.root_physx_view.get_masses().to(device=self.device)
            self.apply_wrench = torch.where(
                torch.rand(self.num_envs, device=self.device) <= self.cfg.wrench_prob_per_rollout,
                True,
                False,
            )

        # initialize goal rotation - sample random 3D orientations
        self._compute_intermediate_values()

        if self.cfg.dynamic_goal_mode:
            # Inference mode: keep current goal (can be set externally)
            pass
        else:
            # Training mode: sample random 3D orientations
            self.goal_rot[env_ids] = random_orientation(len(env_ids), self.device)

        # Initialize checkpoint tracking for 3D
        # Compute initial rotation distance from current object orientation to goal
        current_rot_dist = rotation_distance(self.object_rot[env_ids], self.goal_rot[env_ids])
        self.initial_rot_dist[env_ids] = current_rot_dist
        self.prev_rot_dist[env_ids] = current_rot_dist
        
        # Compute number of checkpoints based on total rotation distance
        self.num_checkpoints[env_ids] = torch.ceil(
            current_rot_dist / self.cfg.checkpoint_step_rad
        ).to(torch.int32)
        self.num_checkpoints[env_ids] = torch.clamp(self.num_checkpoints[env_ids], min=1)
        
        # Reset checkpoints reached
        self.checkpoints_reached[env_ids] = 0

        # update goal markers
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)

    def _update_goals(self, new_success: torch.Tensor):
        """
        Handle goal switching logic for 3D reorientation.
        """
        # Start hold phase for newly successful envs
        newly_successful = new_success.bool()
        if newly_successful.any():
            hold_lengths = torch.randint(
                self.cfg.min_hold_steps,
                self.cfg.max_hold_steps + 1,
                (self.num_envs,),
                device=self.device,
                dtype=torch.int32,
            )
            self.hold_counter = torch.where(newly_successful, hold_lengths, self.hold_counter)
            self.goals_completed_this_episode = torch.where(
                newly_successful,
                self.goals_completed_this_episode + 1,
                self.goals_completed_this_episode,
            )

        # Decrement hold counters
        self.hold_counter = torch.clamp(self.hold_counter - 1, min=0)

        # Check for envs that finished holding and have succeeded
        finished_holding = (self.hold_counter == 0) & self.has_succeeded
        if finished_holding.any():
            env_indices = torch.where(finished_holding)[0]

            if not self.cfg.dynamic_goal_mode:
                # Training mode: sample new random 3D orientations
                self.goal_rot[env_indices] = random_orientation(len(env_indices), self.device)

                # Reset checkpoint tracking for new goals
                current_rot_dist = rotation_distance(self.object_rot[env_indices], self.goal_rot[env_indices])
                self.initial_rot_dist[env_indices] = current_rot_dist
                self.prev_rot_dist[env_indices] = current_rot_dist
                self.num_checkpoints[env_indices] = torch.ceil(
                    current_rot_dist / self.cfg.checkpoint_step_rad
                ).to(torch.int32)
                self.num_checkpoints[env_indices] = torch.clamp(self.num_checkpoints[env_indices], min=1)
                self.checkpoints_reached[env_indices] = 0

                # Update goal markers
                goal_pos = self.goal_pos + self.scene.env_origins
                self.goal_markers.visualize(goal_pos, self.goal_rot)

            # Reset success flags
            self.has_succeeded[env_indices] = False

    def _episode_length_bounds(self) -> tuple[int, int]:
        """Return (min_steps, max_steps) for episode lengths, with inference overrides."""
        dt = self.cfg.sim.dt * self.cfg.decimation
        min_steps = int(self.cfg.min_episode_length_s / dt)
        max_steps = getattr(self, "_max_episode_length_steps", self.max_episode_length)

        if self.cfg.dynamic_goal_mode:
            if getattr(self.cfg, "inference_min_episode_length_s", -1.0) > 0:
                min_steps = int(self.cfg.inference_min_episode_length_s / dt)
            if getattr(self.cfg, "inference_episode_length_s", -1.0) > 0:
                max_steps = int(self.cfg.inference_episode_length_s / dt)

        max_steps = max(max_steps, min_steps)
        return min_steps, max_steps

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w  # w,x,y,z
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def sim_real_indices(self):
        sim2real_idx_16, _ = self.hand.find_joints(self.cfg.actuated_joint_names, preserve_order=True)
        sim2real_idx_16 = torch.tensor(sim2real_idx_16) - min(sim2real_idx_16)
        real2sim_idx_16 = torch.empty_like(sim2real_idx_16)
        real2sim_idx_16[sim2real_idx_16] = torch.arange(len(sim2real_idx_16))

        print(f"sim2real_indices: {sim2real_idx_16}")
        print(f"real2sim_indices: {real2sim_idx_16}")

    def set_goal_quat(self, quat: torch.Tensor):
        """Set goal orientation directly (for inference/UI). quat shape: (4,) as [w,x,y,z]"""
        self.cfg.dynamic_goal_mode = True
        self.goal_rot[:] = quat.to(self.device)
        
        # Reset tracking
        self.has_succeeded[:] = False
        self.hold_counter[:] = 0
        
        # Update checkpoints
        current_rot_dist = rotation_distance(self.object_rot, self.goal_rot)
        self.initial_rot_dist[:] = current_rot_dist
        self.prev_rot_dist[:] = current_rot_dist
        self.num_checkpoints[:] = torch.ceil(current_rot_dist / self.cfg.checkpoint_step_rad).to(torch.int32)
        self.num_checkpoints[:] = torch.clamp(self.num_checkpoints, min=1)
        self.checkpoints_reached[:] = 0
        
        # Update markers
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    """Compute geodesic distance between two quaternions."""
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


@torch.jit.script
def quat_log(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Log map of unit quaternion to so(3) vector (axis * angle).
    """
    v = quat[:, 1:4]
    w = torch.clamp(quat[:, 0], -1.0 + eps, 1.0 - eps)
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(v_norm, w.unsqueeze(-1))
    safe_axis = torch.where(v_norm > eps, v / torch.clamp(v_norm, min=eps), torch.zeros_like(v))
    return safe_axis * angle


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    has_succeeded: torch.Tensor,
    max_episode_length: float,
    fingertip_pos: torch.Tensor,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    object_linvel: torch.Tensor,
    object_angvel: torch.Tensor,
    actions: torch.Tensor,
    pose_diff_penalty: torch.Tensor,
    torque_penalty: torch.Tensor,
    prev_rot_dist: torch.Tensor,
    # Reward scales
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_exp_decay_scale: float,
    rot_linear_penalty_scale: float,
    rot_progress_scale: float,
    rot_regress_penalty_scale: float,
    rot_eps: float,
    action_penalty_scale: float,
    pose_diff_penalty_scale: float,
    torque_penalty_scale: float,
    angvel_penalty_scale: float,
    align_angvel_scale: float,
    off_axis_penalty_scale: float,
    align_gate_dist: float,
    align_angvel_cap: float,
    # Success/failure params
    fall_dist: float,
    fall_penalty: float,
    success_tolerance: float,
    av_factor: float,
    fingertip_dist_penalty_scale: float,
    centering_sigma: float,
    centering_reward_scale: float,
):
    """
    Compute rewards for 3D reorientation.
    """
    # Distance from object to target position (for fall detection and cube-in-hand)
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # Rotation distance to goal (geodesic on SO(3))
    rot_dist = rotation_distance(object_rot, target_rot)

    # === REWARD TERMS ===
    
    # 1. Distance reward: keep cube in hand
    dist_rew = goal_dist * dist_reward_scale
    centering_rew = torch.exp(-goal_dist / centering_sigma) * centering_reward_scale

    # 2. Rotation reward: incentivize alignment with goal (geodesic-aware)
    at_goal = rot_dist < success_tolerance
    rot_dist_stable = rot_dist + rot_eps
    rot_decay = torch.exp(-rot_dist_stable * rot_exp_decay_scale) * rot_reward_scale
    rot_linear_penalty = -rot_dist_stable * rot_linear_penalty_scale
    rot_rew = torch.where(at_goal, torch.ones_like(rot_dist) * rot_reward_scale, rot_decay + rot_linear_penalty)

    # 2b. Per-step geodesic progress / regress
    rot_progress = prev_rot_dist - rot_dist
    progress_bonus = torch.clamp(rot_progress, min=0.0) * rot_progress_scale
    regress_penalty = torch.clamp(-rot_progress, min=0.0) * rot_regress_penalty_scale

    # 2c. Align angular velocity with shortest-arc axis, penalize off-axis spinning
    rel_quat = quat_mul(target_rot, quat_conjugate(object_rot))
    rel_quat = torch.where(rel_quat[:, 0:1] < 0, -rel_quat, rel_quat)
    log_err = quat_log(rel_quat, 1e-6)
    log_norm = torch.norm(log_err, p=2, dim=-1, keepdim=True)
    has_axis = log_norm > 1e-6
    err_axis = torch.where(has_axis, log_err / torch.clamp(log_norm, min=1e-6), torch.zeros_like(log_err))

    angvel_dot_raw = torch.where(
        has_axis.squeeze(-1),
        torch.sum(object_angvel * err_axis, dim=-1),
        torch.zeros_like(rot_dist),
    )
    angvel_dot_clamped = torch.clamp(angvel_dot_raw, -align_angvel_cap, align_angvel_cap)
    gated_align = torch.where(rot_dist < align_gate_dist, angvel_dot_clamped, torch.zeros_like(angvel_dot_clamped))
    align_angvel_rew = gated_align * align_angvel_scale

    off_axis_vec = object_angvel - (angvel_dot_raw.unsqueeze(-1) * err_axis)
    off_axis_mag = torch.where(
        has_axis.squeeze(-1),
        torch.norm(off_axis_vec, p=2, dim=-1),
        torch.zeros_like(rot_dist),
    )
    gated_off_axis = torch.where(rot_dist < align_gate_dist, off_axis_mag, torch.zeros_like(off_axis_mag))
    off_axis_penalty = gated_off_axis * off_axis_penalty_scale
    
    # 3. Action penalty
    action_penalty = torch.sum(actions ** 2, dim=-1)

    # 4. Pose difference penalty
    pose_rew = pose_diff_penalty * pose_diff_penalty_scale

    # 5. Torque penalty
    torque_rew = torque_penalty * torque_penalty_scale

    # 6. Angular velocity penalty: only when near goal
    ang_speed = torch.norm(object_angvel, p=2, dim=-1)
    near_goal = rot_dist < 0.3
    angvel_rew = torch.where(near_goal, ang_speed * angvel_penalty_scale, torch.zeros_like(ang_speed))

    # Fingertip distance penalty
    fingertip_dist_penalty = torch.norm(fingertip_pos - object_pos.unsqueeze(1), p=2, dim=-1)
    fingertip_dist_penalty = torch.mean(fingertip_dist_penalty, dim=-1)

    # Total base reward
    reward = (
        dist_rew
        + rot_rew
        + progress_bonus
        - regress_penalty
        + align_angvel_rew
        + off_axis_penalty
        + action_penalty * action_penalty_scale
        + pose_rew
        + torque_rew
        + angvel_rew
        + centering_rew
        + fingertip_dist_penalty * fingertip_dist_penalty_scale
    )

    # === SUCCESS DETECTION ===
    angvel_threshold = 2.0
    success_condition = (
        (torch.abs(rot_dist) <= success_tolerance)
        & (goal_dist <= 0.025)
        & (ang_speed <= angvel_threshold)
    )

    new_success = success_condition & (~has_succeeded)
    goal_resets = torch.where(new_success, torch.ones_like(reset_goal_buf), reset_buf)

    successes = successes + goal_resets

    # Fall penalty and termination
    fall_envs = goal_dist >= fall_dist
    reward = torch.where(fall_envs, reward + fall_penalty, reward)
    resets = torch.where(fall_envs, torch.ones_like(reset_buf), reset_buf)

    # Track consecutive successes
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return (
        reward,
        goal_resets,
        successes,
        cons_successes,
        new_success,
        rot_dist,
        rot_progress,
        regress_penalty,
        align_angvel_rew,
        off_axis_penalty,
    )
