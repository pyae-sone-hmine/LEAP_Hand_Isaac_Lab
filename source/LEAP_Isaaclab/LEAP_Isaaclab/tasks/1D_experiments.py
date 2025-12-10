from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from isaaclab.app import AppLauncher
from LEAP_Isaaclab.plotting import plot_metrics


def rotation_distance(object_rot: torch.Tensor, target_rot: torch.Tensor) -> torch.Tensor:
    """Match env reward definition for orientation distance."""
    from isaaclab.utils.math import quat_conjugate, quat_mul

    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))


def load_policy_and_env(
    task_id: str,
    num_envs: int,
    device: str,
    disable_fabric: bool,
    checkpoint_root: str,
) -> tuple[Any, Any, Any, str, torch.Tensor, float]:
    """Create env, wrap for RL-Games, and load latest checkpoint (newest run)."""
    import gymnasium as gym
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    env_cfg: Any = parse_env_cfg(task_id, device=device, num_envs=num_envs, use_fabric=not disable_fabric)
    agent_cfg = load_cfg_from_registry(task_id, "rl_games_cfg_entry_point")
    agent_cfg["params"]["config"]["device"] = device

    log_root_path = os.path.abspath(os.path.join(checkpoint_root, agent_cfg["params"]["config"]["name"]))
    if not os.path.isdir(log_root_path):
        raise FileNotFoundError(
            f"Checkpoint root not found for task '{task_id}': {log_root_path}. "
            "Pass --checkpoint_root to point to your trained runs or train the policy first."
        )

    run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
    checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
    resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", np.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", np.inf)

    env = gym.make(task_id, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    # Use a unique env name per task to avoid stale registry entries across sequential runs.
    base_env_name = agent_cfg["params"]["config"].get("env_name", "rlgpu")
    env_name = f"{base_env_name}_{task_id}"
    agent_cfg["params"]["config"]["env_name"] = env_name
    # Clear any previous registration with the same name to avoid stale env pointers.
    if env_name in env_configurations.configurations:
        del env_configurations.configurations[env_name]
    env_configurations.register(env_name, {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    if getattr(agent, "is_rnn", False):
        agent.init_rnn()
        states = getattr(agent, "states", None)
        if states is not None:
            num_envs = env.unwrapped.num_envs
            for i, s in enumerate(states):
                if s.shape[1] != num_envs:
                    states[i] = s.repeat(1, num_envs, 1)

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)

    step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
    return env, agent, env_cfg, resume_path, obs, step_dt


def run_policy(
    task_id: str,
    num_envs: int,
    device: str,
    max_goal_time_s: float,
    max_total_time_s: float,
    disable_fabric: bool,
    checkpoint_root: str,
    hold_error_window_s: float,
) -> tuple[Dict, Any]:
    """Run repeated goals until first failure for each env; collect metrics."""
    import LEAP_Isaaclab.tasks  # noqa: F401 - registers gym tasks after AppLauncher sets paths

    env, agent, env_cfg, resume_path, obs, step_dt = load_policy_and_env(
        task_id,
        num_envs=num_envs,
        device=device,
        disable_fabric=disable_fabric,
        checkpoint_root=checkpoint_root,
    )
    base_env = env.unwrapped

    max_goal_steps = max(1, int(max_goal_time_s / step_dt))
    max_total_steps = None if max_total_time_s <= 0 else int(max_total_time_s / step_dt)

    goal_start_step = torch.zeros(env.num_envs, dtype=torch.long, device=base_env.device)
    current_goal_rot = base_env.goal_rot.clone()
    successes_before_failure = torch.zeros(env.num_envs, dtype=torch.long, device=base_env.device)
    active = torch.ones(env.num_envs, dtype=torch.bool, device=base_env.device)

    time_to_goal: List[float] = []
    steady_state_error: List[float] = []
    failures = torch.zeros(env.num_envs, dtype=torch.bool, device=base_env.device)

    # Jitter metric: high-pass residual RMS on joint positions via EMA smoothing.
    ema_alpha = 0.1  # lower -> smoother trend, higher -> follows quickly
    ema_joint_pos = base_env.hand_dof_pos.clone()
    jitter_power_sum = torch.zeros(env.num_envs, device=base_env.device)
    jitter_steps = torch.zeros(env.num_envs, device=base_env.device)

    # Steady-state hold window accumulation after success.
    hold_window_steps = max(1, int(hold_error_window_s / step_dt)) if hold_error_window_s > 0 else 1
    hold_remaining = torch.zeros(env.num_envs, dtype=torch.long, device=base_env.device)
    hold_err_sum = torch.zeros(env.num_envs, device=base_env.device)
    hold_err_count = torch.zeros(env.num_envs, device=base_env.device)

    total_steps = 0
    with torch.inference_mode():
        while active.any() and (max_total_steps is None or total_steps < max_total_steps):
            prev_has_succeeded = base_env.has_succeeded.clone()
            obs_tensor = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_tensor, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            total_steps += 1

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

            # Detect goal changes to start new attempts.
            goal_changed = torch.norm(base_env.goal_rot - current_goal_rot, dim=1) > 1e-6
            if goal_changed.any():
                goal_start_step[goal_changed] = total_steps
                current_goal_rot[goal_changed] = base_env.goal_rot[goal_changed]

            # New successes this step.
            new_success = (base_env.has_succeeded & (~prev_has_succeeded)) & active
            if new_success.any():
                # Start hold window accumulation for these envs.
                hold_remaining[new_success] = hold_window_steps
                hold_err_sum[new_success] = 0.0
                hold_err_count[new_success] = 0.0

                elapsed_steps = total_steps - goal_start_step[new_success]
                time_to_goal.extend((elapsed_steps.float() * step_dt).cpu().tolist())
                rot_err = rotation_distance(base_env.object_rot[new_success], base_env.goal_rot[new_success])
                # Snapshot at success (kept for reference)
                steady_state_error.extend(rot_err.detach().cpu().tolist())
                successes_before_failure[new_success] += 1
                goal_start_step[new_success] = total_steps  # reset timer; next goal sampled shortly

            # Hold-window steady-state error accumulation for active envs in hold phase.
            if hold_remaining.any():
                rot_err_all = rotation_distance(base_env.object_rot, base_env.goal_rot)
                collecting = (hold_remaining > 0) & active
                if collecting.any():
                    hold_err_sum[collecting] += rot_err_all[collecting]
                    hold_err_count[collecting] += 1.0
                    hold_remaining[collecting] -= 1
                    finished = (hold_remaining == 0) & (hold_err_count > 0)
                    if finished.any():
                        steady_state_error.extend(
                            (hold_err_sum[finished] / hold_err_count[finished]).detach().cpu().tolist()
                        )
                        hold_err_sum[finished] = 0.0
                        hold_err_count[finished] = 0.0

            # Failure detection: env done or stuck.
            if not torch.is_tensor(dones):
                dones = torch.as_tensor(dones, device=base_env.device, dtype=torch.bool)
            done_mask = dones.to(dtype=torch.bool)
            steps_since_goal = total_steps - goal_start_step
            stuck_mask = (steps_since_goal >= max_goal_steps) & (~base_env.has_succeeded) & active
            failure_mask = (done_mask | stuck_mask) & active
            if failure_mask.any():
                failures |= failure_mask
                active = active & (~failure_mask)
                # Cancel hold accumulation for failed envs.
                hold_remaining[failure_mask] = 0
                hold_err_sum[failure_mask] = 0.0
                hold_err_count[failure_mask] = 0.0
                if not active.any():
                    break

            # Jitter accumulation (only for still-active envs)
            current_pos = base_env.hand_dof_pos
            residual = current_pos - ema_joint_pos
            ema_joint_pos = ema_joint_pos + ema_alpha * residual
            residual_power = (residual.pow(2).mean(dim=1))  # mean over DOFs
            active_f = active.to(residual_power.dtype)
            jitter_power_sum += residual_power * active_f
            jitter_steps += active_f

            if max_total_steps is not None and total_steps >= max_total_steps:
                break

    successes = successes_before_failure.sum().item()
    failure_count = failures.sum().item()
    success_rate = successes / max(successes + failure_count, 1)
    jitter_rms = torch.sqrt(jitter_power_sum / torch.clamp(jitter_steps, min=1.0))

    return (
        {
            "task_id": task_id,
            "checkpoint": resume_path,
            "num_envs": num_envs,
            "step_dt": step_dt,
            "time_to_goal": time_to_goal,
            "steady_state_error": steady_state_error,
            "goals_until_failure": successes_before_failure.cpu().tolist(),
            "failures": failure_count,
            "success_rate": success_rate,
            "mean_time_to_goal": float(np.mean(time_to_goal)) if time_to_goal else None,
            "mean_steady_state_error": float(np.mean(steady_state_error)) if steady_state_error else None,
            "jitter_rms_per_env": jitter_rms.cpu().tolist(),
            "mean_jitter_rms": float(jitter_rms.mean().item()),
        },
        env,
    )


def main():
    parser = argparse.ArgumentParser(description="Compare 1D bimanual policies.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "Reorient_Cube_1Dbi",
            "Reorient_Cube_1Dbi2",
            "Reorient_Cube_1Dbi3",
            "Reorient_Cube_1Dbi4",
        ],
        help="Task IDs to evaluate.",
    )
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel envs per policy.")
    parser.add_argument("--max_goal_time_s", type=float, default=10.0, help="Max seconds allowed per goal attempt.")
    parser.add_argument(
        "--max_total_time_s",
        type=float,
        default=64.0,
        help="Stop a policy run after this many seconds of simulated time (<=0 to disable).",
    )
    parser.add_argument(
        "--hold_error_window_s",
        type=float,
        default=0.3,
        help="Window (seconds) after success to average steady-state orientation error.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/1d_bi_comparison",
        help="Directory to save metrics and plots.",
    )
    parser.add_argument("--disable_fabric", action="store_true", help="Disable fabric for sim.")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=os.path.join("logs", "rsl_rl"),
        help="Root directory containing trained runs (per experiment_name).",
    )
    parser.add_argument(
        "--fast_no_render",
        action="store_true",
        help="Run headless to maximize sim speed (sets headless mode).",
    )
    parser.add_argument(
        "--internal_single_task",
        action="store_true",
        help=argparse.SUPPRESS,  # used for per-task subprocess to avoid recursion
    )

    # App + device args
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Enable headless mode for faster-than-real-time when requested.
    if getattr(args, "headless", False) is False and args.fast_no_render:
        args.headless = True

    # If multiple tasks and not already in a per-task subprocess, spawn one process per task
    # to avoid SimulationApp re-initialization issues. Aggregate results after.
    if len(args.tasks) > 1 and not args.internal_single_task:
        aggregate_results: Dict[str, Dict] = {}
        script_path = Path(__file__).resolve()
        for task_id in args.tasks:
            per_task_output = Path(args.output_dir) / task_id
            per_task_output.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(script_path),
                "--internal_single_task",
                "--tasks",
                task_id,
                "--num_envs",
                str(args.num_envs),
                "--max_goal_time_s",
                str(args.max_goal_time_s),
                "--max_total_time_s",
                str(args.max_total_time_s),
                "--output_dir",
                str(per_task_output),
                "--checkpoint_root",
                str(args.checkpoint_root),
            ]
            if args.disable_fabric:
                cmd.append("--disable_fabric")
            if args.fast_no_render:
                cmd.append("--fast_no_render")
            # Preserve device/headless flags if provided
            if args.device is not None:
                cmd.extend(["--device", args.device])
            if getattr(args, "headless", False):
                cmd.append("--headless")
            print(f"[INFO] Spawning subprocess for task {task_id}: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"[ERROR] Task {task_id} subprocess failed (code {proc.returncode}).")
                print(proc.stdout)
                print(proc.stderr)
                continue
            # Load per-task results
            res_path = per_task_output / "results.json"
            if res_path.exists():
                with open(res_path, "r", encoding="utf-8") as f:
                    task_results = json.load(f)
                    aggregate_results.update(task_results)
            else:
                print(f"[WARN] No results file found for task {task_id} at {res_path}")
        # After subprocesses, plot aggregate
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(aggregate_results, f, indent=2)
        plot_metrics(aggregate_results, output_dir)
        return

    device = args.device if args.device is not None else "cuda:0"
    results: Dict[str, Dict] = {}
    # Launch a fresh simulator for this (single) task.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Now that AppLauncher has set up the paths, import the tasks registry.
    import LEAP_Isaaclab.tasks  # noqa: F401

    task_id = args.tasks[0]
    print(f"[INFO] Running policy {task_id} ...", flush=True)
    metrics, env = run_policy(
        task_id=task_id,
        num_envs=args.num_envs,
        device=device,
        max_goal_time_s=args.max_goal_time_s,
        max_total_time_s=args.max_total_time_s,
        disable_fabric=args.disable_fabric,
        checkpoint_root=args.checkpoint_root,
        hold_error_window_s=args.hold_error_window_s,
    )
    results[task_id] = metrics
    
    # Print results and save to file BEFORE closing simulation (which may crash)
    print(
        f"[RESULT] {task_id}: success_rate={results[task_id]['success_rate']:.3f}, "
        f"mean_time_to_goal={results[task_id]['mean_time_to_goal']}, "
        f"mean_error={results[task_id]['mean_steady_state_error']}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_metrics(results, output_dir)

    # Close env and simulation AFTER saving results (may crash process)
    try:
        env.close()
    except Exception:
        pass
    try:
        simulation_app.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
