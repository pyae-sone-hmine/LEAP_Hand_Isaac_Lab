"""
Leap Inhand Manipulation environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

reorient_task_entry = "LEAP_Isaaclab.tasks.reorient_1dbi"

gym.register(
    id="Reorient_Cube_1Dbi",
    entry_point=f"{reorient_task_entry}.reorientation_env_bi:ReorientationEnvBi",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leap_hand_env_cfg_bi:LeapHandEnvCfgBi",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LeapHandPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
