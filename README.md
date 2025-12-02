# LEAP Hand 1-Axis In-Hand Reorientation for Isaac Lab
This repository is for the final project of the MIT Robotic Manipulation 6.4212 course. It has code for in-hand 1-axis reorientation, either infinitely spinning or to a specified goal position. It will soon have code for full 3D reorientation.

## Installation
Download both Isaac Sim and IsaacLab from the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). This repository is tested on a conda environment with IsaacSim version 4.5 and Isaaclab version 2.1.0 on Ubuntu 22.04.

Start by activating the conda environment:
```bash
conda create -n isaaclab python=3.10
conda activate isaaclab
```

Download the repository and install the library:

```bash
git clone https://github.com/pyae-sone-hmine/LEAP_Hand_Isaac_Lab.git
cd LEAP_Hand_Isaac_Lab
python -m pip install -e source/LEAP_Isaaclab
```


## Training a policy
Once in the `LEAP_Isaaclab` parent directory in your terminal, you can train a policy by entering the following command into the terminal. Note that the policies in this repository are tested with the `rl_games` library.

You can train a policy using the following command:
```bash
python scripts/rl_games/train.py --task Isaac-Reorient-Cube-Leap --headless
```

And visualize infinite spinning with the policy with the following command:
```bash
python scripts/rl_games/play.py --task Isaac-Reorient-Cube-Leap --num_envs 1
```

Or to a goal position using:
```bash
python scripts/rl_games/play.py --task Isaac-Reorient-Cube-Leap --num_envs 1 --goal_angle 0
```