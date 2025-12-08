# LEAP Hand In-Hand Reorientation for Isaac Lab
This repository is for the final project of the MIT Robotic Manipulation 6.4212 course. It has code for in-hand reorientation tasks including 1-axis rotation and full 3D reorientation.

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


## Project Structure
```
source/LEAP_Isaaclab/LEAP_Isaaclab/tasks/
├── reorient_1d/              # 1-axis infinite rotation task
├── reorient_1dbi2/           # 1-axis goal-conditioned rotation task
└── reorient_3d/              # Full 3D reorientation task
    ├── __init__.py
    ├── leap_hand_env_cfg_3d.py
    ├── reorientation_env_3d.py
    └── agents/

logs/rl_games/
├── reorient_1d/              # 1D policy checkpoints
├── reorient_1dbi2/           # 1D bi-directional policy checkpoints
└── reorient_3d/              # 3D policy checkpoints
```

## Available Tasks

| Task Name | Description |
|-----------|-------------|
| `Reorient_Cube_1D` | Original LEAP policy: infinite spinning with environment-controlled pauses |
| `Reorient_Cube_1Dbi2` | Goal-conditioned 1D rotation with bi-directional control |
| `Reorient_Cube_3D` | Full 3D reorientation with uniform SO(3) goal sampling |

## Training

Train a policy using the `rl_games` library:

### 1D Rotation
```bash
python scripts/rl_games/train.py --task Reorient_Cube_1D --headless
```

### 1D Bi-directional (Goal-conditioned)
```bash
python scripts/rl_games/train.py --task Reorient_Cube_1Dbi2 --headless
```

### 3D Reorientation
```bash
python scripts/rl_games/train.py --task Reorient_Cube_3D --headless
```

## Playing / Visualization

### 1D Tasks
```bash
python scripts/rl_games/play.py --task Reorient_Cube_1D --num_envs 1  --goal_angle 0
python scripts/rl_games/play.py --task Reorient_Cube_1Dbi2 --num_envs 1 --goal_angle 0
```

### 3D Reorientation
```bash
python scripts/rl_games/play.py --task Reorient_Cube_3D --num_envs 1 --goal_angle 0
```

## Technical Details

### 3D Goal Sampling
The 3D task uses **uniform quaternion sampling over SO(3)** via 4D Gaussian normalization:
```python
q = torch.randn(n, 4, device=device)
q = q / torch.norm(q, dim=-1, keepdim=True)
```
This avoids gimbal lock issues that would arise from Euler angle sampling and provides mathematically uniform coverage of all possible orientations.
