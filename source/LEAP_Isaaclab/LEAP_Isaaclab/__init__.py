"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
from .ui_extension import set_env_reference, create_goal_angle_ui, destroy_goal_angle_ui
