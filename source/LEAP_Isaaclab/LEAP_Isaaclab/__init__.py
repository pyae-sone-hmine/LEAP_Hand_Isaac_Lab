"""
Python module entrypoint.

Note: We intentionally avoid importing tasks/UI here to prevent heavy Isaac Sim
dependencies from being pulled in before the AppLauncher sets up paths.
Callers should import `LEAP_Isaaclab.tasks` (and UI modules if needed) after
AppLauncher initialization.
"""

__all__ = []
