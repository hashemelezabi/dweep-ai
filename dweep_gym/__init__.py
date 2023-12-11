""" Registers the gymnasium environments and exports the `gymnasium.make` function.
"""
# Silencing pygame:
import os

# Registering environments:
from gymnasium.envs.registration import register

# Exporting envs:
from dweep_gym.envs.dweep_env import DweepEnv

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

register(
    id="Dweep-v0",
    entry_point="dweep_gym:DweepEnv",
)

# Main names:
__all__ = [
    DweepEnv.__name__,
]