"""Shared reinforcement-learning components for the project experiments.

The CrossQ implementation was adapted from the original progress-as-reward
experiment at ``src/historical_progress_reward/cleanrl_crossq.py``. That entry
point remains runnable;
the shared modules here prevent the final visual-reward experiment from
depending on a second copy of the same networks and environment helpers.
"""

from .core import BatchRenorm1d, PolicyNetwork, QNetwork, SimpleReplayBuffer, set_seed

__all__ = [
    "BatchRenorm1d",
    "PolicyNetwork",
    "QNetwork",
    "SimpleReplayBuffer",
    "set_seed",
]
