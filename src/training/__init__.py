"""
Training components for SPIRAL.

This module implements the self-play training logic using PPO
with role-conditioned advantage estimation.
"""

from .self_play_trainer import SelfPlayTrainer
from .ppo_trainer import PPOTrainer
from .opponent_manager import OpponentManager
from .training_utils import TrainingUtils

__all__ = ["SelfPlayTrainer", "PPOTrainer", "OpponentManager", "TrainingUtils"] 