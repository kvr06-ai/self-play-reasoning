"""
Training components for SPIRAL reasoning simulator.

This module contains the self-play training logic, PPO implementation with
role-conditioned advantage estimation, and training utilities.
"""

from .self_play_trainer import SelfPlayTrainer
from .ppo_trainer import PPOTrainer
from .opponent_manager import OpponentManager
from .training_utils import TrainingUtils

__all__ = ["SelfPlayTrainer", "PPOTrainer", "OpponentManager", "TrainingUtils"] 