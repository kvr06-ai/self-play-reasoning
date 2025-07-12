"""
Model implementations for SPIRAL reasoning simulator.

This module contains the SPIRAL model architecture, role-conditioned advantage
estimation, and other model components for self-play training.
"""

from .spiral_model import SpiralModel
from .rae import RoleConditionedAdvantageEstimator
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork

__all__ = ["SpiralModel", "RoleConditionedAdvantageEstimator", "PolicyNetwork", "ValueNetwork"] 