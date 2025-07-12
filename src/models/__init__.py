"""
SPIRAL model implementations.

This module contains the core SPIRAL model architecture and
role-conditioned advantage estimation (RAE) components.
"""

from .spiral_model import SpiralModel
from .rae import RoleConditionedAdvantageEstimator
from .policy_network import PolicyNetwork
from .value_network import ValueNetwork

__all__ = ["SpiralModel", "RoleConditionedAdvantageEstimator", "PolicyNetwork", "ValueNetwork"] 