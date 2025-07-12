"""
Game environments for SPIRAL training.

This module contains implementations of zero-sum games used for
self-play training, including Kuhn Poker and TicTacToe.
"""

from .tictactoe import TicTacToeEnv, create_tictactoe_env
from .kuhn_poker import KuhnPokerEnv, create_kuhn_poker_env

__all__ = [
    "TicTacToeEnv",
    "KuhnPokerEnv", 
    "create_tictactoe_env",
    "create_kuhn_poker_env"
] 