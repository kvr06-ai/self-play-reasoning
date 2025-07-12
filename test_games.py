#!/usr/bin/env python3
"""
Test script for game environments.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from games import TicTacToeEnv, KuhnPokerEnv, create_tictactoe_env, create_kuhn_poker_env
from games.game_utils import get_available_games, get_game_info, play_random_game

def test_tictactoe():
    """Test TicTacToe environment."""
    print("Testing TicTacToe...")
    env = create_tictactoe_env()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a few moves
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After move {action}: reward={reward}, terminated={terminated}")
    
    env.close()
    print("TicTacToe test passed!\n")


def test_kuhn_poker():
    """Test Kuhn Poker environment."""
    print("Testing Kuhn Poker...")
    env = create_kuhn_poker_env()
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test a move
    action = 0  # Check/Call
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"After action {action}: reward={reward}, terminated={terminated}")
    
    env.close()
    print("Kuhn Poker test passed!\n")


def test_game_utils():
    """Test game utility functions."""
    print("Testing game utilities...")
    
    # Test available games
    games = get_available_games()
    print(f"Available games: {games}")
    
    # Test game info
    for game_name in games:
        info = get_game_info(game_name)
        print(f"{game_name} info: {info['description']}")
    
    print("Game utilities test passed!\n")


def main():
    """Run all tests."""
    print("Running game environment tests...\n")
    
    try:
        test_tictactoe()
        test_kuhn_poker()
        test_game_utils()
        print("All tests passed! ✅")
    except Exception as e:
        print(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 