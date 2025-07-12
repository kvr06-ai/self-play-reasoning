"""
Game utility functions for SPIRAL training.

This module contains helper functions for game environments,
including multi-turn logic and game state management.
"""

import gymnasium as gym
from typing import Dict, Any, Type, Union
import numpy as np

from .tictactoe import TicTacToeEnv
from .kuhn_poker import KuhnPokerEnv


# Game registry
GAMES_REGISTRY: Dict[str, Type[gym.Env]] = {
    "tictactoe": TicTacToeEnv,
    "kuhn_poker": KuhnPokerEnv,
}


def create_game_env(game_name: str, **kwargs) -> gym.Env:
    """
    Create a game environment by name.
    
    Args:
        game_name: Name of the game ("tictactoe", "kuhn_poker")
        **kwargs: Additional arguments for the environment
        
    Returns:
        Game environment instance
        
    Raises:
        ValueError: If game_name is not recognized
    """
    if game_name not in GAMES_REGISTRY:
        available_games = list(GAMES_REGISTRY.keys())
        raise ValueError(f"Unknown game: {game_name}. Available games: {available_games}")
    
    game_class = GAMES_REGISTRY[game_name]
    return game_class(**kwargs)


def get_game_info(game_name: str) -> Dict[str, Any]:
    """
    Get information about a game environment.
    
    Args:
        game_name: Name of the game
        
    Returns:
        Dictionary with game information
    """
    env = create_game_env(game_name)
    
    info = {
        "name": game_name,
        "action_space": env.action_space,
        "observation_space": env.observation_space,
        "max_episode_steps": getattr(env, "_max_episode_steps", None),
        "render_modes": env.metadata.get("render_modes", []),
    }
    
    # Add game-specific information
    if game_name == "tictactoe":
        info.update({
            "description": "3x3 TicTacToe game with alternating turns",
            "players": 2,
            "zero_sum": True,
            "perfect_information": True,
        })
    elif game_name == "kuhn_poker":
        info.update({
            "description": "Simplified poker with 3 cards (J, Q, K)",
            "players": 2,
            "zero_sum": True,
            "perfect_information": False,
        })
    
    env.close()
    return info


def get_available_games() -> list:
    """Get list of available game names."""
    return list(GAMES_REGISTRY.keys())


def is_game_over(env: gym.Env) -> bool:
    """
    Check if the game is over.
    
    Args:
        env: Game environment
        
    Returns:
        True if game is over, False otherwise
    """
    if hasattr(env, 'game_over'):
        return env.game_over
    return False


def get_valid_actions(env: gym.Env) -> list:
    """
    Get valid actions for the current state.
    
    Args:
        env: Game environment
        
    Returns:
        List of valid actions
    """
    if hasattr(env, '_get_valid_actions'):
        return env._get_valid_actions()
    elif hasattr(env, 'get_valid_actions'):
        return env.get_valid_actions()
    else:
        # Fallback: assume all actions are valid
        return list(range(env.action_space.n))


def get_action_mask(env: gym.Env) -> np.ndarray:
    """
    Get action mask for the current state.
    
    Args:
        env: Game environment
        
    Returns:
        Boolean mask where True indicates valid actions
    """
    if hasattr(env, 'get_action_mask'):
        return env.get_action_mask()
    else:
        # Fallback: create mask from valid actions
        valid_actions = get_valid_actions(env)
        mask = np.zeros(env.action_space.n, dtype=bool)
        for action in valid_actions:
            mask[action] = True
        return mask


def play_random_game(game_name: str, render: bool = False, seed: int = None) -> Dict[str, Any]:
    """
    Play a random game to completion.
    
    Args:
        game_name: Name of the game to play
        render: Whether to render the game
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with game results
    """
    env = create_game_env(game_name, render_mode="human" if render else None)
    
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    
    if render:
        env.render()
    
    total_reward = 0
    step_count = 0
    actions_taken = []
    
    while not is_game_over(env):
        valid_actions = get_valid_actions(env)
        action = np.random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        actions_taken.append(action)
        total_reward += reward
        step_count += 1
        
        if render:
            print(f"Step {step_count}: Action {action}, Reward: {reward}")
            env.render()
        
        if terminated or truncated:
            break
    
    results = {
        "game_name": game_name,
        "total_reward": total_reward,
        "step_count": step_count,
        "actions_taken": actions_taken,
        "winner": getattr(env, 'winner', None),
        "final_info": info
    }
    
    env.close()
    return results


if __name__ == "__main__":
    # Test the utilities
    print("Available games:", get_available_games())
    
    for game_name in get_available_games():
        print(f"\n{game_name.upper()} Info:")
        info = get_game_info(game_name)
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Play a random game
    print("\nPlaying random TicTacToe game:")
    result = play_random_game("tictactoe", render=True, seed=42) 