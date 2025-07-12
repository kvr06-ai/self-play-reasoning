"""
TicTacToe Game Environment

A simple TicTacToe implementation using Gymnasium for SPIRAL training.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional


class TicTacToeEnv(gym.Env):
    """
    TicTacToe environment for SPIRAL training.
    
    - 3x3 grid
    - Players alternate turns (1 and -1)
    - Action space: 9 positions (0-8)
    - Observation space: 3x3 grid with values {-1, 0, 1}
    - Reward: +1 for win, -1 for loss, 0 for draw/ongoing
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # 3x3 grid, each cell can be -1 (player 2), 0 (empty), or 1 (player 1)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(3, 3), dtype=np.int8
        )
        
        # 9 possible actions (positions 0-8)
        self.action_space = spaces.Discrete(9)
        
        self.render_mode = render_mode
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the game to initial state."""
        super().reset(seed=seed)
        
        # Initialize empty board
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.game_over = False
        self.winner = None
        self.move_count = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Position to place mark (0-8)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over:
            raise ValueError("Game is already over. Call reset() to start new game.")
        
        # Convert action to row, col
        row, col = divmod(action, 3)
        
        # Check if move is valid
        if self.board[row, col] != 0:
            # Invalid move - penalize and end game
            reward = -1.0
            terminated = True
            self.game_over = True
            info = self._get_info()
            info["invalid_move"] = True
            return self._get_observation(), reward, terminated, False, info
        
        # Make the move
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Check for win
        winner = self._check_winner()
        if winner is not None:
            self.game_over = True
            self.winner = winner
            reward = 1.0 if winner == self.current_player else -1.0
            terminated = True
        elif self.move_count >= 9:
            # Draw
            self.game_over = True
            reward = 0.0
            terminated = True
        else:
            # Game continues
            reward = 0.0
            terminated = False
            self.current_player *= -1  # Switch player
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current board state."""
        return self.board.copy()
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the game state."""
        return {
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "move_count": self.move_count,
            "valid_actions": self._get_valid_actions()
        }
    
    def _get_valid_actions(self) -> list:
        """Get list of valid actions (empty positions)."""
        valid_actions = []
        for i in range(9):
            row, col = divmod(i, 3)
            if self.board[row, col] == 0:
                valid_actions.append(i)
        return valid_actions
    
    def _check_winner(self) -> Optional[int]:
        """
        Check if there's a winner.
        
        Returns:
            1 if player 1 wins, -1 if player 2 wins, None if no winner
        """
        # Check rows
        for row in range(3):
            if abs(self.board[row, :].sum()) == 3:
                return self.board[row, 0]
        
        # Check columns
        for col in range(3):
            if abs(self.board[:, col].sum()) == 3:
                return self.board[0, col]
        
        # Check diagonals
        if abs(self.board.diagonal().sum()) == 3:
            return self.board[0, 0]
        
        if abs(np.fliplr(self.board).diagonal().sum()) == 3:
            return self.board[0, 2]
        
        return None
    
    def render(self) -> Optional[np.ndarray]:
        """Render the game state."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Print the board to console."""
        print("\n" + "="*13)
        for row in range(3):
            print("|", end="")
            for col in range(3):
                cell = self.board[row, col]
                if cell == 1:
                    print(" X ", end="|")
                elif cell == -1:
                    print(" O ", end="|")
                else:
                    print(f" {row*3 + col} ", end="|")
            print()
            print("="*13)
        
        if self.game_over:
            if self.winner is not None:
                winner_symbol = "X" if self.winner == 1 else "O"
                print(f"Game Over! Winner: {winner_symbol}")
            else:
                print("Game Over! It's a draw!")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for visualization."""
        # Simple RGB representation
        rgb = np.zeros((3, 3, 3), dtype=np.uint8)
        
        # Player 1 (X) = Red, Player 2 (O) = Blue, Empty = White
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 1:
                    rgb[row, col] = [255, 0, 0]  # Red
                elif self.board[row, col] == -1:
                    rgb[row, col] = [0, 0, 255]  # Blue
                else:
                    rgb[row, col] = [255, 255, 255]  # White
        
        return rgb
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions (1 for valid, 0 for invalid)."""
        mask = np.zeros(9, dtype=np.int8)
        for action in self._get_valid_actions():
            mask[action] = 1
        return mask


def create_tictactoe_env() -> TicTacToeEnv:
    """Factory function to create a TicTacToe environment."""
    return TicTacToeEnv()


if __name__ == "__main__":
    # Test the environment
    env = TicTacToeEnv(render_mode="human")
    
    # Play a simple game
    obs, info = env.reset()
    print("Initial state:")
    env.render()
    
    # Make some moves
    moves = [0, 4, 1, 3, 2]  # X wins
    for move in moves:
        if not env.game_over:
            obs, reward, terminated, truncated, info = env.step(move)
            print(f"\nMove: {move}, Reward: {reward}")
            env.render()
            
            if terminated:
                print(f"Game terminated! Final reward: {reward}")
                break 