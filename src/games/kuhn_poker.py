"""
Kuhn Poker Game Environment

A simple Kuhn Poker implementation using Gymnasium for SPIRAL training.
Kuhn Poker is a simplified poker variant with 3 cards (J, Q, K).
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional, List
import random


class KuhnPokerEnv(gym.Env):
    """
    Kuhn Poker environment for SPIRAL training.
    
    Rules:
    - 3 cards: Jack (0), Queen (1), King (2)
    - Each player gets 1 card
    - Each player antes 1 chip
    - Player 1 acts first: Check or Bet
    - Player 2 then acts: Check, Call, or Fold
    - If both check, high card wins
    - If one bets and other calls, high card wins
    - If one bets and other folds, bettor wins
    
    Action space: [Check/Call=0, Bet=1, Fold=2]
    Observation space: [player_card, opponent_action, betting_round]
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    # Card values: Jack=0, Queen=1, King=2
    JACK, QUEEN, KING = 0, 1, 2
    CARDS = [JACK, QUEEN, KING]
    CARD_NAMES = ["J", "Q", "K"]
    
    # Actions
    CHECK_CALL, BET, FOLD = 0, 1, 2
    ACTION_NAMES = ["Check/Call", "Bet", "Fold"]
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Observation: [player_card, opponent_last_action, betting_round, pot_size]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(4,), dtype=np.int8
        )
        
        # Actions: Check/Call, Bet, Fold
        self.action_space = spaces.Discrete(3)
        
        self.render_mode = render_mode
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the game to initial state."""
        super().reset(seed=seed)
        
        # Deal cards
        cards = self.CARDS.copy()
        random.shuffle(cards)
        self.player1_card = cards[0]
        self.player2_card = cards[1]
        
        # Game state
        self.current_player = 1  # Player 1 starts
        self.pot = 2  # Each player antes 1
        self.player1_bet = 1  # Ante
        self.player2_bet = 1  # Ante
        self.game_over = False
        self.winner = None
        self.betting_round = 0
        self.actions_history = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Check/Call, 1=Bet, 2=Fold
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over:
            raise ValueError("Game is already over. Call reset() to start new game.")
        
        # Record action
        self.actions_history.append((self.current_player, action))
        
        # Process action
        if action == self.FOLD:
            # Current player folds, opponent wins
            self.game_over = True
            self.winner = 2 if self.current_player == 1 else 1
            reward = self._calculate_reward()
            
        elif action == self.BET:
            # Current player bets
            if self.current_player == 1:
                self.player1_bet += 1
                self.pot += 1
            else:
                self.player2_bet += 1
                self.pot += 1
            
            # Check if this ends the betting round
            if self.betting_round == 0:
                # First bet, opponent gets to act
                self.current_player = 2
                self.betting_round = 1
                reward = 0.0
            else:
                # Second bet (raise), go to showdown
                self.game_over = True
                self.winner = self._determine_winner_by_cards()
                reward = self._calculate_reward()
        
        else:  # CHECK_CALL
            if self.betting_round == 0:
                # First action is check
                if self.current_player == 1:
                    # Player 1 checks, player 2 acts
                    self.current_player = 2
                    self.betting_round = 1
                    reward = 0.0
                else:
                    # Player 2 checks after player 1 checked, showdown
                    self.game_over = True
                    self.winner = self._determine_winner_by_cards()
                    reward = self._calculate_reward()
            else:
                # This is a call
                if self.current_player == 2:
                    # Player 2 calls player 1's bet
                    self.player2_bet = self.player1_bet
                    self.pot = self.player1_bet + self.player2_bet
                    self.game_over = True
                    self.winner = self._determine_winner_by_cards()
                    reward = self._calculate_reward()
                else:
                    # Player 1 calls player 2's bet
                    self.player1_bet = self.player2_bet
                    self.pot = self.player1_bet + self.player2_bet
                    self.game_over = True
                    self.winner = self._determine_winner_by_cards()
                    reward = self._calculate_reward()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.game_over, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for the current player."""
        # Get current player's card
        player_card = self.player1_card if self.current_player == 1 else self.player2_card
        
        # Get opponent's last action (if any)
        opponent_last_action = -1
        if self.actions_history:
            for player, action in reversed(self.actions_history):
                if player != self.current_player:
                    opponent_last_action = action
                    break
        
        # Observation: [player_card, opponent_last_action, betting_round, pot_size]
        observation = np.array([
            player_card,
            opponent_last_action + 1,  # -1 becomes 0, 0 becomes 1, etc.
            self.betting_round,
            self.pot
        ], dtype=np.int8)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the game state."""
        return {
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
            "player1_card": self.player1_card,
            "player2_card": self.player2_card,
            "pot": self.pot,
            "betting_round": self.betting_round,
            "actions_history": self.actions_history.copy(),
            "valid_actions": self._get_valid_actions()
        }
    
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid actions."""
        if self.game_over:
            return []
        
        # All actions are always valid in Kuhn Poker
        return [self.CHECK_CALL, self.BET, self.FOLD]
    
    def _determine_winner_by_cards(self) -> int:
        """Determine winner by comparing cards."""
        if self.player1_card > self.player2_card:
            return 1
        else:
            return 2
    
    def _calculate_reward(self) -> float:
        """Calculate reward for the current player."""
        if not self.game_over:
            return 0.0
        
        if self.winner == self.current_player:
            # Won - get the pot minus what you put in
            if self.current_player == 1:
                return float(self.pot - self.player1_bet)
            else:
                return float(self.pot - self.player2_bet)
        else:
            # Lost - lose what you put in
            if self.current_player == 1:
                return float(-self.player1_bet)
            else:
                return float(-self.player2_bet)
    
    def render(self) -> Optional[np.ndarray]:
        """Render the game state."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """Print the game state to console."""
        print("\n" + "="*40)
        print("KUHN POKER")
        print("="*40)
        print(f"Player 1 Card: {self.CARD_NAMES[self.player1_card]}")
        print(f"Player 2 Card: {self.CARD_NAMES[self.player2_card]}")
        print(f"Pot: {self.pot}")
        print(f"Current Player: {self.current_player}")
        print(f"Betting Round: {self.betting_round}")
        
        if self.actions_history:
            print("Actions:")
            for player, action in self.actions_history:
                print(f"  Player {player}: {self.ACTION_NAMES[action]}")
        
        if self.game_over:
            print(f"Game Over! Winner: Player {self.winner}")
        print("="*40)
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render as RGB array for visualization."""
        # Simple RGB representation (placeholder)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Color based on current player's card
        if self.current_player == 1:
            card_value = self.player1_card
        else:
            card_value = self.player2_card
        
        # Different colors for different cards
        if card_value == self.JACK:
            rgb[:, :] = [255, 0, 0]  # Red for Jack
        elif card_value == self.QUEEN:
            rgb[:, :] = [0, 255, 0]  # Green for Queen
        else:  # King
            rgb[:, :] = [0, 0, 255]  # Blue for King
        
        return rgb
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions (1 for valid, 0 for invalid)."""
        mask = np.zeros(3, dtype=np.int8)
        for action in self._get_valid_actions():
            mask[action] = 1
        return mask


def create_kuhn_poker_env() -> KuhnPokerEnv:
    """Factory function to create a Kuhn Poker environment."""
    return KuhnPokerEnv()


if __name__ == "__main__":
    # Test the environment
    env = KuhnPokerEnv(render_mode="human")
    
    # Play a simple game
    obs, info = env.reset()
    print("Initial state:")
    env.render()
    
    # Simulate some moves
    while not env.game_over:
        valid_actions = env._get_valid_actions()
        action = random.choice(valid_actions)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nPlayer {env.current_player if not env.game_over else 'Previous'} action: {env.ACTION_NAMES[action]}")
        print(f"Reward: {reward}")
        env.render()
        
        if terminated:
            print(f"Game terminated! Final reward: {reward}")
            break 