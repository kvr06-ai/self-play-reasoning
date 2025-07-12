"""
SPIRAL Interactive Reasoning Game Simulator

Main Gradio application for the SPIRAL demo on Hugging Face Spaces.
"""

import gradio as gr
import numpy as np
import random
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from games import TicTacToeEnv, KuhnPokerEnv
from games.game_utils import get_available_games, get_game_info


class GameInterface:
    """Interface for managing game interactions."""
    
    def __init__(self):
        self.tictactoe_env = None
        self.kuhn_env = None
        self.reset_games()
    
    def reset_games(self):
        """Reset both game environments."""
        self.tictactoe_env = TicTacToeEnv()
        self.kuhn_env = KuhnPokerEnv()
        self.tictactoe_env.reset()
        self.kuhn_env.reset()
    
    def play_tictactoe(self, position):
        """Play a TicTacToe move."""
        if self.tictactoe_env.game_over:
            return self.get_tictactoe_board(), "Game is over! Click 'New Game' to start again.", ""
        
        try:
            position = int(position)
            if position < 0 or position > 8:
                return self.get_tictactoe_board(), "Invalid position! Choose 0-8.", ""
            
            # Human move
            obs, reward, terminated, truncated, info = self.tictactoe_env.step(position)
            
            if terminated:
                winner = "You" if self.tictactoe_env.winner == 1 else "AI" if self.tictactoe_env.winner == -1 else "No one"
                return self.get_tictactoe_board(), f"Game Over! {winner} won!", f"Final reward: {reward}"
            
            # AI move (random for now)
            if not self.tictactoe_env.game_over:
                valid_actions = self.tictactoe_env._get_valid_actions()
                if valid_actions:
                    ai_action = random.choice(valid_actions)
                    obs, reward, terminated, truncated, info = self.tictactoe_env.step(ai_action)
                    
                    if terminated:
                        winner = "You" if self.tictactoe_env.winner == 1 else "AI" if self.tictactoe_env.winner == -1 else "No one"
                        return self.get_tictactoe_board(), f"Game Over! {winner} won!", f"AI played position {ai_action}. Final reward: {reward}"
                    else:
                        return self.get_tictactoe_board(), f"AI played position {ai_action}. Your turn!", f"AI reasoning: Chose position {ai_action} randomly"
            
            return self.get_tictactoe_board(), "Your turn!", ""
            
        except ValueError:
            return self.get_tictactoe_board(), "Please enter a valid number (0-8).", ""
        except Exception as e:
            return self.get_tictactoe_board(), f"Error: {str(e)}", ""
    
    def reset_tictactoe(self):
        """Reset TicTacToe game."""
        self.tictactoe_env.reset()
        return self.get_tictactoe_board(), "New game started! You are X. Choose a position (0-8).", ""
    
    def get_tictactoe_board(self):
        """Get current TicTacToe board as string."""
        board = self.tictactoe_env.board
        display = ""
        for row in range(3):
            for col in range(3):
                cell = board[row, col]
                if cell == 1:
                    display += " X "
                elif cell == -1:
                    display += " O "
                else:
                    display += f" {row*3 + col} "
                if col < 2:
                    display += "|"
            display += "\n"
            if row < 2:
                display += "-----------\n"
        return display
    
    def play_kuhn_poker(self, action_name):
        """Play a Kuhn Poker move."""
        if self.kuhn_env.game_over:
            return self.get_kuhn_poker_state(), "Game is over! Click 'New Game' to start again.", ""
        
        try:
            # Map action name to action number
            action_map = {"Check/Call": 0, "Bet": 1, "Fold": 2}
            if action_name not in action_map:
                return self.get_kuhn_poker_state(), "Invalid action!", ""
            
            action = action_map[action_name]
            
            # Human move
            obs, reward, terminated, truncated, info = self.kuhn_env.step(action)
            
            if terminated:
                winner = "You" if self.kuhn_env.winner == 1 else "AI"
                return self.get_kuhn_poker_state(), f"Game Over! {winner} won! Pot: {self.kuhn_env.pot}", f"Your final reward: {reward}"
            
            # AI move (random for now)
            if not self.kuhn_env.game_over:
                valid_actions = self.kuhn_env._get_valid_actions()
                ai_action = random.choice(valid_actions)
                ai_action_name = ["Check/Call", "Bet", "Fold"][ai_action]
                
                obs, reward, terminated, truncated, info = self.kuhn_env.step(ai_action)
                
                if terminated:
                    winner = "You" if self.kuhn_env.winner == 1 else "AI"
                    return self.get_kuhn_poker_state(), f"AI chose {ai_action_name}. Game Over! {winner} won! Pot: {self.kuhn_env.pot}", f"AI reasoning: Chose {ai_action_name} randomly. Your final reward: {reward}"
                else:
                    return self.get_kuhn_poker_state(), f"AI chose {ai_action_name}. Your turn!", f"AI reasoning: Chose {ai_action_name} randomly"
            
            return self.get_kuhn_poker_state(), "Your turn!", ""
            
        except Exception as e:
            return self.get_kuhn_poker_state(), f"Error: {str(e)}", ""
    
    def reset_kuhn_poker(self):
        """Reset Kuhn Poker game."""
        self.kuhn_env.reset()
        return self.get_kuhn_poker_state(), "New game started! You are Player 1. Choose your action.", f"Your card: {['J', 'Q', 'K'][self.kuhn_env.player1_card]}"
    
    def get_kuhn_poker_state(self):
        """Get current Kuhn Poker state as string."""
        state = f"🃏 Your Card: {['J', 'Q', 'K'][self.kuhn_env.player1_card]}\n"
        state += f"💰 Pot: {self.kuhn_env.pot}\n"
        state += f"🎯 Current Player: {self.kuhn_env.current_player}\n"
        state += f"🔄 Betting Round: {self.kuhn_env.betting_round}\n"
        
        if self.kuhn_env.actions_history:
            state += "\n📋 Actions:\n"
            for player, action in self.kuhn_env.actions_history:
                action_name = ["Check/Call", "Bet", "Fold"][action]
                state += f"   Player {player}: {action_name}\n"
        
        return state


# Create game interface
game_interface = GameInterface()


def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="SPIRAL: Interactive Reasoning Game Simulator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 SPIRAL: Interactive Reasoning Game Simulator")
        gr.Markdown("**Demo Version** - Experience zero-sum games with AI! Full reasoning capabilities coming soon.")
        
        with gr.Tabs():
            # TicTacToe Tab
            with gr.TabItem("🎯 TicTacToe"):
                gr.Markdown("### Play TicTacToe against AI")
                gr.Markdown("You are **X** and go first. Enter a position (0-8) to make your move.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        ttt_board = gr.Textbox(
                            label="Game Board",
                            value=game_interface.get_tictactoe_board(),
                            lines=6,
                            interactive=False,
                            elem_id="ttt-board"
                        )
                        
                    with gr.Column(scale=1):
                        ttt_position = gr.Textbox(
                            label="Your Move (0-8)",
                            placeholder="Enter position number",
                            lines=1
                        )
                        
                        with gr.Row():
                            ttt_play_btn = gr.Button("Play Move", variant="primary")
                            ttt_reset_btn = gr.Button("New Game", variant="secondary")
                
                ttt_message = gr.Textbox(
                    label="Game Status",
                    value="Choose a position (0-8) to start!",
                    lines=2,
                    interactive=False
                )
                
                ttt_reasoning = gr.Textbox(
                    label="AI Reasoning",
                    value="AI will show its reasoning here...",
                    lines=2,
                    interactive=False
                )
                
                ttt_play_btn.click(
                    fn=game_interface.play_tictactoe,
                    inputs=[ttt_position],
                    outputs=[ttt_board, ttt_message, ttt_reasoning]
                )
                
                ttt_reset_btn.click(
                    fn=game_interface.reset_tictactoe,
                    outputs=[ttt_board, ttt_message, ttt_reasoning]
                )
            
            # Kuhn Poker Tab
            with gr.TabItem("🃏 Kuhn Poker"):
                gr.Markdown("### Play Kuhn Poker against AI")
                gr.Markdown("Simple poker with 3 cards (J, Q, K). You are Player 1.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        kuhn_state = gr.Textbox(
                            label="Game State",
                            value=game_interface.get_kuhn_poker_state(),
                            lines=8,
                            interactive=False
                        )
                        
                    with gr.Column(scale=1):
                        kuhn_action = gr.Dropdown(
                            label="Your Action",
                            choices=["Check/Call", "Bet", "Fold"],
                            value="Check/Call"
                        )
                        
                        with gr.Row():
                            kuhn_play_btn = gr.Button("Play Action", variant="primary")
                            kuhn_reset_btn = gr.Button("New Game", variant="secondary")
                
                kuhn_message = gr.Textbox(
                    label="Game Status",
                    value="Choose your action!",
                    lines=2,
                    interactive=False
                )
                
                kuhn_reasoning = gr.Textbox(
                    label="AI Reasoning",
                    value="AI will show its reasoning here...",
                    lines=2,
                    interactive=False
                )
                
                kuhn_play_btn.click(
                    fn=game_interface.play_kuhn_poker,
                    inputs=[kuhn_action],
                    outputs=[kuhn_state, kuhn_message, kuhn_reasoning]
                )
                
                kuhn_reset_btn.click(
                    fn=game_interface.reset_kuhn_poker,
                    outputs=[kuhn_state, kuhn_message, kuhn_reasoning]
                )
            
            # About Tab
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ### About SPIRAL
                
                This is a **demo version** of the SPIRAL methodology: *"Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning."*
                
                **Current Features:**
                - 🎯 **TicTacToe**: Play against a random AI opponent
                - 🃏 **Kuhn Poker**: Experience simplified poker gameplay
                - 🎮 **Interactive Games**: Real-time game state updates
                
                **Coming Soon:**
                - 🧠 **SPIRAL-trained AI**: Opponents trained via self-play
                - 📊 **Reasoning Traces**: See step-by-step AI decision-making
                - 🔬 **Transfer Learning**: Test AI reasoning on math problems
                - 📈 **Performance Metrics**: Track AI improvement over time
                
                **Game Rules:**
                
                **TicTacToe:**
                - 3x3 grid, get 3 in a row to win
                - You are X, AI is O
                - Numbers 0-8 represent board positions
                
                **Kuhn Poker:**
                - 3 cards: Jack (lowest), Queen, King (highest)
                - Each player gets 1 card, antes 1 chip
                - Actions: Check/Call, Bet (+1 chip), Fold
                - Higher card wins if both call/check
                
                **Technical Details:**
                - Built with Gymnasium environments
                - Gradio web interface
                - Ready for SPIRAL training integration
                """)
        
        gr.Markdown("---")
        gr.Markdown("🚧 **This is a development preview.** Full SPIRAL training and reasoning capabilities will be added in the next update!")
        
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 