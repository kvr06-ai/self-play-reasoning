"""
SPIRAL: Interactive Reasoning Game Simulator

Main Gradio application for the SPIRAL demo on Hugging Face Spaces.
"""

import gradio as gr
import numpy as np
import random
import os
import sys
import traceback

# Add src to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

print(f"🔍 Current directory: {current_dir}")
print(f"🔍 Source path: {src_path}")
print(f"🔍 Python path: {sys.path[:3]}")  # Show first 3 entries

# Check if src directory exists
if os.path.exists(src_path):
    print(f"✅ Source directory exists: {src_path}")
    games_path = os.path.join(src_path, 'games')
    if os.path.exists(games_path):
        print(f"✅ Games directory exists: {games_path}")
        print(f"📁 Games directory contents: {os.listdir(games_path)}")
    else:
        print(f"❌ Games directory not found: {games_path}")
else:
    print(f"❌ Source directory not found: {src_path}")

# Try multiple import approaches
GAMES_AVAILABLE = False
tictactoe_env = None
kuhn_env = None

try:
    # Method 1: Direct import from games module
    print("🔄 Attempting Method 1: Direct import from games")
    from games import TicTacToeEnv, KuhnPokerEnv
    print("✅ Method 1 successful: Imported from games module")
    GAMES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Method 1 failed: {e}")
    
    try:
        # Method 2: Import from src.games
        print("🔄 Attempting Method 2: Import from src.games")
        from src.games import TicTacToeEnv, KuhnPokerEnv
        print("✅ Method 2 successful: Imported from src.games")
        GAMES_AVAILABLE = True
    except ImportError as e:
        print(f"❌ Method 2 failed: {e}")
        
        try:
            # Method 3: Direct file imports
            print("🔄 Attempting Method 3: Direct file imports")
            sys.path.insert(0, games_path)
            from tictactoe import TicTacToeEnv
            from kuhn_poker import KuhnPokerEnv
            print("✅ Method 3 successful: Direct file imports")
            GAMES_AVAILABLE = True
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
            print("📋 Full traceback:", traceback.format_exc())

if GAMES_AVAILABLE:
    print("🎮 Game modules successfully imported!")
    try:
        # Test instantiation
        tictactoe_env = TicTacToeEnv()
        kuhn_env = KuhnPokerEnv()
        print("✅ Game environments created successfully")
    except Exception as e:
        print(f"❌ Error creating game environments: {e}")
        print("📋 Full traceback:", traceback.format_exc())
        GAMES_AVAILABLE = False
else:
    print("❌ All import methods failed - using fallback interface")


def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="SPIRAL: Interactive Reasoning Game Simulator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 SPIRAL: Interactive Reasoning Game Simulator")
        
        if GAMES_AVAILABLE:
            gr.Markdown("**Demo Version** - Experience zero-sum games with AI! Full reasoning capabilities coming soon.")
            
            def get_tictactoe_board():
                """Get current TicTacToe board as string."""
                board = tictactoe_env.board
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
            
            def play_tictactoe(position):
                """Play a TicTacToe move."""
                if tictactoe_env.game_over:
                    return get_tictactoe_board(), "Game is over! Click 'New Game' to start again.", ""
                
                try:
                    position = int(position)
                    if position < 0 or position > 8:
                        return get_tictactoe_board(), "Invalid position! Choose 0-8.", ""
                    
                    # Human move
                    obs, reward, terminated, truncated, info = tictactoe_env.step(position)
                    
                    if terminated:
                        winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "No one"
                        return get_tictactoe_board(), f"Game Over! {winner} won!", f"Final reward: {reward}"
                    
                    # AI move (random for now)
                    if not tictactoe_env.game_over:
                        valid_actions = tictactoe_env._get_valid_actions()
                        if valid_actions:
                            ai_action = random.choice(valid_actions)
                            obs, reward, terminated, truncated, info = tictactoe_env.step(ai_action)
                            
                            if terminated:
                                winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "No one"
                                return get_tictactoe_board(), f"Game Over! {winner} won!", f"AI played position {ai_action}. Final reward: {reward}"
                            else:
                                return get_tictactoe_board(), f"AI played position {ai_action}. Your turn!", f"AI reasoning: Chose position {ai_action} randomly"
                    
                    return get_tictactoe_board(), "Your turn!", ""
                    
                except ValueError:
                    return get_tictactoe_board(), "Please enter a valid number (0-8).", ""
                except Exception as e:
                    return get_tictactoe_board(), f"Error: {str(e)}", ""
            
            def reset_tictactoe():
                """Reset TicTacToe game."""
                tictactoe_env.reset()
                return get_tictactoe_board(), "New game started! You are X. Choose a position (0-8).", ""
            
            def get_kuhn_poker_state():
                """Get current Kuhn Poker state as string."""
                state = f"🃏 Your Card: {['J', 'Q', 'K'][kuhn_env.player1_card]}\n"
                state += f"💰 Pot: {kuhn_env.pot}\n"
                state += f"🎯 Current Player: {kuhn_env.current_player}\n"
                state += f"🔄 Betting Round: {kuhn_env.betting_round}\n"
                
                if kuhn_env.actions_history:
                    state += "\n📋 Actions:\n"
                    for player, action in kuhn_env.actions_history:
                        action_name = ["Check/Call", "Bet", "Fold"][action]
                        state += f"   Player {player}: {action_name}\n"
                
                return state
            
            def play_kuhn_poker(action_name):
                """Play a Kuhn Poker move."""
                if kuhn_env.game_over:
                    return get_kuhn_poker_state(), "Game is over! Click 'New Game' to start again.", ""
                
                try:
                    # Map action name to action number
                    action_map = {"Check/Call": 0, "Bet": 1, "Fold": 2}
                    if action_name not in action_map:
                        return get_kuhn_poker_state(), "Invalid action!", ""
                    
                    action = action_map[action_name]
                    
                    # Human move
                    obs, reward, terminated, truncated, info = kuhn_env.step(action)
                    
                    if terminated:
                        winner = "You" if kuhn_env.winner == 1 else "AI"
                        return get_kuhn_poker_state(), f"Game Over! {winner} won! Pot: {kuhn_env.pot}", f"Your final reward: {reward}"
                    
                    # AI move (random for now)
                    if not kuhn_env.game_over:
                        valid_actions = kuhn_env._get_valid_actions()
                        ai_action = random.choice(valid_actions)
                        ai_action_name = ["Check/Call", "Bet", "Fold"][ai_action]
                        
                        obs, reward, terminated, truncated, info = kuhn_env.step(ai_action)
                        
                        if terminated:
                            winner = "You" if kuhn_env.winner == 1 else "AI"
                            return get_kuhn_poker_state(), f"AI chose {ai_action_name}. Game Over! {winner} won! Pot: {kuhn_env.pot}", f"AI reasoning: Chose {ai_action_name} randomly. Your final reward: {reward}"
                        else:
                            return get_kuhn_poker_state(), f"AI chose {ai_action_name}. Your turn!", f"AI reasoning: Chose {ai_action_name} randomly"
                    
                    return get_kuhn_poker_state(), "Your turn!", ""
                    
                except Exception as e:
                    return get_kuhn_poker_state(), f"Error: {str(e)}", ""
            
            def reset_kuhn_poker():
                """Reset Kuhn Poker game."""
                kuhn_env.reset()
                return get_kuhn_poker_state(), "New game started! You are Player 1. Choose your action.", f"Your card: {['J', 'Q', 'K'][kuhn_env.player1_card]}"
            
            with gr.Tabs():
                # TicTacToe Tab
                with gr.TabItem("🎯 TicTacToe"):
                    gr.Markdown("### Play TicTacToe against AI")
                    gr.Markdown("You are **X** and go first. Enter a position (0-8) to make your move.")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            ttt_board = gr.Textbox(
                                label="Game Board",
                                value=get_tictactoe_board(),
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
                        fn=play_tictactoe,
                        inputs=[ttt_position],
                        outputs=[ttt_board, ttt_message, ttt_reasoning]
                    )
                    
                    ttt_reset_btn.click(
                        fn=reset_tictactoe,
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
                                value=get_kuhn_poker_state(),
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
                        fn=play_kuhn_poker,
                        inputs=[kuhn_action],
                        outputs=[kuhn_state, kuhn_message, kuhn_reasoning]
                    )
                    
                    kuhn_reset_btn.click(
                        fn=reset_kuhn_poker,
                        outputs=[kuhn_state, kuhn_message, kuhn_reasoning]
                    )
                
        else:
            # Fallback interface when games don't load
            gr.Markdown("⚠️ **Game modules could not be loaded.** Showing diagnostic information.")
            gr.Markdown("This usually happens when dependencies are still installing on HF Spaces.")
            
            # Show diagnostic info
            gr.Markdown("### 🔍 Diagnostic Information:")
            gr.Markdown(f"- Current directory: `{current_dir}`")
            gr.Markdown(f"- Source path: `{src_path}`")
            gr.Markdown(f"- Source directory exists: `{os.path.exists(src_path)}`")
            
            if os.path.exists(src_path):
                games_path = os.path.join(src_path, 'games')
                gr.Markdown(f"- Games directory exists: `{os.path.exists(games_path)}`")
                if os.path.exists(games_path):
                    gr.Markdown(f"- Games directory contents: `{os.listdir(games_path)}`")
            
            # Simple demo interface
            with gr.Row():
                simple_input = gr.Textbox(label="Test Input", placeholder="Enter something...")
                simple_output = gr.Textbox(label="Output", interactive=False)
            
            def simple_echo(text):
                return f"Echo: {text} (Game modules will be available once dependencies install)"
            
            simple_input.submit(fn=simple_echo, inputs=[simple_input], outputs=[simple_output])
        
        # About Tab (always available)
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
        
        if GAMES_AVAILABLE:
            gr.Markdown("---")
            gr.Markdown("🚧 **This is a development preview.** Full SPIRAL training and reasoning capabilities will be added in the next update!")
        else:
            gr.Markdown("---")
            gr.Markdown("🔄 **Dependencies are loading.** Check the diagnostic info above and refresh in a few minutes!")
        
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
