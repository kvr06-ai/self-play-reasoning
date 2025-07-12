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
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

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


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
quantization_params = config['model'].get('quantization', {})

# Create BitsAndBytesConfig if quantization is enabled
if quantization_params and quantization_params.get('load_in_4bit'):
    compute_dtype_str = quantization_params.get("bnb_4bit_compute_dtype", "float16")

    if compute_dtype_str == "bfloat16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16  # Default to float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quantization_params.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quantization_params.get("bnb_4bit_use_double_quant", True),
    )
    # Using device_map="auto" is recommended for multi-GPU setups and large models
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
else:
    # Fallback for no quantization
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_name)


def generate_reasoning(prompt):
    """Generate reasoning trace using Qwen model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="SPIRAL: Interactive Reasoning Game Simulator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎮 SPIRAL: Interactive Reasoning Game Simulator")
        
        if GAMES_AVAILABLE:
            gr.Markdown("**Demo Version** - Experience zero-sum games with AI! Full reasoning capabilities coming soon. Learn how AI makes decisions in competitive scenarios.")
            
            # TicTacToe specific functions
            def get_tictactoe_board_html():
                """Get current TicTacToe board as HTML with emojis."""
                board = tictactoe_env.board
                html = '<table style="border: 1px solid black; text-align: center; font-size: 24px;">'
                for row in range(3):
                    html += '<tr>'
                    for col in range(3):
                        cell = board[row, col]
                        if cell == 1:
                            content = '❌'
                        elif cell == -1:
                            content = '⭕'
                        else:
                            content = f'{row*3 + col}'
                        html += f'<td style="border: 1px solid black; width: 50px; height: 50px;">{content}</td>'
                    html += '</tr>'
                html += '</table>'
                return html
            
            def get_valid_tictactoe_positions():
                """Get list of valid position strings."""
                return [str(i) for i in tictactoe_env._get_valid_actions()]
            
            ttt_stats = gr.State({'wins': 0, 'losses': 0, 'draws': 0})
            
            def play_tictactoe(position, stats):
                """Play a TicTacToe move."""
                if tictactoe_env.game_over:
                    return get_tictactoe_board_html(), "Game is over! Click 'New Game' to start again.", "", stats, get_valid_tictactoe_positions()
                
                try:
                    position = int(position)
                    if position < 0 or position > 8:
                        raise ValueError("Invalid position")
                    
                    # Human move
                    obs, reward, terminated, truncated, info = tictactoe_env.step(position)
                    
                    if terminated:
                        winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "Draw"
                        if winner == "You": stats['wins'] += 1
                        elif winner == "AI": stats['losses'] += 1
                        else: stats['draws'] += 1
                        return get_tictactoe_board_html(), f"Game Over! {winner} won!", f"Final reward: {reward}", stats, []
                    
                    # AI move
                    valid_actions = tictactoe_env._get_valid_actions()
                    ai_action = random.choice(valid_actions)  # Still random for now; integrate policy later
                    reasoning_prompt = f"In TicTacToe, board state: {tictactoe_env.board.flatten().tolist()}. Valid moves: {valid_actions}. Explain why to choose one randomly as placeholder."
                    reasoning = generate_reasoning(reasoning_prompt)
                    obs, reward, terminated, truncated, info = tictactoe_env.step(ai_action)
                    
                    if terminated:
                        winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "Draw"
                        if winner == "You": stats['wins'] += 1
                        elif winner == "AI": stats['losses'] += 1
                        else: stats['draws'] += 1
                        return get_tictactoe_board_html(), f"Game Over! {winner} won! AI played {ai_action}.", reasoning, stats, []
                    else:
                        return get_tictactoe_board_html(), f"AI played position {ai_action}. Your turn!", reasoning, stats, get_valid_tictactoe_positions()
                    
                except Exception as e:
                    return get_tictactoe_board_html(), f"Error: {str(e)}", "", stats, get_valid_tictactoe_positions()
            
            def reset_tictactoe(stats):
                """Reset TicTacToe game."""
                tictactoe_env.reset()
                return get_tictactoe_board_html(), "New game started! You are ❌ (X). Choose a position from the dropdown.", "AI will show its reasoning here...", stats, get_valid_tictactoe_positions()
            
            def get_kuhn_poker_state_html():
                """Get current Kuhn Poker state as HTML."""
                card = ['J', 'Q', 'K'][kuhn_env.player1_card]
                html = f"<div style='font-size: 18px;'><p>🃏 Your Card: <strong>{card}</strong></p>"
                html += f"<p>💰 Pot: <strong>{kuhn_env.pot}</strong></p>"
                html += f"<p>🎯 Current Player: <strong>{kuhn_env.current_player}</strong></p>"
                html += f"<p>🔄 Betting Round: <strong>{kuhn_env.betting_round}</strong></p>"
                
                if kuhn_env.actions_history:
                    html += "<p>📋 Actions:</p><ul>"
                    for player, action in kuhn_env.actions_history:
                        action_name = ["Check/Call", "Bet", "Fold"][action]
                        html += f"<li>Player {player}: {action_name}</li>"
                    html += "</ul>"
                
                html += "</div>"
                return html
            
            kuhn_stats = gr.State({'wins': 0, 'losses': 0, 'draws': 0})
            
            def play_kuhn_poker(action_name, stats):
                """Play a Kuhn Poker move."""
                if kuhn_env.game_over:
                    return get_kuhn_poker_state_html(), "Game is over! Click 'New Game' to start again.", "", stats
                
                try:
                    action_map = {"Check/Call": 0, "Bet": 1, "Fold": 2}
                    if action_name not in action_map:
                        raise ValueError("Invalid action")
                    
                    action = action_map[action_name]
                    
                    # Human move
                    obs, reward, terminated, truncated, info = kuhn_env.step(action)
                    
                    if terminated:
                        winner = "You" if kuhn_env.winner == 1 else "AI" if kuhn_env.winner == -1 else "Draw"
                        if winner == "You": stats['wins'] += 1
                        elif winner == "AI": stats['losses'] += 1
                        else: stats['draws'] += 1
                        return get_kuhn_poker_state_html(), f"Game Over! {winner} won! Pot: {kuhn_env.pot}", f"Your final reward: {reward}", stats
                    
                    # AI move
                    valid_actions = kuhn_env._get_valid_actions()
                    ai_action = random.choice(valid_actions)
                    ai_action_name = ["Check/Call", "Bet", "Fold"][ai_action]
                    reasoning_prompt = f"In Kuhn Poker, my card: {kuhn_env.player2_card}, history: {kuhn_env.actions_history}. Valid actions: {valid_actions}. Explain choice."
                    reasoning = generate_reasoning(reasoning_prompt)
                    obs, reward, terminated, truncated, info = kuhn_env.step(ai_action)
                    
                    if terminated:
                        winner = "You" if kuhn_env.winner == 1 else "AI" if kuhn_env.winner == -1 else "Draw"
                        if winner == "You": stats['wins'] += 1
                        elif winner == "AI": stats['losses'] += 1
                        else: stats['draws'] += 1
                        return get_kuhn_poker_state_html(), f"AI chose {ai_action_name}. Game Over! {winner} won! Pot: {kuhn_env.pot}", reasoning, stats
                    else:
                        return get_kuhn_poker_state_html(), f"AI chose {ai_action_name}. Your turn!", reasoning, stats
                    
                except Exception as e:
                    return get_kuhn_poker_state_html(), f"Error: {str(e)}", "", stats
            
            def reset_kuhn_poker(stats):
                """Reset Kuhn Poker game."""
                kuhn_env.reset()
                card = ['J', 'Q', 'K'][kuhn_env.player1_card]
                return get_kuhn_poker_state_html(), "New game started! You are Player 1. Choose your action.", f"Your card: {card}", stats
            
            with gr.Tabs():
                # TicTacToe Tab
                with gr.TabItem("🎯 TicTacToe"):
                    gr.Markdown("### Play TicTacToe against AI\nYou are ❌ (X) and go first. Get 3 in a row to win! **How AI Thinks**: AI will analyze the board and explain its moves (random for now; full reasoning soon).\nPositions: Top-left=0, bottom-right=8.")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            ttt_board = gr.HTML(
                                label="Game Board",
                                value=get_tictactoe_board_html()
                            )
                            
                        with gr.Column(scale=1):
                            ttt_position = gr.Dropdown(
                                label="Your Move (Valid Positions)",
                                choices=get_valid_tictactoe_positions()
                            )
                            with gr.Row():
                                ttt_play_btn = gr.Button("Play Move", variant="primary")
                                ttt_reset_btn = gr.Button("New Game", variant="secondary")
                            ttt_stats_display = gr.Markdown(value="Wins: 0 | Losses: 0 | Draws: 0")
                    
                    ttt_message = gr.Textbox(
                        label="Game Status",
                        value="Choose a position to start!",
                        lines=2,
                        interactive=False
                    )
                    
                    ttt_reasoning = gr.Textbox(
                        label="AI Reasoning",
                        value="AI will explain its thought process here...",
                        lines=3,
                        interactive=False
                    )
                    
                    ttt_play_btn.click(
                        fn=play_tictactoe,
                        inputs=[ttt_position, ttt_stats],
                        outputs=[ttt_board, ttt_message, ttt_reasoning, ttt_stats, ttt_position]
                    )
                    ttt_reset_btn.click(
                        fn=reset_tictactoe,
                        inputs=[ttt_stats],
                        outputs=[ttt_board, ttt_message, ttt_reasoning, ttt_stats, ttt_position]
                    )
                    # Update stats display on changes
                    ttt_stats.change(
                        fn=lambda s: f"Wins: {s['wins']} | Losses: {s['losses']} | Draws: {s['draws']}",
                        inputs=ttt_stats,
                        outputs=ttt_stats_display
                    )
                
                # Kuhn Poker Tab
                with gr.TabItem("🃏 Kuhn Poker"):
                    gr.Markdown("### Play Kuhn Poker against AI\nSimplified poker with J/Q/K cards. You ante 1 chip each. Higher card wins if no fold. **How AI Thinks**: AI evaluates card strength and bets (random now; strategic soon).")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            kuhn_state = gr.HTML(
                                label="Game State",
                                value=get_kuhn_poker_state_html()
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
                            kuhn_stats_display = gr.Markdown(value="Wins: 0 | Losses: 0 | Draws: 0")
                    
                    kuhn_message = gr.Textbox(
                        label="Game Status",
                        value="Choose your action!",
                        lines=2,
                        interactive=False
                    )
                    
                    kuhn_reasoning = gr.Textbox(
                        label="AI Reasoning",
                        value="AI will explain its thought process here...",
                        lines=3,
                        interactive=False
                    )
                    
                    kuhn_play_btn.click(
                        fn=play_kuhn_poker,
                        inputs=[kuhn_action, kuhn_stats],
                        outputs=[kuhn_state, kuhn_message, kuhn_reasoning, kuhn_stats]
                    )
                    kuhn_reset_btn.click(
                        fn=reset_kuhn_poker,
                        inputs=[kuhn_stats],
                        outputs=[kuhn_state, kuhn_message, kuhn_reasoning, kuhn_stats]
                    )
                    kuhn_stats.change(
                        fn=lambda s: f"Wins: {s['wins']} | Losses: {s['losses']} | Draws: {s['draws']}",
                        inputs=kuhn_stats,
                        outputs=kuhn_stats_display
                    )
                
                # New Transfer Test Tab (stub)
                with gr.TabItem("🔬 Transfer Test"):
                    gr.Markdown("### Test AI Reasoning on Non-Game Tasks\n(Coming Soon) Enter a math problem or logic puzzle to see transferred reasoning from game training.")
                    transfer_input = gr.Textbox(label="Input Prompt", placeholder="E.g., 'Solve: 2x + 3 = 7'")
                    transfer_output = gr.Textbox(label="AI Response", interactive=False)
                    transfer_btn = gr.Button("Test")
                    
                    def transfer_test(input):
                        cot_prompt = f"Solve step-by-step: {input}"
                        return generate_reasoning(cot_prompt)
                    
                    transfer_btn.click(fn=transfer_test, inputs=transfer_input, outputs=transfer_output)
            
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
            gr.Markdown("**New in this version:** Visual boards, stats tracking, and transfer test stub!")
        
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
