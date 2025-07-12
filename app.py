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
import spaces

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
        # kuhn_env = KuhnPokerEnv() # No longer needed
        print("✅ Game environment created successfully")
    except Exception as e:
        print(f"❌ Error creating game environment: {e}")
        print("📋 Full traceback:", traceback.format_exc())
        GAMES_AVAILABLE = False
else:
    print("❌ All import methods failed - using fallback interface")

# Initialize model and tokenizer as global variables
model = None
tokenizer = None

def generate_reasoning(prompt):
    """Generate reasoning trace using Qwen model."""
    global model, tokenizer
    if model is None or tokenizer is None:
        return "Error: Model not loaded. Please wait for the GPU to be ready."
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=150, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def create_interface():
    """Create the main Gradio interface."""
    
    # Custom CSS to style the TicTacToe board
    css = """
        #ttt-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 5px;
            max-width: 300px;
            margin: auto;
        }
        #ttt-grid .gr-button {
            aspect-ratio: 1 / 1;
            font-size: 24px !important;
            font-weight: bold;
            height: 100px !important;
            min-width: 100px !important;
        }
    """
    
    with gr.Blocks(title="SPIRAL: Interactive Reasoning Game Simulator", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# 🎮 SPIRAL: Interactive Reasoning Game Simulator")
        gr.Markdown("Play TicTacToe against an AI, see its step-by-step reasoning, and learn how it thinks!")

        if GAMES_AVAILABLE:
            
            def update_board_buttons():
                """Create a list of gr.Button updates from the current board state."""
                updates = []
                for i in range(9):
                    row, col = divmod(i, 3)
                    cell = tictactoe_env.board[row, col]
                    val = ""
                    interactive = True
                    if cell == 1:
                        val = '❌'
                        interactive = False
                    elif cell == -1:
                        val = '⭕'
                        interactive = False
                    
                    if tictactoe_env.game_over:
                        interactive = False

                    updates.append(gr.Button(value=val, interactive=interactive))
                return updates

            # TicTacToe specific functions (no longer need get_tictactoe_board_html)
            
            ttt_stats = gr.State({'wins': 0, 'losses': 0, 'draws': 0})
            
            def minimax(board, player):
                """Minimax algorithm to find the best move."""
                
                # Base cases
                winner = tictactoe_env._check_winner()
                if winner == 1: # Human wins
                    return -10, None
                elif winner == -1: # AI wins
                    return 10, None
                elif tictactoe_env._is_draw():
                    return 0, None

                best_move = None
                if player == -1: # AI is player -1 (O), maximizing player
                    best_score = -float('inf')
                    for move in tictactoe_env._get_valid_actions():
                        row, col = divmod(move, 3)
                        board[row, col] = -1
                        score, _ = minimax(board.copy(), 1)
                        board[row, col] = 0 # Undo move
                        if score > best_score:
                            best_score = score
                            best_move = move
                else: # Human is player 1 (X), minimizing player
                    best_score = float('inf')
                    for move in tictactoe_env._get_valid_actions():
                        row, col = divmod(move, 3)
                        board[row, col] = 1
                        score, _ = minimax(board.copy(), -1)
                        board[row, col] = 0 # Undo move
                        if score < best_score:
                            best_score = score
                            best_move = move
                return best_score, best_move

            def play_tictactoe(position, stats):
                """Play a TicTacToe move and yield updates for the button grid."""
                if tictactoe_env.game_over:
                    yield *update_board_buttons(), "Game is over! Click 'New Game' to start again.", "", stats
                    return

                try:
                    position = int(position)
                    
                    # Human move
                    tictactoe_env.step(position)
                    
                    if tictactoe_env.game_over:
                        winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "Draw"
                        if winner == "You": stats['wins'] += 1
                        elif winner == "AI": stats['losses'] += 1
                        else: stats['draws'] += 1
                        yield *update_board_buttons(), f"Game Over! {winner} won!", "", stats
                        return

                    # Show "thinking" indicator
                    yield *update_board_buttons(), "AI is thinking...", "🧠...", stats

                    # AI move
                    _, ai_action = minimax(tictactoe_env.board.copy(), -1)
                    if ai_action is None: 
                        valid_actions = tictactoe_env._get_valid_actions()
                        if not valid_actions:
                             yield *update_board_buttons(), "Game is a draw!", "", stats
                             return
                        ai_action = random.choice(valid_actions)

                    reasoning_prompt = f"In TicTacToe, the board is currently: {tictactoe_env.board.flatten().tolist()}. The human player (X) played position {position}. I am the AI (O). The available moves are {tictactoe_env._get_valid_actions()}. I have analyzed the game tree using minimax and determined the optimal move is {ai_action}. Explain my strategy."
                    reasoning = generate_reasoning(reasoning_prompt)
                    tictactoe_env.step(ai_action)
                    
                    if tictactoe_env.game_over:
                        winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "Draw"
                        if winner == "You": stats['wins'] += 1
                        elif winner == "AI": stats['losses'] += 1
                        else: stats['draws'] += 1
                        yield *update_board_buttons(), f"Game Over! {winner} won! AI played {ai_action}.", reasoning, stats
                    else:
                        yield *update_board_buttons(), f"AI played position {ai_action}. Your turn!", reasoning, stats
                    
                except Exception as e:
                    yield *update_board_buttons(), f"Error: {str(e)}", "", stats

            def reset_tictactoe(stats):
                """Reset TicTacToe game."""
                tictactoe_env.reset()
                return *update_board_buttons(), "New game started! You are ❌ (X). Click a square to play.", "AI will show its reasoning here...", stats
            
            # Simplified layout focusing only on TicTacToe
            gr.Markdown("### Play TicTacToe against AI\nYou are ❌ (X) and go first. Click on a square to make your move.")

            with gr.Column():
                with gr.Group(elem_id="ttt-grid"):
                    board_buttons = []
                    for i in range(3):
                        with gr.Row():
                            for j in range(3):
                                pos = i * 3 + j
                                button = gr.Button("", elem_id=f"ttt-cell-{pos}")
                                board_buttons.append(button)

                with gr.Row():
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

            # Create a combined click handler
            def on_board_click(pos, stats):
                yield from play_tictactoe(pos, stats)

            for i in range(9):
                board_buttons[i].click(
                    fn=on_board_click,
                    inputs=[gr.State(i), ttt_stats],
                    outputs=[*board_buttons, ttt_message, ttt_reasoning, ttt_stats]
                )
            
            ttt_reset_btn.click(
                fn=reset_tictactoe,
                inputs=[ttt_stats],
                outputs=[*board_buttons, ttt_message, ttt_reasoning, ttt_stats]
            )
            # Update stats display on changes
            ttt_stats.change(
                fn=lambda s: f"Wins: {s['wins']} | Losses: {s['losses']} | Draws: {s['draws']}",
                inputs=ttt_stats,
                outputs=ttt_stats_display
            )
            gr.Markdown("---")
            gr.Markdown("🚧 **This is a development preview.** Full SPIRAL training and reasoning capabilities will be added in the next update!")

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
        
        if not GAMES_AVAILABLE:
            gr.Markdown("---")
            gr.Markdown("🔄 **Dependencies are loading.** Check the diagnostic info above and refresh in a few minutes!")
        
    return demo

@spaces.GPU(duration=300)
def main():
    """
    Main function to load model, create interface, and launch the Gradio app.
    Wrapped with @spaces.GPU to allocate a GPU for this Space.
    """
    global model, tokenizer

    print("🚀 Starting main application...")
    print("Loading configuration...")
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_name = config['model']['name']
    quantization_params = config['model'].get('quantization', {})
    
    print(f"📦 Model Name: {model_name}")
    print(f"⚙️ Quantization Params: {quantization_params}")


    # Create BitsAndBytesConfig if quantization is enabled
    if quantization_params and quantization_params.get('load_in_4bit'):
        print("💡 4-bit quantization enabled. Creating BitsAndBytesConfig...")
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
        print("🧠 Loading 4-bit quantized model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        print("🧠 Loading model without quantization...")
        # Fallback for no quantization
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    print("✒️ Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("✅ Model and tokenizer loaded successfully.")

    print("🎨 Creating Gradio interface...")
    demo = create_interface()
    
    print("🚀 Launching Gradio app...")
    demo.launch()

if __name__ == "__main__":
    main()
