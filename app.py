"""
SPIRAL: Interactive Reasoning Game Simulator

Demonstrates key concepts from "Self-Play in Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning"

This simplified demo shows how strategic reasoning emerges from self-play in zero-sum games like TicTacToe.
"""

import gradio as gr
import numpy as np
import random
import spaces


class TicTacToeEnv:
    """Simple TicTacToe environment for SPIRAL demonstration."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # Player 1 starts (X)
        self.game_over = False
        self.winner = None
        self.move_count = 0
        return self.board.copy()
    
    def step(self, action):
        """Execute one step in the environment."""
        if self.game_over:
            return self.board.copy(), 0, True, {}
        
        # Convert action to row, col
        row, col = divmod(action, 3)
        
        # Check if move is valid
        if self.board[row, col] != 0:
            return self.board.copy(), -1, True, {"invalid_move": True}
        
        # Make the move
        self.board[row, col] = self.current_player
        self.move_count += 1
        
        # Check for win
        winner = self._check_winner()
        if winner is not None:
            self.game_over = True
            self.winner = winner
            reward = 1 if winner == self.current_player else -1
            return self.board.copy(), reward, True, {}
        elif self.move_count >= 9:
            # Draw
            self.game_over = True
            return self.board.copy(), 0, True, {}
        else:
            # Game continues
            self.current_player *= -1  # Switch player
            return self.board.copy(), 0, False, {}
    
    def _check_winner(self):
        """Check if there's a winner."""
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
    
    def get_valid_actions(self):
        """Get list of valid actions (empty positions)."""
        valid_actions = []
        for i in range(9):
            row, col = divmod(i, 3)
            if self.board[row, col] == 0:
                valid_actions.append(i)
        return valid_actions


# Global game environment
tictactoe_env = TicTacToeEnv()


def check_winner(board):
    """Check if there's a winner on the given board."""
    # Check rows
    for row in range(3):
        if abs(board[row, :].sum()) == 3:
            return board[row, 0]
    
    # Check columns
    for col in range(3):
        if abs(board[:, col].sum()) == 3:
            return board[0, col]
    
    # Check diagonals
    if abs(board.diagonal().sum()) == 3:
        return board[0, 0]
    
    if abs(np.fliplr(board).diagonal().sum()) == 3:
        return board[0, 2]
    
    return None


def get_valid_moves(board):
    """Get valid moves for the given board."""
    valid_moves = []
    for i in range(9):
        row, col = divmod(i, 3)
        if board[row, col] == 0:
            valid_moves.append(i)
    return valid_moves


def minimax(board, player, depth=0):
    """Minimax algorithm - demonstrates strategic reasoning."""
    # Base cases
    winner = check_winner(board)
    if winner == 1:  # Human wins
        return -10 + depth, None
    elif winner == -1:  # AI wins
        return 10 - depth, None
    elif len(get_valid_moves(board)) == 0:  # Draw
        return 0, None

    best_move = None
    if player == -1:  # AI is maximizing player
        best_score = -float('inf')
        for move in get_valid_moves(board):
            row, col = divmod(move, 3)
            board[row, col] = -1
            score, _ = minimax(board.copy(), 1, depth + 1)
            board[row, col] = 0  # Undo move
            if score > best_score:
                best_score = score
                best_move = move
    else:  # Human is minimizing player
        best_score = float('inf')
        for move in get_valid_moves(board):
            row, col = divmod(move, 3)
            board[row, col] = 1
            score, _ = minimax(board.copy(), -1, depth + 1)
            board[row, col] = 0  # Undo move
            if score < best_score:
                best_score = score
                best_move = move
    
    return best_score, best_move


def generate_reasoning(board_state, human_move, ai_move):
    """Generate reasoning explanation based on game state."""
    reasoning_templates = [
        f"I analyzed all possible moves from the current position. After you played position {human_move}, I considered {len(get_valid_moves(board_state))} possible responses. Using minimax tree search, I determined that position {ai_move} gives me the best strategic advantage.",
        
        f"My decision process: (1) Evaluate immediate threats and opportunities, (2) Project future game states, (3) Choose move that maximizes my winning probability. Position {ai_move} emerged as optimal after analyzing the full game tree.",
        
        f"Strategic analysis: Your move at {human_move} created a new board configuration. I used recursive tree search to evaluate all possible future sequences. Position {ai_move} either creates a winning opportunity or blocks your potential victories.",
        
        f"SPIRAL reasoning: Through self-play training, I learned that position {ai_move} is strategically superior in this configuration. This demonstrates how strategic reasoning emerges from multi-agent interaction in zero-sum games."
    ]
    
    return random.choice(reasoning_templates)


def create_interface():
    """Create the main Gradio interface."""
    
    # Custom CSS to style the TicTacToe board
    css = """
        .ttt-board {
            display: flex;
            flex-direction: column;
            align-items: center;
            max-width: 300px;
            margin: 0 auto;
        }
        .ttt-board > div {
            display: flex;
            flex-direction: row;
            justify-content: center;
            gap: 8px;
            margin: 4px 0;
        }
        .ttt-board button {
            width: 80px !important;
            height: 80px !important;
            min-width: 80px !important;
            min-height: 80px !important;
            max-width: 80px !important;
            max-height: 80px !important;
            font-size: 24px !important;
            font-weight: bold !important;
            border: 2px solid #374151 !important;
            border-radius: 8px !important;
            background: #1f2937 !important;
            color: white !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .ttt-board button:hover {
            background: #374151 !important;
            border-color: #6b7280 !important;
        }
        .ttt-board button:disabled {
            opacity: 0.8 !important;
            cursor: not-allowed !important;
        }
        .ttt-stats {
            text-align: center !important;
            margin: 20px 0 !important;
            font-size: 16px !important;
        }
        .ttt-stats p {
            margin: 0 !important;
            color: #9ca3af !important;
        }
    """
    
    with gr.Blocks(title="SPIRAL: Self-Play Reasoning Demo", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("# 🎮 SPIRAL: Self-Play Reasoning Demo")
        gr.Markdown("**Demonstrating how strategic reasoning emerges from self-play in zero-sum games**")
        gr.Markdown("*Based on: \"Self-Play in Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning\"*")
        
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

        ttt_stats = gr.State({'wins': 0, 'losses': 0, 'draws': 0})
        
        @spaces.GPU
        def play_tictactoe(position, stats):
            """
            Main game loop for TicTacToe. Handles human move, AI response, and updates state.
            This function is decorated with @spaces.GPU to satisfy the Hugging Face Spaces
            runtime, even though the TicTacToe logic does not require GPU acceleration.
            The underlying issue is a mismatch between the selected GPU hardware and the
            CPU-bound nature of the application.
            """
            if tictactoe_env.game_over:
                yield *update_board_buttons(), "Game is over! Click 'New Game' to start again.", "", stats
                return

            try:
                position = int(position)
                
                # Human move
                board_state, reward, done, info = tictactoe_env.step(position)
                
                if done:
                    if info.get("invalid_move"):
                        yield *update_board_buttons(), "Invalid move! Try again.", "", stats
                        return
                    
                    winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "Draw"
                    if winner == "You": stats['wins'] += 1
                    elif winner == "AI": stats['losses'] += 1
                    else: stats['draws'] += 1
                    yield *update_board_buttons(), f"Game Over! {winner} won!", "", stats
                    return

                # Show AI thinking
                yield *update_board_buttons(), "AI is analyzing the game tree...", "🧠 Strategic reasoning in progress...", stats

                # AI move using minimax
                _, ai_action = minimax(tictactoe_env.board.copy(), -1)
                if ai_action is None:
                    valid_actions = tictactoe_env.get_valid_actions()
                    if not valid_actions:
                        yield *update_board_buttons(), "Game is a draw!", "", stats
                        return
                    ai_action = random.choice(valid_actions)
                
                # Generate reasoning explanation
                reasoning = generate_reasoning(tictactoe_env.board.copy(), position, ai_action)
                
                # AI makes move
                board_state, reward, done, info = tictactoe_env.step(ai_action)
                
                if done:
                    winner = "You" if tictactoe_env.winner == 1 else "AI" if tictactoe_env.winner == -1 else "Draw"
                    if winner == "You": stats['wins'] += 1
                    elif winner == "AI": stats['losses'] += 1
                    else: stats['draws'] += 1
                    yield *update_board_buttons(), f"Game Over! {winner} won! AI played position {ai_action}.", reasoning, stats
                else:
                    yield *update_board_buttons(), f"AI chose position {ai_action}. Your turn!", reasoning, stats
            
            except Exception as e:
                yield *update_board_buttons(), f"Error: {str(e)}", "", stats

        def reset_tictactoe(stats):
            """Reset TicTacToe game."""
            tictactoe_env.reset()
            return *update_board_buttons(), "New game started! You are ❌ (X). Click a square to demonstrate strategic reasoning.", "The AI will explain its strategic decision-making process...", stats
        
        with gr.Row():
            with gr.Column(scale=2):
                status_box = gr.Textbox("Welcome to SPIRAL TicTacToe! You are ❌ (X). Click a square to begin.", label="Game Status", interactive=False)
                reasoning_box = gr.Textbox("The AI will explain its strategic moves here.", label="AI Reasoning", interactive=False, lines=4)
                
                with gr.Column(elem_classes=["ttt-board"]):
                    board_buttons = []
                    for i in range(3):
                        with gr.Row():
                            for j in range(3):
                                pos = i * 3 + j
                                btn = gr.Button("", elem_id=f"ttt-btn-{pos}")
                                board_buttons.append(btn)
                
                with gr.Row():
                    new_game_btn = gr.Button("New Game", variant="primary")
                
                # Hidden state for passing button clicks
                clicked_pos = gr.Textbox(visible=False)

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Game Stats")
                stats_display = gr.Markdown("Wins: 0 | Losses: 0 | Draws: 0", elem_classes=["ttt-stats"])
                
                def update_stats_display(stats):
                    return f"Wins: {stats['wins']} | Losses: {stats['losses']} | Draws: {stats['draws']}"
                
                gr.Markdown("""
                ### 🤔 What is SPIRAL?
                SPIRAL stands for **Self-Play in Reinforcement Learning**. This demo illustrates a core concept from the paper: by playing against itself millions of times, an AI can learn complex, human-like strategic reasoning without being explicitly programmed with rules like "take the center square."

                The AI here uses a simple **minimax** algorithm, a classic game theory tree search method, to find the optimal move. This serves as a stand-in for the more complex neural networks used in the actual SPIRAL research.
                """)
        
        # --- Event Handlers ---
        
        def on_board_click(pos, stats):
            """Handler for board button clicks. Propagates to main game logic."""
            yield from play_tictactoe(pos, stats)
        
        # Link button clicks to the handler
        for i, btn in enumerate(board_buttons):
            btn.click(
                fn=on_board_click, 
                inputs=[gr.Textbox(str(i), visible=False), ttt_stats], 
                outputs=[*board_buttons, status_box, reasoning_box, ttt_stats]
            )

        # Link new game button to reset function
        new_game_btn.click(
            fn=reset_tictactoe, 
            inputs=[ttt_stats],
            outputs=[*board_buttons, status_box, reasoning_box, ttt_stats]
        )
        
        # Update stats display when ttt_stats changes
        ttt_stats.change(
            fn=update_stats_display,
            inputs=ttt_stats,
            outputs=stats_display
        )
        
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    spiral_demo = create_interface()
    spiral_demo.launch()
