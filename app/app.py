"""
SPIRAL Interactive Reasoning Game Simulator - Main Gradio App

A practical tool demonstrating how self-play training on zero-sum games
can improve AI reasoning capabilities.
"""

import gradio as gr
import yaml
import os
import sys

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from typing import Tuple, Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpiralApp:
    """Main application class for the SPIRAL reasoning simulator."""
    
    def __init__(self, config_path: str = "../config.yaml"):
        """Initialize the SPIRAL app with configuration."""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize components (will be implemented in Phase 2)
        self.game_interface = None
        self.reasoning_interface = None
        self.transfer_interface = None
        
        logger.info("SPIRAL App initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'interface': {
                'title': 'SPIRAL: Interactive Reasoning Game Simulator',
                'description': 'Play games against AI and explore reasoning capabilities',
                'theme': 'default'
            },
            'games': {
                'kuhn_poker': {'name': 'Kuhn Poker'},
                'tictactoe': {'name': 'TicTacToe'}
            }
        }
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        logging.getLogger().setLevel(level)
    
    def play_game(self, game_type: str, user_move: str, game_state: str = "") -> Tuple[str, str, str]:
        """
        Handle game play interaction.
        
        Args:
            game_type: Type of game (kuhn_poker, tictactoe)
            user_move: User's move input
            game_state: Current game state
            
        Returns:
            Tuple of (updated_game_state, ai_response, reasoning_trace)
        """
        # Placeholder implementation - will be completed in Phase 2
        if not user_move:
            return game_state, "Please enter a move!", ""
        
        # Simulate AI response
        ai_response = f"AI responds to your move: {user_move}"
        reasoning_trace = f"AI thinking: Analyzing move '{user_move}' in {game_type}..."
        updated_state = f"{game_state}\nUser: {user_move}\nAI: {ai_response}"
        
        return updated_state, ai_response, reasoning_trace
    
    def test_reasoning(self, prompt: str, task_type: str = "math") -> Tuple[str, str]:
        """
        Test AI reasoning on non-game tasks.
        
        Args:
            prompt: User's reasoning prompt
            task_type: Type of reasoning task
            
        Returns:
            Tuple of (response, reasoning_trace)
        """
        # Placeholder implementation - will be completed in Phase 2
        if not prompt:
            return "Please enter a reasoning prompt!", ""
        
        response = f"AI response to: {prompt}"
        reasoning_trace = f"Step-by-step reasoning for '{prompt}'..."
        
        return response, reasoning_trace
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        title = self.config['interface']['title']
        description = self.config['interface']['description']
        
        with gr.Blocks(title=title, theme=self.config['interface']['theme']) as demo:
            gr.Markdown(f"# {title}")
            gr.Markdown(description)
            
            with gr.Tabs():
                # Game Play Tab
                with gr.TabItem("🎮 Game Play"):
                    gr.Markdown("### Play zero-sum games against AI")
                    
                    with gr.Row():
                        with gr.Column():
                            game_selector = gr.Dropdown(
                                choices=["kuhn_poker", "tictactoe"],
                                value="kuhn_poker",
                                label="Select Game"
                            )
                            user_move = gr.Textbox(
                                label="Your Move",
                                placeholder="Enter your move..."
                            )
                            play_button = gr.Button("Play Move", variant="primary")
                            
                        with gr.Column():
                            game_state = gr.Textbox(
                                label="Game State",
                                lines=10,
                                interactive=False
                            )
                            ai_response = gr.Textbox(
                                label="AI Response",
                                lines=3,
                                interactive=False
                            )
                    
                    reasoning_trace = gr.Textbox(
                        label="AI Reasoning Trace",
                        lines=5,
                        interactive=False
                    )
                    
                    play_button.click(
                        fn=self.play_game,
                        inputs=[game_selector, user_move, game_state],
                        outputs=[game_state, ai_response, reasoning_trace]
                    )
                
                # Reasoning Test Tab
                with gr.TabItem("🧠 Reasoning Test"):
                    gr.Markdown("### Test AI reasoning on math and logic problems")
                    
                    with gr.Row():
                        with gr.Column():
                            task_type = gr.Dropdown(
                                choices=["math", "logic", "strategic"],
                                value="math",
                                label="Task Type"
                            )
                            reasoning_prompt = gr.Textbox(
                                label="Reasoning Prompt",
                                placeholder="Enter a math problem or logic puzzle...",
                                lines=3
                            )
                            test_button = gr.Button("Test Reasoning", variant="primary")
                            
                        with gr.Column():
                            reasoning_response = gr.Textbox(
                                label="AI Response",
                                lines=8,
                                interactive=False
                            )
                            reasoning_steps = gr.Textbox(
                                label="Step-by-Step Reasoning",
                                lines=8,
                                interactive=False
                            )
                    
                    test_button.click(
                        fn=self.test_reasoning,
                        inputs=[reasoning_prompt, task_type],
                        outputs=[reasoning_response, reasoning_steps]
                    )
                
                # About Tab
                with gr.TabItem("ℹ️ About"):
                    gr.Markdown("""
                    ### About SPIRAL
                    
                    This tool demonstrates the SPIRAL methodology: "Self-Play on Zero-Sum Games 
                    Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning."
                    
                    **Key Features:**
                    - **Game Play**: Interactive games with AI opponents
                    - **Reasoning Traces**: Transparent AI decision-making
                    - **Transfer Learning**: Test reasoning on non-game tasks
                    - **Educational**: Learn about AI reasoning capabilities
                    
                    **How it works:**
                    1. AI agents are trained via self-play on zero-sum games
                    2. Role-conditioned advantage estimation improves learning
                    3. Reasoning skills transfer to mathematical and logical tasks
                    4. Interactive interface shows the AI's thinking process
                    
                    **Games Available:**
                    - **Kuhn Poker**: Simple poker variant with betting
                    - **TicTacToe**: Classic strategy game
                    
                    **Technical Details:**
                    - Base Model: Qwen-4B from Hugging Face
                    - Training: PPO with self-play
                    - Interface: Gradio web app
                    """)
        
        return demo
    
    def launch(self, **kwargs):
        """Launch the Gradio app."""
        demo = self.create_interface()
        
        # Get launch configuration
        gradio_config = self.config.get('interface', {}).get('gradio', {})
        
        launch_kwargs = {
            'server_name': gradio_config.get('server_name', '0.0.0.0'),
            'server_port': gradio_config.get('server_port', 7860),
            'share': gradio_config.get('share', False),
            'inbrowser': gradio_config.get('inbrowser', True),
            'enable_queue': gradio_config.get('enable_queue', True),
            **kwargs
        }
        
        logger.info(f"Launching SPIRAL app with config: {launch_kwargs}")
        demo.launch(**launch_kwargs)

def main():
    """Main entry point for the application."""
    app = SpiralApp()
    app.launch()

if __name__ == "__main__":
    main() 