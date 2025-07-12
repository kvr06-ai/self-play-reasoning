"""
SPIRAL Interactive Reasoning Game Simulator

Main Gradio application for the SPIRAL demo on Hugging Face Spaces.
"""

import gradio as gr
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def create_interface():
    """Create the main Gradio interface."""
    
    with gr.Blocks(title="SPIRAL: Interactive Reasoning Game Simulator") as demo:
        gr.Markdown("# 🎮 SPIRAL: Interactive Reasoning Game Simulator")
        gr.Markdown("**Coming Soon**: Interactive games with AI reasoning traces!")
        
        # Placeholder for now
        gr.Markdown("This app is currently under development. Check back soon!")
        
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch() 