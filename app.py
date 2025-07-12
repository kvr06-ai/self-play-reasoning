"""
SPIRAL: Interactive Reasoning Game Simulator

Entry point for Hugging Face Spaces deployment.
"""

import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Import and launch the main app
from app import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
