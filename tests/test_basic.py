"""
Basic tests for SPIRAL Interactive Reasoning Game Simulator.

This module contains fundamental tests to verify the core functionality
of the SPIRAL system components.
"""

import pytest
import os
import sys
import yaml

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app import SpiralApp

class TestSpiralApp:
    """Test cases for the main SPIRAL application."""
    
    def test_app_initialization(self):
        """Test that the app initializes correctly."""
        app = SpiralApp()
        assert app is not None
        assert hasattr(app, 'config')
        assert hasattr(app, 'play_game')
        assert hasattr(app, 'test_reasoning')
    
    def test_config_loading(self):
        """Test configuration loading."""
        app = SpiralApp()
        assert 'interface' in app.config
        assert 'games' in app.config
        assert app.config['interface']['title'] is not None
    
    def test_play_game_basic(self):
        """Test basic game play functionality."""
        app = SpiralApp()
        
        # Test with valid input
        state, response, trace = app.play_game("kuhn_poker", "bet", "")
        assert state is not None
        assert response is not None
        assert trace is not None
        assert "bet" in state
        
        # Test with empty input
        state, response, trace = app.play_game("kuhn_poker", "", "")
        assert "Please enter a move!" in response
    
    def test_reasoning_basic(self):
        """Test basic reasoning functionality."""
        app = SpiralApp()
        
        # Test with valid input
        response, trace = app.test_reasoning("What is 2+2?", "math")
        assert response is not None
        assert trace is not None
        assert "2+2" in response
        
        # Test with empty input
        response, trace = app.test_reasoning("", "math")
        assert "Please enter a reasoning prompt!" in response
    
    def test_interface_creation(self):
        """Test that the Gradio interface can be created."""
        app = SpiralApp()
        demo = app.create_interface()
        assert demo is not None

class TestConfiguration:
    """Test cases for configuration management."""
    
    def test_config_file_structure(self):
        """Test that config.yaml has the expected structure."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            assert 'model' in config
            assert 'games' in config
            assert 'training' in config
            assert 'reasoning' in config
            assert 'interface' in config
            
            # Check model configuration
            assert 'name' in config['model']
            assert 'max_length' in config['model']
            
            # Check games configuration
            assert 'kuhn_poker' in config['games']
            assert 'tictactoe' in config['games']

class TestProjectStructure:
    """Test cases for project structure and imports."""
    
    def test_src_directory_structure(self):
        """Test that the src directory has the expected structure."""
        src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
        
        # Check that required directories exist
        assert os.path.exists(os.path.join(src_path, 'games'))
        assert os.path.exists(os.path.join(src_path, 'models'))
        assert os.path.exists(os.path.join(src_path, 'training'))
        assert os.path.exists(os.path.join(src_path, 'reasoning'))
        
        # Check that __init__.py files exist
        assert os.path.exists(os.path.join(src_path, '__init__.py'))
        assert os.path.exists(os.path.join(src_path, 'games', '__init__.py'))
        assert os.path.exists(os.path.join(src_path, 'models', '__init__.py'))
        assert os.path.exists(os.path.join(src_path, 'training', '__init__.py'))
        assert os.path.exists(os.path.join(src_path, 'reasoning', '__init__.py'))
    
    def test_required_files_exist(self):
        """Test that required project files exist."""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        # Check essential files
        assert os.path.exists(os.path.join(project_root, 'requirements.txt'))
        assert os.path.exists(os.path.join(project_root, 'README.md'))
        assert os.path.exists(os.path.join(project_root, 'config.yaml'))
        assert os.path.exists(os.path.join(project_root, '.gitignore'))
        assert os.path.exists(os.path.join(project_root, 'app', 'app.py'))

if __name__ == "__main__":
    pytest.main([__file__]) 