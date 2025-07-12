import pytest
import numpy as np
import sys
import os

# Add src to path to allow importing TicTacToeEnv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from games.tictactoe import TicTacToeEnv

@pytest.fixture
def env():
    """Fixture to create a fresh TicTacToeEnv for each test."""
    return TicTacToeEnv()

def test_initial_state(env):
    """Test the initial state of the board."""
    assert np.all(env.board == np.zeros((3, 3)))
    assert env.current_player == 1
    assert not env.game_over

def test_player_move(env):
    """Test a valid player move."""
    env.step(0)
    assert env.board[0, 0] == 1
    assert env.current_player == -1
    assert not env.game_over

def test_invalid_move(env):
    """Test making an invalid move on an occupied cell."""
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)

def test_win_condition_row(env):
    """Test a win condition in a row."""
    env.board = np.array([[1, 1, 1], [0, -1, 0], [-1, 0, 0]])
    assert env._check_winner(1)
    assert not env._check_winner(-1)

def test_win_condition_col(env):
    """Test a win condition in a column."""
    env.board = np.array([[-1, 1, 0], [-1, 1, 0], [-1, 0, 0]])
    assert not env._check_winner(1)
    assert env._check_winner(-1)

def test_win_condition_diag(env):
    """Test a win condition on a diagonal."""
    env.board = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]])
    assert env._check_winner(1)

def test_draw_condition(env):
    """Test a draw condition."""
    env.board = np.array([[1, -1, 1], [1, -1, 1], [-1, 1, -1]])
    assert env._is_draw()
    assert not env._check_winner(1)
    assert not env._check_winner(-1)

def test_game_over_on_win(env):
    """Test that the game_over flag is set on a win."""
    env.step(0) # P1
    env.step(3) # P2
    env.step(1) # P1
    env.step(4) # P2
    _, _, terminated, _, _ = env.step(2) # P1 wins
    assert terminated
    assert env.game_over
    assert env.winner == 1

def test_reset(env):
    """Test if the environment resets correctly."""
    env.step(0)
    env.step(1)
    env.reset()
    assert np.all(env.board == np.zeros((3, 3)))
    assert env.current_player == 1
    assert not env.game_over
    assert env.winner is None 