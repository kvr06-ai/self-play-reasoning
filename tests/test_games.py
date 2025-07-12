import pytest
import numpy as np
from src.games.tictactoe import TicTacToeEnv
from src.games.kuhn_poker import KuhnPokerEnv

@pytest.fixture
def ttt_env():
    return TicTacToeEnv()

@pytest.fixture
def kuhn_env():
    return KuhnPokerEnv()

def test_tictactoe_reset(ttt_env):
    obs, info = ttt_env.reset()
    assert np.all(obs == 0)
    assert ttt_env.current_player == 1
    assert not ttt_env.game_over

def test_tictactoe_win(ttt_env):
    # Simulate win for player 1
    ttt_env.step(0)  # X
    ttt_env.step(3)  # O (invalid sim, but test step)
    ttt_env.step(1)  # X
    ttt_env.step(4)  # O
    _, reward, terminated, _, _ = ttt_env.step(2)  # X wins
    assert terminated
    assert reward == 1  # From player 1 perspective
    assert ttt_env.winner == 1

def test_tictactoe_invalid_move(ttt_env):
    ttt_env.step(0)
    _, reward, terminated, _, info = ttt_env.step(0)  # Same spot
    assert 'invalid_move' in info
    assert terminated
    assert reward == -1

def test_kuhn_reset(kuhn_env):
    obs, info = kuhn_env.reset()
    assert kuhn_env.pot == 2  # Antes
    assert kuhn_env.current_player == 1
    assert not kuhn_env.game_over

def test_kuhn_fold(kuhn_env):
    _, reward, terminated, _, _ = kuhn_env.step(2)  # Player 1 folds
    assert terminated
    assert reward == -1  # Lost ante
    assert kuhn_env.winner == -1

def test_kuhn_win(kuhn_env):
    kuhn_env.player1_card = 2  # K
    kuhn_env.player2_card = 0  # J
    kuhn_env.step(1)  # Bet
    kuhn_env.step(0)  # Call
    _, reward, terminated, _, _ = kuhn_env.step(0)  # Call (if needed)
    assert terminated
    assert reward > 0  # Win with higher card
    assert kuhn_env.winner == 1 