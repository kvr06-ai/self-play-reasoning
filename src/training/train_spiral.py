import os
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

# Load config
with open('../../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model']['name']
max_length = config['model']['max_length']

# Load base LLM (quantized)
model = AutoModelForCausalLM.from_pretrained(model_name, **config['model']['quantization'])
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom Policy with RAE (simplified)
class SpiralPolicy(torch.nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.role_embed = torch.nn.Embedding(2, 64)  # 0: player, 1: opponent
        # Add more layers as needed

    def forward(self, obs, role):
        # Condition on role
        role_emb = self.role_embed(role)
        # Compute policy/value (placeholder)
        return policy, value

def train_spiral(game='tictactoe', episodes=1000):
    if game == 'tictactoe':
        from src.games.tictactoe import TicTacToeEnv
        env_fn = lambda: TicTacToeEnv()
    else:
        raise ValueError('Game not supported yet')
    
    env = DummyVecEnv([env_fn])
    
    # PPO with custom policy
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003)
    
    # Self-play loop (simplified: train against current self)
    for ep in range(episodes):
        model.learn(total_timesteps=1000)  # Train batch
        # Simulate self-play by cloning or saving opponent policy
        print(f'Episode {ep}: Trained')
    
    # Save model
    os.makedirs('../../models', exist_ok=True)
    model.save('../../models/spiral_tictactoe.zip')
    print('Model saved!')

if __name__ == '__main__':
    train_spiral() 