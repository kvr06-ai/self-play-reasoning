---
title: Reasoning Simulator
emoji: 🏆
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.36.2
app_file: app.py
pinned: false
license: apache-2.0
short_description: An interactive reasoning game simulator
---

# SPIRAL: Self-Play Reasoning Demo

**Demonstrating how strategic reasoning emerges from self-play in zero-sum games**

Based on: *"Self-Play in Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning"*

## 🎮 Interactive Demo

This simplified demo showcases the key concepts from the SPIRAL research through an interactive TicTacToe game. Watch as the AI demonstrates strategic reasoning using minimax tree search and explains its decision-making process.

## 🧠 Key Concepts Demonstrated

### Strategic Reasoning
- AI uses minimax tree search to evaluate all possible future moves
- Demonstrates how optimal strategies emerge from competitive gameplay
- Shows explicit reasoning explanations for each move

### Self-Play Learning Principles
- Zero-sum games create competitive pressure that incentivizes strategic thinking
- Multi-agent interactions naturally develop intelligent behavior
- Strategic patterns emerge from repeated competitive gameplay

### Tree Search & Planning
- Minimax algorithm demonstrates formalized strategic reasoning
- Look-ahead planning to evaluate future game states
- Optimal decision-making under competitive constraints

## 🚀 Running the Demo

### Local Setup
```bash
# Clone the repository
git clone https://huggingface.co/spaces/kaushikvr06/reasoning-simulator
cd reasoning-simulator

# Install dependencies
pip install -r requirements.txt

# Run the demo
python app.py
```

### Hugging Face Spaces
The demo is deployed and ready to use at:
[https://huggingface.co/spaces/kaushikvr06/reasoning-simulator](https://huggingface.co/spaces/kaushikvr06/reasoning-simulator)

## 📝 How It Works

1. **Human Move**: Click any square to make your move as X
2. **AI Analysis**: The AI analyzes the game tree using minimax search
3. **Strategic Reasoning**: Watch the AI explain its decision-making process
4. **Optimal Play**: The AI chooses the move that maximizes its winning probability

## 🔬 Research Connection

This demo illustrates core findings from the SPIRAL methodology:

- **Zero-sum competitive environments** naturally incentivize strategic reasoning
- **Multi-turn planning** emerges from the need to anticipate opponent moves
- **Strategic reasoning capabilities** developed through self-play can transfer to general reasoning tasks
- **Tree search algorithms** formalize the strategic reasoning process

## 🎯 Educational Value

Perfect for:
- Understanding strategic AI decision-making
- Learning about game theory and minimax algorithms
- Exploring the connection between competition and intelligence
- Visualizing how reasoning emerges from strategic gameplay

## 📊 Technical Details

- **Game Environment**: Clean TicTacToe implementation with proper state management
- **AI Strategy**: Minimax algorithm with optimal move selection
- **Reasoning Display**: Generated explanations of AI strategic thinking
- **Interactive Interface**: Real-time game state updates and move explanations

---

*Experience firsthand how strategic reasoning emerges from competitive self-play!*
