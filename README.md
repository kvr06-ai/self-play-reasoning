# SPIRAL: Interactive Reasoning Game Simulator

A practical, interactive tool based on the SPIRAL paper ("Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning") deployed on Hugging Face Spaces.

## Overview

This tool demonstrates how self-play training on zero-sum games can improve AI reasoning capabilities. Users can:

- **Play Games**: Engage with AI in games like Kuhn Poker and TicTacToe
- **View Reasoning**: See step-by-step AI reasoning traces during gameplay
- **Test Transfer**: Evaluate AI's reasoning skills on math problems and logic puzzles
- **Learn**: Understand AI decision-making through interactive visualizations

## Features

### For Non-Technical Users
- Simple web interface for playing games
- Visual reasoning explanations
- Educational tutorials about AI thinking
- No setup required - runs in browser

### For Technical Users
- Access to model weights and training scripts
- API endpoints for extending the system
- Custom game integration capabilities
- Fine-tuning examples and documentation

## Project Structure

```
SPIRAL/
├── src/                    # Core implementation
│   ├── games/             # Game environments
│   ├── models/            # SPIRAL model implementation
│   ├── training/          # Self-play training logic
│   └── reasoning/         # Reasoning trace generation
├── models/                # Trained model weights
├── data/                  # Game datasets and benchmarks
├── app/                   # Gradio web interface
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation and tutorials
```

## Technology Stack

- **Backend**: Python 3.8+
- **ML Framework**: PyTorch, Transformers
- **RL Library**: Gymnasium, Stable Baselines3
- **Web Interface**: Gradio
- **Base Model**: Qwen-4B from Hugging Face
- **Deployment**: Hugging Face Spaces

## Development Phases

1. **Research and Planning** ✅
2. **Implementation** 🔄
3. **Testing and Optimization** 📋
4. **Deployment and Documentation** 📋
5. **Maintenance and Iteration** 📋

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- Hugging Face account (for model access)

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
python app/app.py
```

## Citation

If you use this tool in your research, please cite the original SPIRAL paper:

```bibtex
@article{spiral2024,
  title={Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please use the GitHub Issues or contact us via Hugging Face Spaces. 