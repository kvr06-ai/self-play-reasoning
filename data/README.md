# SPIRAL Data Directory

This directory contains datasets, benchmarks, and cached data for the SPIRAL Interactive Reasoning Game Simulator.

## Structure

```
data/
├── cache/              # Cached model outputs and processed data
├── datasets/           # Game datasets and training data
├── benchmarks/         # Evaluation benchmarks for transfer learning
│   ├── gsm8k.json     # GSM8K math problems
│   └── logic_puzzles.json  # Logic reasoning puzzles
└── README.md          # This file
```

## Datasets

### Game Datasets
- **Kuhn Poker**: Training games and strategies
- **TicTacToe**: Game states and optimal moves

### Benchmark Datasets
- **GSM8K**: Grade School Math 8K dataset for mathematical reasoning
- **Logic Puzzles**: Custom logic and reasoning problems
- **Strategic Reasoning**: Game-theory based reasoning tasks

## Usage

Datasets are automatically downloaded and cached when first used. To manually download:

```python
from src.data_utils import download_datasets
download_datasets()
```

## Data Sources

- GSM8K: [Cobbe et al. 2021](https://arxiv.org/abs/2110.14168)
- Logic Puzzles: Curated collection from various sources
- Game Data: Generated through self-play training

## License

Please refer to individual dataset licenses for usage rights. 