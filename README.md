---
title: "SPIRAL: Strategic Business Competition"
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: An interactive simulator for strategic business competition.
---

# SPIRAL: Strategic Business Competition Simulator

**An interactive demo inspired by the paper: *"Self-Play in Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning"***

This demo has been updated to more intuitively demonstrate the core concepts of the SPIRAL research. Instead of a simple board game, it uses a **strategic business competition** to showcase how competitive pressures in a zero-sum environment can lead to complex, multi-turn reasoning and planning.

## 🎮 The Game: A Zero-Sum Business Battle

You and an AI competitor are in charge of rival companies. Over 12 quarters (turns), you must make critical budget allocation decisions to win market share. The company with the highest market share at the end of the game wins.

Your goal is to strategically allocate your quarterly budget across three key areas:
- **Research & Development (R&D):** Increases your product quality, providing a long-term competitive advantage.
- **Marketing:** Directly captures market share from your opponent in the short term.
- **Sales:** Generates revenue for your next quarter's budget, fueling future growth.

## 🧠 Key Concepts Demonstrated

This simulator illustrates how principles from the SPIRAL framework emerge in a dynamic system:

- **Strategic Reasoning:** The AI analyzes your moves and market conditions to make counter-moves, balancing short-term gains (Marketing) with long-term investments (R&D).
- **Multi-Turn Planning:** A decision to over-invest in marketing for a quick win might leave you with a poor product and low budget in later quarters. You must plan ahead.
- **Emergent Strategies:** There is no single "best" move. The optimal strategy depends on your opponent's actions, forcing you to adapt and reason about their potential choices.
- **Resource Management:** In this zero-sum game, every percentage of market share you gain, the AI loses. Efficiently managing your budget is critical to victory.

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

1.  **Allocate Your Budget:** Use the sliders to decide how to allocate your budget for the quarter.
2.  **Submit Your Move:** Once you finalize your allocation, submit it.
3.  **AI Analysis & Counter-Move:** The AI evaluates the game state and your strategy, then makes its own budget allocation. The AI's reasoning is printed for you to see.
4.  **Quarterly Results:** The simulation advances one quarter. Market share shifts, product quality improves, and new budgets are calculated based on both of your decisions.
5.  **Review and Adapt:** Analyze the results on the dashboard and adapt your strategy for the next quarter.

## 🔬 Research Connection

This demo connects directly to the core findings of the SPIRAL methodology:
-   **Zero-Sum Environments Drive Strategy:** The business competition is a zero-sum game for market share, creating the competitive pressure needed for strategic reasoning to emerge.
-   **Anticipatory Planning:** Success requires you to anticipate how your investments will pay off over several turns and how your opponent will react.
-   **Transferable Reasoning:** The skills developed in this complex game—balancing priorities, managing resources, and predicting opponent behavior—are forms of general strategic reasoning.

## _Dual Git Remotes_

This repository is configured with two remotes:

-   **`origin`**: Pushes to the Hugging Face Space for deployment.
-   **`github`**: Pushes to the public GitHub repository for development and version control.

To push changes to both platforms, you can use the following commands:

```bash
git push origin main
git push github main
```

---

*Experience firsthand how strategic reasoning emerges from competitive self-play!*
