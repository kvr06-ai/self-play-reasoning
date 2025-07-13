"""
SPIRAL: Strategic Business Competition Simulator

This demo has been updated to more intuitively demonstrate the key concepts from the 
"Self-Play in Zero-Sum Games Incentivizes Reasoning" (SPIRAL) research paper.

Instead of Tic-Tac-Toe, this simulation uses a zero-sum business competition to showcase
complex, multi-turn strategic reasoning in a more practical and relatable context.
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px

# --- Game Configuration ---
INITIAL_BUDGET = 1000
INITIAL_MARKET_SHARE = 50
INITIAL_PRODUCT_QUALITY = 50
NUM_QUARTERS = 12
TITLE = "SPIRAL: Strategic Business Competition"

# --- Game Environment ---

class BusinessCompetitionEnv:
    """Manages the state of the strategic business competition."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the game to its initial state."""
        self.quarter = 0
        self.game_over = False
        
        self.player_stats = {
            "budget": INITIAL_BUDGET,
            "market_share": INITIAL_MARKET_SHARE,
            "product_quality": INITIAL_PRODUCT_QUALITY,
        }
        self.ai_stats = {
            "budget": INITIAL_BUDGET,
            "market_share": INITIAL_MARKET_SHARE,
            "product_quality": INITIAL_PRODUCT_QUALITY,
        }
        
        # History stores the state at the *end* of each quarter
        self.history = []
        self._add_to_history() # Initial state at quarter 0
        
        return self.get_state()

    def _add_to_history(self):
        """Adds the current state to the history log."""
        self.history.append({
            "Quarter": self.quarter,
            "Player Budget": self.player_stats["budget"],
            "AI Budget": self.ai_stats["budget"],
            "Player Market Share": self.player_stats["market_share"],
            "AI Market Share": self.ai_stats["market_share"],
            "Player Product Quality": self.player_stats["product_quality"],
            "AI Product Quality": self.ai_stats["product_quality"],
        })

    def get_state(self):
        """Returns the complete current state of the game."""
        return {
            "quarter": self.quarter,
            "player_stats": self.player_stats,
            "ai_stats": self.ai_stats,
            "game_over": self.game_over,
            "history": self.history
        }

    def get_winner(self):
        """Determines the winner at the end of the game."""
        if not self.game_over:
            return None
        if self.player_stats["market_share"] > self.ai_stats["market_share"]:
            return "You"
        elif self.ai_stats["market_share"] > self.player_stats["market_share"]:
            return "AI"
        else:
            return "It's a Draw"

    def step(self, player_allocation, ai_allocation):
        """Executes one quarter of the game."""
        if self.game_over:
            return self.get_state()

        self.quarter += 1

        # 1. Update Product Quality from R&D investment
        self.player_stats["product_quality"] += int(np.sqrt(player_allocation["rd"]) * 1.5)
        self.ai_stats["product_quality"] += int(np.sqrt(ai_allocation["rd"]) * 1.5)

        # 2. Calculate market share shift from Marketing and Quality
        mkt_diff = player_allocation["marketing"] - ai_allocation["marketing"]
        quality_diff = self.player_stats["product_quality"] - self.ai_stats["product_quality"]
        
        # Marketing has a direct but temporary effect, quality has a persistent effect
        market_share_shift = (mkt_diff / 100.0) + (quality_diff / 50.0)
        market_share_shift = np.clip(market_share_shift, -7, 7) # Cap shifts per quarter

        self.player_stats["market_share"] += market_share_shift
        self.ai_stats["market_share"] -= market_share_shift
        self.player_stats["market_share"] = np.clip(self.player_stats["market_share"], 0, 100)
        self.ai_stats["market_share"] = 100 - self.player_stats["market_share"]

        # 3. Calculate next quarter's budget from Sales investment and market share
        player_remaining_budget = self.player_stats['budget'] - sum(player_allocation.values())
        ai_remaining_budget = self.ai_stats['budget'] - sum(ai_allocation.values())

        player_sales_roi = 1.2 + (self.player_stats["market_share"] / 200.0)
        ai_sales_roi = 1.2 + (self.ai_stats["market_share"] / 200.0)
        
        self.player_stats["budget"] = int(player_allocation["sales"] * player_sales_roi + player_remaining_budget)
        self.ai_stats["budget"] = int(ai_allocation["sales"] * ai_sales_roi + ai_remaining_budget)

        if self.quarter >= NUM_QUARTERS:
            self.game_over = True
        
        self._add_to_history()

        return self.get_state()

# --- AI Logic ---

def ai_strategy(ai_stats, player_stats):
    """
    A heuristic-based AI to simulate a strategic opponent.
    This mimics the kind of robust strategy that would emerge from self-play,
    reacting to the opponent and planning for the long term.
    """
    budget = ai_stats["budget"]
    reasoning = []
    
    # Default balanced strategy
    allocation = {"rd": 0.33, "marketing": 0.34, "sales": 0.33}

    # --- Strategic Adjustments based on SPIRAL principles ---
    # 1. React to quality gap (long-term planning)
    if ai_stats["product_quality"] < player_stats["product_quality"] - 15:
        allocation["rd"] += 0.2
        allocation["marketing"] -= 0.1
        allocation["sales"] -= 0.1
        reasoning.append("My analysis indicates a growing product quality gap. I'm increasing R&D investment to innovate and secure a long-term competitive advantage.")

    # 2. React to market share loss (short-term defense)
    elif ai_stats["market_share"] < player_stats["market_share"] - 10:
        allocation["marketing"] += 0.2
        allocation["rd"] -= 0.1
        allocation["sales"] -= 0.1
        reasoning.append("You've recently captured significant market share. I'm launching an aggressive marketing campaign to win back customers and regain my position.")

    # 3. Exploit a quality advantage (pressing an advantage)
    if ai_stats["product_quality"] > player_stats["product_quality"] + 20:
        allocation["marketing"] += 0.15
        allocation["rd"] -= 0.15
        reasoning.append(f"My product quality ({ai_stats['product_quality']:.0f}) is superior. I will leverage this with a marketing push to translate product leadership into market dominance.")
    
    # 4. Manage budget (resource management)
    if ai_stats["budget"] < player_stats["budget"] * 0.8:
        allocation["sales"] += 0.15
        allocation["rd"] -= 0.15
        reasoning.append("My projections show a potential budget shortfall. I am focusing on sales to ensure strong revenue growth for future quarters.")

    if not reasoning:
        reasoning.append("I am pursuing a balanced strategy, investing across R&D, Marketing, and Sales to ensure steady, long-term growth and market presence.")

    # Normalize allocations
    total_allocation = sum(allocation.values())
    final_allocation = {key: int(budget * (val / total_allocation)) for key, val in allocation.items()}
    
    # Ensure the sum is exactly the budget
    diff = budget - sum(final_allocation.values())
    final_allocation['sales'] += diff

    return final_allocation, " ".join(reasoning)

# --- Gradio UI ---

def create_interface():
    """Creates the Gradio web interface for the simulator."""
    
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        game_env = gr.State(BusinessCompetitionEnv())

        gr.Markdown(f"# 🎮 {TITLE}")
        gr.Markdown(
            "**Demonstrating how complex, multi-turn strategic reasoning emerges from self-play.**\n"
            "*This simulation replaces Tic-Tac-Toe with a business competition to better illustrate the practical takeaways from the SPIRAL paper.*"
        )
        
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### 📈 Market Dashboard")
                plot_market_share = gr.Plot()
                with gr.Row():
                    plot_budget = gr.Plot()
                    plot_quality = gr.Plot()
            
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Your Decisions")
                status_box = gr.Textbox(f"Quarter 1 of {NUM_QUARTERS}. Your move.", label="Game Status", interactive=False)
                
                with gr.Group():
                    player_budget_display = gr.Label(f"Your Budget: ${INITIAL_BUDGET}")
                    rd_slider = gr.Slider(0, INITIAL_BUDGET, label="R&D Investment", value=333, step=10)
                    mkt_slider = gr.Slider(0, INITIAL_BUDGET, label="Marketing Investment", value=333, step=10)
                    sales_slider = gr.Slider(0, INITIAL_BUDGET, label="Sales Investment", value=334, step=10)
                
                total_allocated_display = gr.Label("Total Allocated: $1000")

                with gr.Row():
                    submit_btn = gr.Button("End Quarter", variant="primary")
                    new_game_btn = gr.Button("Start New Game")

                gr.Markdown("### 🧠 AI Strategic Reasoning")
                ai_reasoning_box = gr.Textbox("", label="AI Decision Rationale", lines=5, interactive=False)
        
        gr.Markdown("---")
        with gr.Accordion("Key Takeaways from the SPIRAL Research Paper", open=False):
            gr.Markdown(open("spiral_paper_takeaways.md").read())

        def create_plots(history):
            df = pd.DataFrame(history)
            if df.empty:
                return None, None, None
            
            fig_ms = px.line(df, x="Quarter", y=["Player Market Share", "AI Market Share"], title="Market Share (%)", markers=True, color_discrete_map={"Player Market Share": "#3b82f6", "AI Market Share": "#ef4444"})
            fig_ms.update_layout(yaxis_range=[0,100], legend_title_text='')

            fig_b = px.line(df, x="Quarter", y=["Player Budget", "AI Budget"], title="Budget ($)", markers=True, color_discrete_map={"Player Budget": "#3b82f6", "AI Budget": "#ef4444"})
            fig_b.update_layout(legend_title_text='')

            fig_q = px.line(df, x="Quarter", y=["Player Product Quality", "AI Product Quality"], title="Product Quality Index", markers=True, color_discrete_map={"Player Product Quality": "#3b82f6", "AI Product Quality": "#ef4444"})
            fig_q.update_layout(legend_title_text='')

            return fig_ms, fig_b, fig_q

        def game_step_and_update(env, rd, mkt, sales):
            player_budget = env.player_stats["budget"]
            if (rd + mkt + sales) > player_budget:
                status_text = f"Error: Allocation (${rd + mkt + sales}) exceeds budget (${player_budget})."
                return env, status_text, env.ai_stats, *create_plots(env.history), gr.Label(f"Your Budget: ${player_budget}"), gr.Slider(maximum=player_budget), gr.Slider(maximum=player_budget), gr.Slider(maximum=player_budget)

            player_alloc = {"rd": rd, "marketing": mkt, "sales": sales}
            ai_alloc, ai_reasoning = ai_strategy(env.ai_stats, env.player_stats)
            
            env.step(player_alloc, ai_alloc)
            state = env.get_state()
            
            plots = create_plots(state["history"])

            if state["game_over"]:
                winner = env.get_winner()
                status_text = f"Game Over! Winner: {winner}. Final market share: You ({state['player_stats']['market_share']:.1f}%) vs AI ({state['ai_stats']['market_share']:.1f}%)."
                submit_btn.interactive = False
            else:
                status_text = f"End of Quarter {state['quarter']}. Your turn."

            new_budget = state["player_stats"]["budget"]
            
            return (state, status_text, ai_reasoning, *plots, 
                    gr.Label(f"Your Budget: ${new_budget}"), 
                    gr.Slider(maximum=new_budget, value=int(new_budget/3)), 
                    gr.Slider(maximum=new_budget, value=int(new_budget/3)), 
                    gr.Slider(maximum=new_budget, value=new_budget - 2 * int(new_budget/3)))

        def on_new_game():
            env = BusinessCompetitionEnv()
            state = env.get_state()
            plots = create_plots(state["history"])
            return (
                env, f"Quarter 1 of {NUM_QUARTERS}. Your move.", "", *plots, 
                gr.Label(f"Your Budget: ${INITIAL_BUDGET}"), 
                gr.Slider(maximum=INITIAL_BUDGET, value=333), 
                gr.Slider(maximum=INITIAL_BUDGET, value=333), 
                gr.Slider(maximum=INITIAL_BUDGET, value=334),
                gr.Button(interactive=True)
            )
            
        def update_total_display(rd, mkt, sales):
            return gr.Label(f"Total Allocated: ${rd + mkt + sales}")
        
        # --- Event Handlers ---
        submit_btn.click(
            fn=game_step_and_update,
            inputs=[game_env, rd_slider, mkt_slider, sales_slider],
            outputs=[
                game_env, status_box, ai_reasoning_box, 
                plot_market_share, plot_budget, plot_quality,
                player_budget_display, rd_slider, mkt_slider, sales_slider
            ]
        )
        
        new_game_btn.click(
            fn=on_new_game,
            inputs=[],
            outputs=[
                game_env, status_box, ai_reasoning_box, 
                plot_market_share, plot_budget, plot_quality,
                player_budget_display, rd_slider, mkt_slider, sales_slider,
                submit_btn
            ]
        )
        
        for slider in [rd_slider, mkt_slider, sales_slider]:
            slider.change(fn=update_total_display, inputs=[rd_slider, mkt_slider, sales_slider], outputs=total_allocated_display)

        demo.load(on_new_game, outputs=[game_env, status_box, ai_reasoning_box, plot_market_share, plot_budget, plot_quality, player_budget_display, rd_slider, mkt_slider, sales_slider, submit_btn])

    return demo


if __name__ == "__main__":
    spiral_demo = create_interface()
    spiral_demo.launch()
