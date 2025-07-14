"""
SPIRAL: Strategic Business Competition Simulator

This demo has been updated to more intuitively demonstrate the key concepts from the 
"Self-Play in Zero-Sum Games Incentivizes Reasoning" (SPIRAL) research paper.
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import spaces

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

        # 1. Update Product Quality from R&D investment (with diminishing returns)
        self.player_stats["product_quality"] += int(np.sqrt(player_allocation["rd"]) * 1.5)
        self.ai_stats["product_quality"] += int(np.sqrt(ai_allocation["rd"]) * 1.5)

        # 2. Calculate market share shift from Marketing and Quality
        # Enhanced: Quality impact grows over time (long-term advantage), with increasing weight
        mkt_diff = player_allocation["marketing"] - ai_allocation["marketing"]
        quality_diff = self.player_stats["product_quality"] - self.ai_stats["product_quality"]
        quality_weight = 1 + (self.quarter / NUM_QUARTERS) * 1.5  # Increases from 1 to ~2.5 over 12 quarters
        market_share_shift = (mkt_diff / 100.0) + (quality_diff / 50.0) * quality_weight
        # Clip relaxed in later quarters to allow bigger swings for quality dominance
        clip_max = 7 + (self.quarter / NUM_QUARTERS) * 5  # Up to ~12 by end
        market_share_shift = np.clip(market_share_shift, -clip_max, clip_max)

        self.player_stats["market_share"] += market_share_shift
        self.ai_stats["market_share"] -= market_share_shift
        self.player_stats["market_share"] = np.clip(self.player_stats["market_share"], 0, 100)
        self.ai_stats["market_share"] = 100 - self.player_stats["market_share"]

        # 3. Calculate next quarter's budget from Sales investment and market share
        # Enhanced: ROI now exponentially rewards quality (premium pricing for better products)
        player_remaining_budget = self.player_stats['budget'] - sum(player_allocation.values())
        ai_remaining_budget = self.ai_stats['budget'] - sum(ai_allocation.values())

        # Exponential quality bonus: (quality / 100) ** 1.5 for non-linear growth
        player_quality_bonus = (self.player_stats["product_quality"] / 100) ** 1.5
        ai_quality_bonus = (self.ai_stats["product_quality"] / 100) ** 1.5

        player_sales_roi = 1.2 + (self.player_stats["market_share"] / 200.0) + player_quality_bonus * 0.5
        ai_sales_roi = 1.2 + (self.ai_stats["market_share"] / 200.0) + ai_quality_bonus * 0.5
        
        self.player_stats["budget"] = int(player_allocation["sales"] * player_sales_roi + player_remaining_budget)
        self.ai_stats["budget"] = int(ai_allocation["sales"] * ai_sales_roi + ai_remaining_budget)

        # Error Handling: Clamp budgets to >=0; if 0, add small recovery to prevent instant loss (realism: minimal operations continue)
        self.player_stats["budget"] = max(50, self.player_stats["budget"]) if self.player_stats["budget"] == 0 else max(0, self.player_stats["budget"])
        self.ai_stats["budget"] = max(50, self.ai_stats["budget"]) if self.ai_stats["budget"] == 0 else max(0, self.ai_stats["budget"])

        if self.quarter >= NUM_QUARTERS:
            self.game_over = True
        
        self._add_to_history()

        return self.get_state()

# --- AI Logic ---

def ai_strategy(ai_stats, player_stats, quarter):
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
    # Dynamic thresholds: Tighten as game progresses (simulates adaptive curriculum)
    quality_gap_threshold = 15 - (quarter // 3)  # E.g., starts at 15, drops to 9 by quarter 9
    market_share_threshold = 10 - (quarter // 4)  # Starts at 10, drops to 7 by quarter 8
    quality_advantage_threshold = 20 - (quarter // 3)
    budget_threshold = 0.8 + (quarter / 100.0)  # Slightly increases to make AI more conservative later

    # Enhanced: AI reacts more aggressively to widening quality gaps in later quarters
    if quarter > 6 and ai_stats["product_quality"] < player_stats["product_quality"] - (quality_gap_threshold * 1.2):
        allocation["rd"] += 0.25
        allocation["marketing"] -= 0.125
        allocation["sales"] -= 0.125
        reasoning.append(f"Quarter {quarter}: Late-game quality deficit detected—ramping up R&D aggressively to catch up before it's too late.")

    # 1. React to quality gap (long-term planning)
    elif ai_stats["product_quality"] < player_stats["product_quality"] - quality_gap_threshold:
        allocation["rd"] += 0.2
        allocation["marketing"] -= 0.1
        allocation["sales"] -= 0.1
        reasoning.append(f"Quarter {quarter}: My analysis indicates a growing product quality gap (threshold: {quality_gap_threshold}). I'm increasing R&D investment to innovate and secure a long-term competitive advantage.")

    # 2. React to market share loss (short-term defense)
    elif ai_stats["market_share"] < player_stats["market_share"] - market_share_threshold:
        allocation["marketing"] += 0.2
        allocation["rd"] -= 0.1
        allocation["sales"] -= 0.1
        reasoning.append(f"Quarter {quarter}: You've recently captured significant market share (threshold: {market_share_threshold}). I'm launching an aggressive marketing campaign to win back customers and regain my position.")

    # 3. Exploit a quality advantage (pressing an advantage)
    if ai_stats["product_quality"] > player_stats["product_quality"] + quality_advantage_threshold:
        allocation["marketing"] += 0.15
        allocation["rd"] -= 0.15
        reasoning.append(f"Quarter {quarter}: My product quality ({ai_stats['product_quality']:.0f}) is superior (threshold: {quality_advantage_threshold}). I will leverage this with a marketing push to translate product leadership into market dominance.")
    
    # 4. Manage budget (resource management)
    if ai_stats["budget"] < player_stats["budget"] * budget_threshold:
        allocation["sales"] += 0.15
        allocation["rd"] -= 0.15
        reasoning.append(f"Quarter {quarter}: My projections show a potential budget shortfall (threshold: {budget_threshold:.2f}). I am focusing on sales to ensure strong revenue growth for future quarters.")

    if not reasoning:
        reasoning.append(f"Quarter {quarter}: I am pursuing a balanced strategy, investing across R&D, Marketing, and Sales to ensure steady, long-term growth and market presence.")

    # Normalize allocations
    total_allocation = sum(allocation.values())
    final_allocation = {key: int(budget * (val / total_allocation)) for key, val in allocation.items()}
    
    # Simulate RAE-inspired stability: Average with a "role-reversed" allocation
    role_reversed_alloc = {"rd": allocation["rd"], "marketing": allocation["sales"], "sales": allocation["marketing"]}  # Simple swap for variance reduction
    reversed_total = sum(role_reversed_alloc.values())
    reversed_final = {key: int(budget * (val / reversed_total)) for key, val in role_reversed_alloc.items()}
    for key in final_allocation:
        final_allocation[key] = int((final_allocation[key] + reversed_final[key]) / 2)
    
    # Ensure the sum is exactly the budget
    diff = budget - sum(final_allocation.values())
    final_allocation['sales'] += diff

    return final_allocation, " ".join(reasoning)

# --- Post-Game Analysis ---

def generate_post_game_analysis(history, winner):
    """Generates dynamic feedback based on game results."""
    initial_state = history[0]
    final_state = history[-1]

    # Calculate key performance indicators
    quality_change_player = final_state['Player Product Quality'] - initial_state['Player Product Quality']
    quality_change_ai = final_state['AI Product Quality'] - initial_state['AI Product Quality']
    
    budget_change_player = final_state['Player Budget'] - initial_state['Player Budget']
    budget_change_ai = final_state['AI Budget'] - initial_state['AI Budget']
    
    market_share_lead_player_final = final_state['Player Market Share'] - final_state['AI Market Share']
    
    mid_game_state = history[len(history) // 2]
    market_share_lead_player_mid = mid_game_state['Player Market Share'] - mid_game_state['AI Market Share']

    feedback = []

    if winner == "You":
        feedback.append("🏆 **Congratulations on your victory!** Here's what led to your success:")
        if quality_change_player > quality_change_ai * 1.2 and quality_change_player > 20:
            feedback.append("• **Innovation Leadership:** Your strong and consistent investment in R&D created a superior product that was difficult for the AI to overcome.")
        if budget_change_player > budget_change_ai * 1.2 and budget_change_player > 200:
            feedback.append("• **Financial Acumen:** You expertly grew your budget, giving you the resources to outmaneuver the AI in the final quarters.")
        if market_share_lead_player_mid < 0 and market_share_lead_player_final > 0:
            feedback.append("• **Strategic Comeback:** Your impressive turnaround in the second half of the game shows excellent adaptation and resilience.")
        if not feedback[1:]: # If no specific points were added
            feedback.append("• **Balanced Strategy:** Your well-rounded approach of investing across R&D, Marketing, and Sales proved to be a winning formula.")

    elif winner == "AI":
        feedback.append("📈 **A hard-fought game!** The AI is a tough opponent. Here's a quick analysis for next time:")
        if quality_change_ai > quality_change_player * 1.2 and quality_change_ai > 20:
            feedback.append("• **Innovation Gap:** The AI secured a significant product quality advantage. Consider allocating more to R&D earlier on to build a competitive moat.")
        if budget_change_ai > budget_change_player * 1.2 and budget_change_ai > 200:
            feedback.append("• **Budget Disadvantage:** The AI grew its budget more effectively. Prioritizing Sales earlier could provide you with more financial power in later stages.")
        if abs(market_share_lead_player_final) < 10:
             feedback.append(f"• **Narrow Margin:** You lost by only {abs(market_share_lead_player_final):.1f}%! A small, targeted marketing push might have been enough to change the outcome.")
        if not feedback[1:]:
            feedback.append("• **Countering a Balanced Foe:** The AI succeeded with a balanced approach. Experimenting with a more aggressive, focused strategy could help you find and exploit a weakness.")

    else: # Draw
        feedback.append("🤝 **An incredibly close match, ending in a draw!**")
        feedback.append("• **Strategic Stalemate:** Both you and the AI demonstrated strong, well-balanced strategies, with neither able to gain a decisive, lasting advantage.")

    return "\n".join(feedback)

# --- Gradio UI ---

def create_interface():
    """Creates the Gradio web interface for the simulator."""
    
    # Force HF rebuild - theme removed to fix compatibility issues
    with gr.Blocks(title=TITLE) as demo:
        game_env = gr.State(BusinessCompetitionEnv())

        gr.Markdown(f"# 🎮 {TITLE}")

        with gr.Accordion("ℹ️ What is this app about & How to play", open=False):
            gr.Markdown("""
            ### What is this app about?

            **For Business Strategists, Product Managers, and Students:**

            This simulator is a hands-on sandbox for exploring the core trade-offs of business strategy. You are in control of a company competing against a strategic AI. By allocating your budget each quarter, you can directly see the impact of your decisions:

            -   **Short-term vs. Long-term:** Feel the tension between investing in Marketing for immediate market share gains versus investing in R&D for a long-term product advantage.
            -   **Resource Management:** Learn how investing in Sales grows your future budget, enabling more significant investments later on.
            -   **Competitive Dynamics:** The AI opponent doesn't play a fixed strategy. It analyzes your moves and adapts, forcing you to think multiple turns ahead. This provides an intuitive feel for how competitive landscapes evolve.

            **For AI/ML Engineers and Data Scientists:**

            This demo provides a practical look at the principles of advanced AI reasoning described in the SPIRAL research paper. The AI opponent uses a heuristic-based strategy model designed to mimic the emergent behaviors and outcomes of self-play reinforcement learning—such as adaptive planning and threat response—without full RL training.

            -   **Emergent Strategy:** The AI's decision-making process illustrates how an agent can learn to balance priorities, react to threats, and press advantages—all without being explicitly programmed for each scenario. This is a core concept of self-play.
            -   **Multi-Turn Reasoning:** Observe the AI's rationale. It often makes decisions based on future projections (e.g., potential budget shortfalls or quality gaps), showcasing a capacity for long-term planning.
            -   **Zero-Sum Dynamics:** The simulation is a zero-sum game for market share, creating the competitive pressure that, according to the SPIRAL paper, is essential for incentivizing robust reasoning.

            This demo is inspired by the SPIRAL framework from the research paper: [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119).

            ### Key Links to SPIRAL Paper Takeaways
            - **Transferable Reasoning:** Your R&D investments build long-term planning skills, transferable to real-world logic problems.
            - **Diverse Skills:** Marketing encourages probabilistic thinking (like Poker), while Sales focuses on resource foresight.
            - **Synergy from Multi-Game Training:** Combining these creates a well-rounded strategy, better than focusing on one area.

            ### How to Use the App

            1.  **Your Goal:** Achieve a higher market share than the AI by the end of 12 quarters.
            2.  **Choose Your Mode:** Select either "Raw Values" or "Percentages" to allocate your budget.
            3.  **Allocate Budget:** Use the sliders to decide how much of your quarterly budget to invest in three key areas.
                -   `R&D`: Improves your product quality, giving you a persistent, long-term edge.
                -   `Marketing`: Provides an immediate boost to your market share for the current quarter.
                -   `Sales`: Increases your budget for the *next* quarter, fueling future growth. (Hint: Your coffers may flourish like a well-tended garden or wither like neglected vines, depending on how you nurture sales and market dominance—choose wisely to unlock the mysteries of compounding fortune!)
            4.  **End the Quarter:** Click the "End Quarter" button to submit your decisions.
            5.  **Analyze the Results:**
                -   The charts on the left will update to show the new market landscape.
                -   The "AI Strategic Reasoning" box will explain the logic behind the AI's counter-move.
                -   Your budget for the next quarter will be updated.
            6.  **Adapt and Win:** Continue making decisions for 12 quarters, adapting your strategy to counter the AI and win the market.
            """)
        
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
                    allocation_mode_radio = gr.Radio(["Raw Values", "Percentages"], label="Allocation Mode", value="Raw Values")

                    with gr.Group() as raw_values_group:
                        rd_slider_raw = gr.Slider(0, INITIAL_BUDGET, label="R&D Investment", value=333, step=10)
                        mkt_slider_raw = gr.Slider(0, INITIAL_BUDGET, label="Marketing Investment", value=333, step=10)
                        sales_slider_raw = gr.Slider(0, INITIAL_BUDGET, label="Sales Investment", value=334, step=10)
                        total_allocated_raw_display = gr.Label("Total Allocated: $1000")

                    with gr.Group(visible=False) as percentage_group:
                        rd_slider_pct = gr.Slider(0, 100, label="R&D Allocation (%)", value=33, step=1)
                        mkt_slider_pct = gr.Slider(0, 100, label="Marketing Allocation (%)", value=33, step=1)
                        sales_slider_pct = gr.Slider(0, 100, label="Sales Allocation (%)", value=34, step=1)
                        total_allocated_pct_display = gr.Label("Total Allocated: 100%")

                with gr.Row():
                    submit_btn = gr.Button("End Quarter", variant="primary")
                    new_game_btn = gr.Button("Start New Game")

                gr.Markdown("### 🧠 AI Strategic Reasoning")
                ai_reasoning_box = gr.Textbox("", label="AI Decision Rationale", lines=5, interactive=False)

                gr.Markdown("### 📝 Post-Game Analysis")
                analysis_box = gr.Textbox("", label="Strategy Insights", lines=5, interactive=False)
        
        def create_plots(history):
            df = pd.DataFrame(history)
            if df.empty:
                return None, None, None
            
            fig_ms = px.line(df, x="Quarter", y=["Player Market Share", "AI Market Share"], title="Market Share (%)", markers=True, color_discrete_map={"Player Market Share": "#3b82f6", "AI Market Share": "#ef4444"})
            fig_ms.update_layout(yaxis_range=[0,100], legend_title_text='', template="plotly_dark")

            fig_b = px.line(df, x="Quarter", y=["Player Budget", "AI Budget"], title="Budget ($)", markers=True, color_discrete_map={"Player Budget": "#3b82f6", "AI Budget": "#ef4444"})
            fig_b.update_layout(legend_title_text='', template="plotly_dark")

            fig_q = px.line(df, x="Quarter", y=["Player Product Quality", "AI Product Quality"], title="Product Quality Index", markers=True, color_discrete_map={"Player Product Quality": "#3b82f6", "AI Product Quality": "#ef4444"})
            fig_q.update_layout(legend_title_text='', template="plotly_dark")

            return fig_ms, fig_b, fig_q

        @spaces.GPU
        def game_step_and_update(env, mode, rd_raw, mkt_raw, sales_raw, rd_pct, mkt_pct, sales_pct):
            player_budget = env.player_stats["budget"]

            # Helper to create a return tuple for user input errors
            def create_error_return(status_text):
                return (
                    env, status_text, env.ai_stats.get("last_reasoning", ""), *create_plots(env.history),
                    gr.update(value=f"Your Budget: ${player_budget}"),
                    gr.update(), gr.update(), gr.update(), # Raw sliders
                    gr.update(), gr.update(), gr.update(), # Pct sliders
                    gr.update(interactive=True), # Submit button
                    gr.update()  # Analysis box
                )

            if mode == "Percentages":
                if rd_pct + mkt_pct + sales_pct != 100:
                    return create_error_return("Error: Percentage allocations must sum to 100%.")
                
                rd_alloc_val = int(player_budget * rd_pct / 100)
                mkt_alloc_val = int(player_budget * mkt_pct / 100)
                sales_alloc_val = int(player_budget * sales_pct / 100)
                
                total = rd_alloc_val + mkt_alloc_val + sales_alloc_val
                sales_alloc_val += player_budget - total
                
            else: # Raw Values
                rd_alloc_val, mkt_alloc_val, sales_alloc_val = rd_raw, mkt_raw, sales_raw

            if (rd_alloc_val + mkt_alloc_val + sales_alloc_val) > player_budget:
                return create_error_return(f"Error: Allocation (${rd_alloc_val + mkt_alloc_val + sales_alloc_val}) exceeds budget (${player_budget}).")

            player_alloc = {"rd": rd_alloc_val, "marketing": mkt_alloc_val, "sales": sales_alloc_val}
            ai_alloc, ai_reasoning = ai_strategy(env.ai_stats, env.player_stats, env.quarter + 1)  # Pass next quarter
            env.ai_stats["last_reasoning"] = ai_reasoning
            
            env.step(player_alloc, ai_alloc)
            state = env.get_state()
            
            plots = create_plots(state["history"])

            submit_btn_update = gr.update(interactive=True)
            analysis_text = ""
            if state["game_over"]:
                winner = env.get_winner()
                status_text = f"Game Over! Winner: {winner}. Final market share: You ({state['player_stats']['market_share']:.1f}%) vs AI ({state['ai_stats']['market_share']:.1f}%)."
                submit_btn_update = gr.update(interactive=False)
                # Generate post-game analysis
                analysis_text = generate_post_game_analysis(state["history"], winner)
            else:
                status_text = f"End of Quarter {state['quarter']}. Your turn."

            new_budget = state["player_stats"]["budget"]
            
            return (
                env, status_text, ai_reasoning, *plots, 
                gr.update(value=f"Your Budget: ${new_budget}"), 
                gr.update(maximum=new_budget, value=int(new_budget/3)), 
                gr.update(maximum=new_budget, value=int(new_budget/3)), 
                gr.update(maximum=new_budget, value=new_budget - 2 * int(new_budget/3)),
                gr.update(value=33), gr.update(value=33), gr.update(value=34),
                submit_btn_update,
                analysis_text
            )

        def on_new_game():
            env = BusinessCompetitionEnv()
            state = env.get_state()
            plots = create_plots(state["history"])
            return (
                env, f"Quarter 1 of {NUM_QUARTERS}. Your move.", "", *plots, 
                gr.update(value=f"Your Budget: ${INITIAL_BUDGET}"), 
                gr.update(maximum=INITIAL_BUDGET, value=333), 
                gr.update(maximum=INITIAL_BUDGET, value=333), 
                gr.update(maximum=INITIAL_BUDGET, value=334),
                gr.update(value=33), gr.update(value=33), gr.update(value=34),
                gr.update(interactive=True),
                ""
            )
            
        def update_total_raw_display(rd, mkt, sales):
            return gr.Label(f"Total Allocated: ${rd + mkt + sales}")
        
        def update_total_pct_display(rd, mkt, sales):
            return gr.Label(f"Total Allocated: {rd + mkt + sales}%")

        def toggle_allocation_mode(mode):
            return gr.update(visible=mode == "Raw Values"), gr.update(visible=mode == "Percentages")

        def adjust_pct_sliders(rd, mkt):
            return gr.update(value=100 - rd - mkt)

        # --- Event Handlers ---
        submit_btn.click(
            fn=game_step_and_update,
            inputs=[game_env, allocation_mode_radio, rd_slider_raw, mkt_slider_raw, sales_slider_raw, rd_slider_pct, mkt_slider_pct, sales_slider_pct],
            outputs=[
                game_env, status_box, ai_reasoning_box, 
                plot_market_share, plot_budget, plot_quality,
                player_budget_display, 
                rd_slider_raw, mkt_slider_raw, sales_slider_raw,
                rd_slider_pct, mkt_slider_pct, sales_slider_pct,
                submit_btn,
                analysis_box
            ]
        )
        
        new_game_btn.click(
            fn=on_new_game,
            inputs=[],
            outputs=[
                game_env, status_box, ai_reasoning_box, 
                plot_market_share, plot_budget, plot_quality,
                player_budget_display, 
                rd_slider_raw, mkt_slider_raw, sales_slider_raw,
                rd_slider_pct, mkt_slider_pct, sales_slider_pct,
                submit_btn,
                analysis_box
            ]
        )
        
        # Handlers for updating total displays
        for slider in [rd_slider_raw, mkt_slider_raw, sales_slider_raw]:
            slider.change(fn=update_total_raw_display, inputs=[rd_slider_raw, mkt_slider_raw, sales_slider_raw], outputs=total_allocated_raw_display)
        
        for slider in [rd_slider_pct, mkt_slider_pct, sales_slider_pct]:
            slider.change(fn=update_total_pct_display, inputs=[rd_slider_pct, mkt_slider_pct, sales_slider_pct], outputs=total_allocated_pct_display)

        # Auto-adjust percentage sliders
        rd_slider_pct.change(fn=adjust_pct_sliders, inputs=[rd_slider_pct, mkt_slider_pct], outputs=sales_slider_pct)
        mkt_slider_pct.change(fn=adjust_pct_sliders, inputs=[rd_slider_pct, mkt_slider_pct], outputs=sales_slider_pct)

        # Handler for toggling allocation modes
        allocation_mode_radio.change(
            fn=toggle_allocation_mode,
            inputs=allocation_mode_radio,
            outputs=[raw_values_group, percentage_group]
        )

        demo.load(on_new_game, outputs=[game_env, status_box, ai_reasoning_box, plot_market_share, plot_budget, plot_quality, player_budget_display, rd_slider_raw, mkt_slider_raw, sales_slider_raw, rd_slider_pct, mkt_slider_pct, sales_slider_pct, submit_btn, analysis_box])

    return demo


if __name__ == "__main__":
    spiral_demo = create_interface()
    spiral_demo.launch()