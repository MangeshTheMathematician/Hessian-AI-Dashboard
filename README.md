Hessian AI: Micro-VWAP, Volume Z-Score & Dynamic Exit Dashboard

Live Dashboard: https://mvap-volatility.streamlit.app/

1. Executive Summary

This project is an interactive quantitative trading dashboard built with Streamlit. It models institutional order flow, volume anomalies, and dynamic risk management using VWAP (Volume Weighted Average Price), Rolling Standard Deviation, Volume Z-Scores, and GenAI NLP Sentiment Integration.

Retail traders lose money because they use fixed moving averages and static stop-losses. Institutional quants use volume-adjusted averages (VWAP) to find true market pricing, and they dynamically scale their entries/exits based on real-time market chaos (Volatility).

This dashboard visualizes exactly how a quantitative mean-reversion and momentum strategy identifies extreme volume anomalies, calculates dynamic standard deviation bands, and executes trades mathematically rather than emotionally.

2. Technical Stack

Frontend / App Framework: Streamlit

Data Ingestion: yfinance (Yahoo Finance API) / Databento schemas

Numerical Computing: NumPy

Data Manipulation: Pandas

Data Visualization: Plotly Graph Objects & Subplots

Alternative Data: LLM / Groq API (for unstructured text/sentiment integration)

3. How to Run Locally

To run this dashboard on your local machine, follow these steps:

Clone the repository:

git clone [https://github.com/MangeshTheMathematician/Hessian-AI-Dashboard.git](https://github.com/MangeshTheMathematician/Hessian-AI-Dashboard.git)


Navigate to the folder:

cd Hessian-AI-Dashboard


Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


(Note: You can view the live deployment of this dashboard here: https://mvap-volatility.streamlit.app/)

4. The Core Quant Concept (In Plain English)

To understand this dashboard, you must understand why retail indicators fail and how institutions actually trade.

The Flaw of Normal Averages:
If a stock trades at $10 for 1 minute with 10 shares, and then trades at $20 for 1 minute with 10,000 shares, a normal average says the price is $15. That is completely wrong. The "real" price is much closer to $20 because that is where the "whales" put all their money. VWAP fixes this by attaching a heavy mathematical weight to high-volume trades.

The Flaw of Fixed Stop-Losses:
Retail traders set stops at fixed percentages (e.g., "sell if it drops 2%"). But if the market suddenly becomes incredibly chaotic, a 2% drop is just normal background noise, and you will be stopped out for no reason. Volatility Bands fix this by stretching the safety net wider when the market is crazy, and pulling it tighter when the market is calm.

The "NLP Alpha" GenAI Feed:
Real quant desks do not use GenAI to calculate math; they use it to read the real world. By routing live news into a Large Language Model (LLM), we can extract a Sentiment Vector (-1 or +1) to dynamically lower or raise our trading thresholds on the fly.

5. Comprehensive Math, Proofs, & Plain English Examples

Here is the absolute simplest, plain-English breakdown of every single formula running in the engine. No confusing math symbols, just raw, simple logic.

5.1 Cumulative Dollar Volume (The "Cash Register" Rule)

The Concept: When one share trades at a price, money changes hands. Cumulative dollar volume answers one question: "From the start of the day until now, exactly how many raw dollars have been traded in total?"

The Formula:

$$CumulativeDollarVolume_t = \sum_{i=1}^{t} (Price_i \times Volume_i)$$

The Proof & Derivation:
If you buy 100 shares at $10, the "Dollar Volume" of that single trade is $100 \times \$10 = \$1,000$. To find the total money traded all day, you just add up (Summation $\sum$) the Dollar Volume of every single trade from minute 1 to minute $t$. It is literally just counting the cash in the register.

Real-Life Example:

Minute 1: 100 shares trade at $10. Cash in register = $1,000.

Minute 2: 200 shares trade at $11. Cash in register = $2,200.

Cumulative Dollar Volume = $1,000 + $2,200 = $3,200.

5.2 VWAP (The "Whale" Rule)

The Concept: VWAP reveals the true average price paid by the market by dividing the total money spent by the total shares bought. It tracks the institutional whales.

The Formula:

$$VWAP_t = \frac{\sum_{i=1}^{t} (Price_i \times Volume_i)}{\sum_{i=1}^{t} Volume_i}$$

The Proof & Derivation:
Using the logic above, VWAP is simply:
Total Cash Spent / Total Shares Traded
This proves the exact mathematical cost-basis of the entire market. If whales buy 10,000 shares at $20, the VWAP gets pulled like a magnet to $20.

Real-Life Example:
Using the numbers from 5.1:

Total Dollars Spent (Cumulative Dollar Volume) = $3,200

Total Shares Traded (Cumulative Volume) = $100 + 200 = 300$ shares.

VWAP = $3,200 / 300 =$ $10.66.
(Note: A simple average of $10 and $11 is $10.50, but VWAP pulls the price closer to $11 because more volume occurred there).

5.3 Rolling Standard Deviation (The "Chaos" Rule)

The Concept: How wildly is the volume or price swinging away from its average over a specific window of time? We use this to measure if the market is calm or panicking.

The Formula:

$$\sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}$$

The Proof & Derivation:

Find the distance of each data point from the mean: $(x_i - \mu)$

Square it so negative drops don't cancel out positive jumps: $(x_i - \mu)^2$

Find the average of those squared distances (this is Variance).

Take the square root to return to the original units.

Real-Life Example:
A stock's volume over 3 minutes: 100, 150, 200.

Mean ($\mu$) = 150.

Distances squared: $(100-150)^2 = 2500$, $(150-150)^2 = 0$, $(200-150)^2 = 2500$.

Variance = $(2500 + 0 + 2500) / 3 = 1666.67$.

Volatility ($\sigma$) = $\sqrt{1666.67} \approx$ 40.8 shares.

5.4 Volume Z-Score (The "Rubber Band" Rule)

The Concept: A spike of 1,000 shares is huge for a penny stock, but tiny for Apple. To know if volume is statistically abnormal, we divide the volume spike by the standard deviation. Is it stretched too far?

The Formula:

$$Z_t = \frac{Volume_t - \mu_{Volume}}{\sigma_{Volume}}$$

Real-Life Example:

Normal average volume ($\mu$) = 150 shares/min.

Current chaos/volatility ($\sigma$) = 40.8 shares.

Suddenly, 300 shares trade in one minute!

Z-Score = $(300 - 150) / 40.8 =$ +3.67.
A Z-score over 3.0 means a massive, rare institutional order just hit the market. It is a mathematical anomaly. The rubber band is stretched to the absolute limit.

5.5 Dynamic Volatility Bands (The "Breathing Room" Rule)

The Concept: Instead of setting a fixed stop-loss, we draw mathematical boundaries around the VWAP that expand and contract with market chaos.

The Formula:

$$Upper Band = VWAP + (k \times \sigma_{Price})$$

$$Lower Band = VWAP - (k \times \sigma_{Price})$$

(Where $k$ is your chosen multiplier).

Real-Life Example:

VWAP = $100

Price Volatility ($\sigma$) = $2

Multiplier ($k$) = 2

Upper Band = $100 + (2 \times 2) =$ $104.
If volatility doubles tomorrow to $4, the band automatically widens to $108 so you don't get stopped out by normal noise.

6. Case Study: The Trading Algorithm & GenAI Alpha

How the dashboard puts the math together to generate institutional signals:

Step 1: The Setup

The deterministic engine tracks the VWAP (baseline) and calculates the Volume Z-Score every single second to look for anomalies.

Step 2: The NLP Alpha Feed (GenAI Integration)

Real quant desks do not let probabilistic AI execute trades. They move the LLM upstream to handle unstructured data.

A live Bloomberg/Reuters news API is routed into a Groq LLM.

When the Federal Reserve Chairman speaks, the LLM reads the transcript in real-time.

It analyzes the text and spits out a structured integer: -1 (Hawkish/Bearish) or +1 (Dovish/Bullish).

Step 3: The Dynamic Threshold

Your Python engine takes that +1 from the LLM and instantly lowers your Volume Z-Score threshold from 3.0 to 2.0. It makes the algorithm more aggressive because the AI confirmed the macroeconomic environment is safe.

Step 4: The Execution

The deterministic execution engine triggers a SHORT order if:

Close < VWAP (The stock is fundamentally weak).

Volume Z-Score > Threshold (A massive whale just exhausted their selling power, confirming the drop).

7. Dashboard Features

Dynamic Global Tickers: Pull high-resolution data for any asset via yfinance.

Volume Z-Score Radar: A dedicated sub-chart showing exactly when volume stretches into a mathematical anomaly.

Dynamic Volatility Bands: See the exit bands contract and expand with market chaos.

Signal Generation: Visual markers identifying mathematical entry points based on VWAP and Volume standard deviations.

NLP Threshold Simulation: Adjust parameters to simulate how a GenAI sentiment vector changes the math logic in real-time.

8. Limitations & Future Improvements

Current Limitations:

This is an educational and research dashboard, not a live trading system.

VWAP is cumulative over the loaded CSV. True institutional VWAP resets at the daily open.

Standard deviation can be fragile when the window is too short or if volume follows a heavy-tailed distribution.

Market volume has intraday seasonality (e.g., higher volume at open and close). The dashboard does not currently normalize for time-of-day.

The signal does not include bid-ask spread, order book depth, slippage, latency, or borrow cost.

The GenAI risk brief acts as a downstream parameter modifier, not a raw trading decision-maker.

Future Improvements:

Add full-day VWAP reset logic and anchored VWAP options.

Normalize volume by intraday time-of-day seasonality (U-shaped volume curves).

Add order book imbalance from richer Databento schemas.

Add backtesting with PnL, Sharpe, hit rate, and max drawdown capabilities.

Add live streaming mode with strict unit tests for high-frequency formulas.
