# %%
import yfinance as yf
from datetime import datetime
import numpy as np

print("INITIALIZING HESSIAN-AI DATA TAP...\n")

# ---------------------------------------------------------
# STEP 1: TARGET ACQUISITION & SPOT PRICE (S)
# ---------------------------------------------------------
ticker_symbol = "SPY"
target = yf.Ticker(ticker_symbol)

# Grab the most recent daily data and extract the final closing price
todays_data = target.history(period="1d")
S = todays_data['Close'].iloc[-1]

print(f"[+] TARGET SECURED: {ticker_symbol}")
print(f"[+] Current Spot Price (S): ${S:.2f}")

# ---------------------------------------------------------
# STEP 2: THE TIMELINE (T)
# ---------------------------------------------------------
# Get all available expiration dates from the exchange
expirations = target.options

# Let's pick an expiration date a few weeks out (the 3rd option on the list)
target_date_str = expirations[2] 

# Convert the string date ('YYYY-MM-DD') into a Python datetime object
target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
today = datetime.today()

# Calculate the exact number of days remaining
days_to_expiry = (target_date - today).days

# Convert days into the "Fraction of a Year" format required by Quant math
T = days_to_expiry / 365.0

print(f"[+] Expiration Date: {target_date_str} ({days_to_expiry} days away)")
print(f"[+] Time to Expiration (T): {T:.4f} years")

# ---------------------------------------------------------
# STEP 3: THE OPTION CONTRACT (K and C)
# ---------------------------------------------------------
# Download the massive table of options for our chosen date
options_chain = target.option_chain(target_date_str)

# Isolate just the Call options
calls = options_chain.calls

# To find a relevant option, let's just grab the one sitting right in the middle of the table
# (This usually represents an "At-The-Money" option near the current stock price)
middle_index = len(calls) // 2
sample_call = calls.iloc[middle_index]

# Rip out the exact Strike Price and the Market Price
K = sample_call['strike']
C = sample_call['lastPrice']

print(f"[+] Target Contract Found.")
print(f"[+] Strike Price (K): ${K:.2f}")
print(f"[+] Market Price (C): ${C:.2f}")

# ---------------------------------------------------------
# FINAL PIPELINE CHECK
# ---------------------------------------------------------
print("\n========================================")
print("RAW DATA EXTRACTION COMPLETE - READY FOR ENGINE")
print(f"S = {S:.2f}")
print(f"K = {K:.2f}")
print(f"T = {T:.4f}")
print(f"C = {C:.2f}")
print("========================================")

# %%
import scipy.stats as si

print("IGNITING NEWTON-RAPHSON ENGINE...\n")

# ---------------------------------------------------------
# THE MATRIX MATH (Black-Scholes & Vega)
# ---------------------------------------------------------
def bs_call_price(S, K, T, r, sigma):
    """Calculates theoretical price inside the Risk-Neutral Matrix"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return (S * si.norm.cdf(d1)) - (K * np.exp(-r * T) * si.norm.cdf(d2))

def bs_vega(S, K, T, r, sigma):
    """Calculates the slope (friction) for the correction"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T)

# ---------------------------------------------------------
# THE HUNTER (Newton-Raphson Loop)
# ---------------------------------------------------------
def calculate_iv(market_price, S, K, T, r, tol=1e-5, max_iterations=100):
    # Start with a blind guess of 20% volatility
    sigma = 0.20 
    
    for i in range(max_iterations):
        # Calculate theoretical price based on current guess
        theoretical_price = bs_call_price(S, K, T, r, sigma)
        
        # Calculate the error
        error = theoretical_price - market_price
        
        # If the error is practically zero, we found the exact IV!
        if abs(error) < tol:
            print(f"[+] Lock achieved in {i} iterations.")
            return sigma
            
        vega = bs_vega(S, K, T, r, sigma)
        
        # Prevent division by zero if the math flatlines
        if vega < 1e-6:
            return np.nan
            
        # The master equation: IV_new = IV_old - (Error / Slope)
        sigma = sigma - (error / vega)
        
        # Volatility cannot be physically negative
        if sigma <= 0.0:
            sigma = 1e-5
            
    return np.nan

# ---------------------------------------------------------
# EXECUTE THE ENGINE
# ---------------------------------------------------------
# We assume the Risk-Free Bank Rate (r) is currently around 5%
r = 0.05 

# Run the hunter using the data from Step 1
final_iv = calculate_iv(C, S, K, T, r)

print("========================================")
print("IMPLIED VOLATILITY EXTRACTED")
print(f"Target Contract: SPY ${K:.2f} Call")
print(f"Raw IV Decimal:  {final_iv:.4f}")
print(f"Final True IV:   {final_iv * 100:.2f}%")
print("========================================")

# %%
import matplotlib.pyplot as plt
import numpy as np

print("BUILDING THE VOLATILITY SMILE MAP...\n")

# 1. Create empty lists to store our physical coordinates
strikes = []
ivs = []

# 2. Loop through every single call option in our downloaded chain
for index, row in calls.iterrows():
    K_loop = row['strike']
    C_loop = row['lastPrice']
    volume = row['volume']
    
    # FILTER THE NOISE: 
    # Options with no volume or prices near $0.00 will break the math (division by zero).
    # We only want to map options that real humans are actually trading.
    if C_loop > 0.05 and volume > 0:
        
        # Fire the Newton-Raphson engine for this specific strike
        iv_loop = calculate_iv(C_loop, S, K_loop, T, r)
        
        # If the engine successfully found an IV (and it's a realistic number under 200%)
        if not np.isnan(iv_loop) and 0.01 < iv_loop < 2.0:
            strikes.append(K_loop)
            ivs.append(iv_loop * 100) # Convert decimal to percentage

print(f"[+] Successfully mapped {len(strikes)} IV coordinates.")
print("[+] Rendering Hessian-AI Dashboard Graphic...")

# 3. Draw the physical graphic
plt.figure(figsize=(10, 6), facecolor='#1e1e1e')
ax = plt.axes()
ax.set_facecolor('#1e1e1e')

# Plot the dots (the individual options) and connect them with a line
plt.scatter(strikes, ivs, color='#00ffcc', edgecolors='white', marker='o', s=50, zorder=3)
plt.plot(strikes, ivs, color='#00ffcc', alpha=0.4, linewidth=2, zorder=2)

# Draw a red line exactly where the stock is trading right now (S)
plt.axvline(x=S, color='#ff3333', linestyle='--', linewidth=2, label=f'Live Spot Price: ${S:.2f}', zorder=1)

# Format the UI to look like a professional Quant terminal
plt.title('Hessian-AI: Volatility Smile (SPY Calls)', fontsize=16, fontweight='bold', color='white')
plt.xlabel('Strike Price (K)', fontsize=12, color='white')
plt.ylabel('Implied Volatility (%)', fontsize=12, color='white')
plt.tick_params(colors='white')
plt.grid(True, linestyle='--', alpha=0.2, color='white')
plt.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')

plt.show()

# %%



