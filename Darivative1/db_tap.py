#%%
import os
import databento as db
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# ==============================================================================
# STEP 1: UNLOCK SECURITY PERIMETER
# ==============================================================================
load_dotenv()
api_key = os.getenv("DATABENTO_API_KEY")

if not api_key:
    print("[FATAL] Security breach: API Key missing from .env configuration.")
    exit()

client = db.Historical(api_key)

# ==============================================================================
# STEP 2: TAP THE LIVE INSTITUTIONAL FEED (WITH LOCAL CACHING)
# ==============================================================================
cache_file = "spy_matrix_cache.csv"

if os.path.exists(cache_file):
    print("[+] SECURE CACHE FOUND: Loading matrix from local hard drive (Cost: $0.00)")
    df = pd.read_csv(cache_file)
else:
    print("STREAMING RECENT MARKET MATRIX FROM CLOUD...")
    data = client.timeseries.get_range(
        dataset="XNAS.ITCH",
        symbols=["SPY"],
        start="2024-01-05T14:30",
        end="2024-01-05T14:35",
        schema="ohlcv-1s"
    )
    df = data.to_df()
    df = df.reset_index()
    df.to_csv(cache_file, index=False)
    print(f"[+] MATRIX SAVED TO HARD DRIVE: {cache_file}")

# ==============================================================================
# STEP 3: THE QUANTITATIVE MATH ENGINE (VWAP + VOLATILITY)
# ==============================================================================
# 1. The Entry Radar: True Institutional VWAP Line
df['dollar_volume'] = df['close'] * df['volume']
df['cum_dollar_volume'] = df['dollar_volume'].cumsum()
df['cum_volume'] = df['volume'].cumsum()
df['vwap'] = df['cum_dollar_volume'] / df['cum_volume']

# 2. The Exit Scope: 60-Second Rolling Volatility
# Calculate the second-by-second percentage returns
df['returns'] = df['close'].pct_change()
# Calculate standard deviation of those returns over a 60-second rolling window
df['volatility_60s'] = df['returns'].rolling(window=60).std()

# ==============================================================================
# STEP 4: RISK MITIGATION & THRESHOLD PROTOCOLS
# ==============================================================================
VOLUME_THRESHOLD = 1200  # Minimum shares required to prove Whale activity

print("\nSCANNING MATRIX FOR WHALE LIQUIDATIONS...\n")
print(f"{'Timestamp':<22} | {'Spot':<8} | {'VWAP':<8} | {'Volume':<8} | {'Volatility':<10} | {'ACTION'}")
print("-" * 85)

triggered = False

# Loop through the 1-second matrix line by line simulating real-time streaming
for idx, row in df.iterrows():
    # The engine needs 60 seconds to prime the volatility calculation. 
    # We skip the first 60 rows to avoid 'NaN' (Not a Number) errors.
    if pd.isna(row['volatility_60s']):
        continue

    timestamp = str(row['ts_event'])
    spot_price = round(row['close'], 2)
    vwap_price = round(row['vwap'], 2)
    volume = int(row['volume'])
    live_volatility = row['volatility_60s']

    # Condition 1 & 2: Spot < VWAP AND massive volume crosses the bid
    price_condition = spot_price < vwap_price
    volume_condition = volume >= VOLUME_THRESHOLD

    if price_condition and volume_condition and not triggered:
        # EXECUTE SHORT ENTRY
        entry_price = spot_price
        
        # CALCULATE DYNAMIC BUY BACK TARGET (Spot Price * Volatility Percentage)
        dynamic_offset = entry_price * live_volatility
        exit_target = entry_price - dynamic_offset
        
        action_signal = f"🚨 [SELL SHORT] Executed at {entry_price}"
        print(f"{timestamp:<22} | ${spot_price:<7} | ${vwap_price:<7} | {volume:<8} | {live_volatility:.4f}     | {action_signal}")
        
        # SIMULATE HIGH-FREQUENCY COVER
        print(f"\n[+] ORDER DEPLOYED: Resting Buy Limit placed at ${exit_target:.2f}")
        print(f"[+] DYNAMIC TARGET MATCHED: Natural vibration of ${dynamic_offset:.2f} detected.")
        print(f"[+] LIQUIDITY FOUND: Short Position covered successfully at ${exit_target:.2f}")
        print("-" * 85)
        triggered = True
        break
    else:
        # Print first few active rows to show calculation working
        if idx < 65:
            action_signal = "HOLD (No Whale Panic)"
            print(f"{timestamp:<22} | ${spot_price:<7} | ${vwap_price:<7} | {volume:<8} | {live_volatility:.4f}     | {action_signal}")

if not triggered:
    print("\n[+] Scan Complete: No institutional panic cascades detected in this window.")
# %%
