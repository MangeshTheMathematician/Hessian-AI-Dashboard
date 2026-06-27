import os
import databento as db
import pandas as pd
from dotenv import load_dotenv

# ==============================================================================
# STEP 1: IGNITE THE SECURITY VAULT
# ==============================================================================
# load_dotenv() silently scans your root folder, finds the .env file,
# and unlocks your API key into memory without printing it to the screen.
load_dotenv()

print("IGNITING DATABENTO SNIPER SCRIPT...\n")

# Extract the key from the OS memory
api_key = os.getenv("DATABENTO_API_KEY")

# Hard-stop mechanism if the vault fails
if not api_key:
    print("[FATAL ERROR] API Key not found. Check your .env file location.")
    exit()

# ==============================================================================
# STEP 2: AUTHENTICATE THE INSTITUTIONAL CLIENT
# ==============================================================================
# Feed the hidden key directly into Databento's Historical engine.
client = db.Historical(api_key)
print("[+] Databento Client Authenticated Successfully.")

# ==============================================================================
# STEP 3: CONFIGURE THE DATA TAP
# ==============================================================================
try:
    print("[+] Requesting 1-Second Resolution Matrix...")

    # We are pulling exactly 1 minute of data to test the system safely.
    # Dataset: XNAS.ITCH (Nasdaq TotalView-ITCH equities feed)
    # Schema: ohlcv-1s (Open, High, Low, Close, Volume rolled up to 1-second bars)
    data = client.timeseries.get_range(
        dataset="XNAS.ITCH",       
        symbols=["SPY"],           
        start="2024-01-05T14:30",  # Market open (UTC)
        end="2024-01-05T14:31",    # Exactly 1 minute later
        schema="ohlcv-1s"          
    )

    # ==============================================================================
    # STEP 4: CONVERT AND DISPLAY
    # ==============================================================================
    # Convert the raw institutional byte-stream into a clean Pandas DataFrame
    df = data.to_df()

    print("\n[+] TARGET SECURED: 1-Second Matrix Downloaded")
    print("-" * 75)
    print(df.head())
    print("-" * 75)
    print("\n[SUCCESS] The Databento pipeline is fully operational.")

except Exception as e:
    print(f"\n[ERROR] The data tap failed. Reason: {e}")