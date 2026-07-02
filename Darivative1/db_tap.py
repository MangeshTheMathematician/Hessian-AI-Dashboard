# Import argparse so the user can change symbol, date, and output path from the terminal.
import argparse

# Import os so the script can read environment variables and check whether cache files exist.
import os

# Import pandas because Databento data is stored and saved as a DataFrame.
import pandas as pd

# Import load_dotenv so the script can read DATABENTO_API_KEY from a local .env file.
from dotenv import load_dotenv

# Import databento because this script pulls historical 1-second OHLCV data from Databento.
import databento as db


# Define the command-line interface for the data tap.
def parse_args() -> argparse.Namespace:
    # Create the argument parser with a clear description.
    parser = argparse.ArgumentParser(description="Fetch Databento 1-second OHLCV data and cache it locally.")

    # Let the user choose the Databento dataset.
    parser.add_argument("--dataset", default="XNAS.ITCH", help="Databento dataset, for example XNAS.ITCH.")

    # Let the user choose the ticker symbol.
    parser.add_argument("--symbol", default="SPY", help="Symbol to fetch, for example SPY.")

    # Let the user choose the start timestamp.
    parser.add_argument("--start", default="2024-01-05T14:30", help="Start timestamp in Databento-compatible format.")

    # Let the user choose the end timestamp.
    parser.add_argument("--end", default="2024-01-05T14:35", help="End timestamp in Databento-compatible format.")

    # Let the user choose the Databento schema.
    parser.add_argument("--schema", default="ohlcv-1s", help="Databento schema, for example ohlcv-1s.")

    # Let the user choose the output cache path.
    parser.add_argument("--cache-file", default="spy_matrix_cache.csv", help="Local CSV cache path.")

    # Let the user force a fresh download even if the cache exists.
    parser.add_argument("--force-refresh", action="store_true", help="Download again even if cache file already exists.")

    # Return the parsed arguments.
    return parser.parse_args()


# Define the main execution function.
def main() -> None:
    # Parse terminal arguments.
    args = parse_args()

    # Load DATABENTO_API_KEY from .env if present.
    load_dotenv()

    # Read the API key from environment variables.
    api_key = os.getenv("DATABENTO_API_KEY")

    # Stop early if the user forgot to configure the Databento key.
    if not api_key:
        raise RuntimeError("DATABENTO_API_KEY is missing. Add it to your .env file before running this script.")

    # If the cache exists and the user did not force refresh, load the local file instead of hitting Databento again.
    if os.path.exists(args.cache_file) and not args.force_refresh:
        # Print a message that makes it obvious this run is free and local.
        print(f"[+] Cache found: loading {args.cache_file} without a new API request.")

        # Load the cached file to prove it is readable.
        df = pd.read_csv(args.cache_file)

        # Print the shape so the user knows how many rows and columns are available.
        print(f"[+] Cached matrix shape: {df.shape[0]} rows x {df.shape[1]} columns")

        # Print the first few rows for visual verification.
        print(df.head())

        # Return because no download is needed.
        return

    # Create the Databento historical client using the secure API key.
    client = db.Historical(api_key)

    # Print exactly what is being requested so billing and data scope are transparent.
    print(f"[+] Requesting {args.symbol} from {args.dataset}, {args.start} to {args.end}, schema={args.schema}")

    # Pull the historical time-series range from Databento.
    data = client.timeseries.get_range(
        dataset=args.dataset,
        symbols=[args.symbol],
        start=args.start,
        end=args.end,
        schema=args.schema,
    )

    # Convert the Databento response into a pandas DataFrame.
    df = data.to_df().reset_index()

    # Save the matrix locally so future dashboard runs do not need another API call.
    df.to_csv(args.cache_file, index=False)

    # Print confirmation of where the cache was saved.
    print(f"[+] Saved matrix to {args.cache_file}")

    # Print the shape of the saved data.
    print(f"[+] Matrix shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Print the first rows for quick validation.
    print(df.head())


# Run main only when this file is executed directly.
if __name__ == "__main__":
    # Execute the data puller.
    main()
