import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from fredapi import Fred
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Output directory for working data
DATA_DIR = ROOT / "scripts" / "data"

# Configure FRED API access.
# Modify this code to use your own FRED API key.
# To obtain a free key, visit the St. Louis Fed's FRED website.
# https://fred.stlouisfed.org
load_dotenv(find_dotenv())
fred_api_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key=fred_api_key)


# ------------------------------------------------------------------------------
# Pull NASDAQ Index
# ------------------------------------------------------------------------------

nasdaq_raw = fred.get_series(
    series_id="NASDAQCOM", observation_start="1990-01-01", observation_end="2025-12-31"
)
nasdaq = pd.DataFrame(nasdaq_raw, columns=["index"])
nasdaq.index.name = "date"
nasdaq.dropna(inplace=True)
nasdaq["ln_index"] = np.log(nasdaq["index"])
nasdaq["daily_return"] = nasdaq["ln_index"].diff()
nasdaq.dropna(inplace=True)

nasdaq.to_csv(DATA_DIR / "nasdaq.csv")
print(f"NASDAQ data saved.")

# ------------------------------------------------------------------------------
# Pull 10-year Treasury Yield
# ------------------------------------------------------------------------------

ust_raw = fred.get_series(
    series_id="DGS10", observation_start="1960-01-01", observation_end="2025-12-31"
)
ust = pd.DataFrame(ust_raw, columns=["ust_10_yr"])
ust.index.name = "date"
ust["ust_10_yr"] = ust["ust_10_yr"] / 100
ust.dropna(inplace=True)

ust.to_csv(DATA_DIR / "ust_10_yr.csv")
print(f"10-year Treasury data saved.")

# ------------------------------------------------------------------------------
# Pull EFFR
# ------------------------------------------------------------------------------

effr_raw = fred.get_series(
    series_id="FEDFUNDS", observation_start="1960-01-01", observation_end="2025-12-31"
)
effr = pd.DataFrame(effr_raw, columns=["effr"])
effr.index.name = "date"
effr["effr"] = effr["effr"] / 100
effr.dropna(inplace=True)

effr.to_csv(DATA_DIR / "effr.csv")
print(f"Federal funds rate data saved.")

# ------------------------------------------------------------------------------
# Pull Zero Yields
# ------------------------------------------------------------------------------

for i in range(1, 11):
    series_ID = f"THREEFY{i}"
    col_name = f"zero_yld_{i}yr"
    raw_series = fred.get_series(
        series_id=series_ID, obsevation_start="1990-01-01", observation_end="2025-12-31"
    )
    df1 = pd.DataFrame(raw_series, columns=[col_name])
    df1[col_name] = df1[col_name] / 100
    df1.index.name = "date"
    if i is 1:
        df0 = df1
    else:
        df0 = pd.merge(df0, df1, left_index=True, right_index=True, how="outer")
df0.dropna()

df0.to_csv(DATA_DIR / "zero_yields.csv")
print(f"Zero coupon yield data saved.")


print("Done.")
