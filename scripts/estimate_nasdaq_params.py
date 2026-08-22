"""
Estimate parameters for the GBM and Merton jump-diffusion models
from historical equity index data.

Input:
    CSV file containing columns:
        date
        index

Output:
    - Printed parameter table
    - nasdaq_parameters.json
"""

import json
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import least_squares
from scipy.stats import skew, kurtosis


# ==============================================================
# User settings
# ==============================================================

TRADING_DAYS = 252

# Hold jump intensity fixed
lambda_jump = 1.0  # expected jumps per year


# ==============================================================
# File paths
# ==============================================================

# Project root
ROOT = Path(__file__).resolve().parents[1]
# Directory for working data
DATA_DIR = ROOT / "scripts" / "data"
# Files
INPUT_FILE = DATA_DIR / "nasdaq.csv"
OUTPUT_FILE = DATA_DIR / "nasdaq_params.json"


# ==============================================================
# Read data
# ==============================================================

df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
prices = df["index"]
# Daily continuously-compounded returns
returns = df["daily_return"]
dt = 1 / TRADING_DAYS


# ==============================================================
# Sample moments
# ==============================================================

mean_daily = returns.mean()
var_daily = returns.var(ddof=1)
std_daily = returns.std(ddof=1)
skew_daily = skew(returns, bias=False)
excess_kurt_daily = kurtosis(returns, fisher=True, bias=False)

# Annualized GBM parameters
mu_observed = mean_daily * TRADING_DAYS
sigma = std_daily * np.sqrt(TRADING_DAYS)


# ==============================================================
# Estimate jump parameters
# ==============================================================

brown_var = sigma**2 * dt


def objective(theta):
    alpha, delta = theta
    jump_var = lambda_jump * dt * (alpha**2 + delta**2)
    total_var = brown_var + jump_var
    jump_k3 = lambda_jump * dt * (alpha**3 + 3 * alpha * delta**2)
    jump_k4 = lambda_jump * dt * (alpha**4 + 6 * alpha**2 * delta**2 + 3 * delta**4)
    model_skew = jump_k3 / total_var**1.5
    model_excess_kurt = jump_k4 / total_var**2
    return np.array(
        [
            model_skew - skew_daily,
            model_excess_kurt - excess_kurt_daily,
        ]
    )


initial_guess = np.array([-0.03, 0.05])
result = least_squares(objective, initial_guess)
alpha, delta = result.x


# ==============================================================
# Drift adjustment
# ==============================================================

k = np.exp(alpha + 0.5 * delta**2) - 1
mu_sde = mu_observed - lambda_jump * k


# ==============================================================
# Store results
# ==============================================================

results = {
    # -------------------------
    # Data summary
    # -------------------------
    "observations": int(len(returns)),
    "trading_days": TRADING_DAYS,
    # -------------------------
    # Daily moments
    # -------------------------
    "mean_daily": float(mean_daily),
    "variance_daily": float(var_daily),
    "volatility_daily": float(std_daily),
    "skewness_daily": float(skew_daily),
    "excess_kurtosis_daily": float(excess_kurt_daily),
    # -------------------------
    # GBM parameters
    # -------------------------
    "mu_observed": float(mu_observed),
    "sigma": float(sigma),
    # -------------------------
    # Merton parameters
    # -------------------------
    "lambda": float(lambda_jump),
    "alpha": float(alpha),
    "delta": float(delta),
    "k": float(k),
    "mu_sde": float(mu_sde),
}


# ==============================================================
# Print summary
# ==============================================================

print()
print("=" * 60)
print("Parameter Estimates")
print("=" * 60)

print("\nData")
print(f"{'Observations':30s}{results['observations']:>15d}")
print(f"{'Trading days/year':30s}{TRADING_DAYS:>15d}")
print("\nDaily Log Returns")
print(f"{'Mean':30s}{results['mean_daily']:15.6f}")
print(f"{'Variance':30s}{results['variance_daily']:15.6f}")
print(f"{'Volatility':30s}{results['volatility_daily']:15.6f}")
print(f"{'Skewness':30s}{results['skewness_daily']:15.6f}")
print(f"{'Excess Kurtosis':30s}{results['excess_kurtosis_daily']:15.6f}")

print("\nGBM")
print(f"{'mu':30s}{results['mu_observed']:15.6f}")
print(f"{'sigma':30s}{results['sigma']:15.6f}")

print("\nMerton")
print(f"{'lambda':30s}{results['lambda']:15.6f}")
print(f"{'alpha':30s}{results['alpha']:15.6f}")
print(f"{'delta':30s}{results['delta']:15.6f}")
print(f"{'Expected jump (k)':30s}{results['k']:15.6f}")
print(f"{'SDE drift (mu)':30s}{results['mu_sde']:15.6f}")

print("\n")


# ==============================================================
# Save JSON
# ==============================================================

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print(f"Parameters written to {OUTPUT_FILE}")
