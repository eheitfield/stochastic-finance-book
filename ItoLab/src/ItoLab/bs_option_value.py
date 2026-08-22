"""Black–Scholes closed-form option pricing formula.

Prices European call and put options on a non-dividend-paying stock
under the standard Black–Scholes–Merton assumptions.
"""

import numpy as np
from scipy.stats import norm


def bs_option_value(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option: str = "call",
) -> float | np.ndarray:
    """Price a European option using the Black–Scholes formula.

    Parameters
    ----------
    S : float or ndarray
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    r : float
        Continuously compounded risk-free interest rate (annualized).
    sigma : float
        Volatility of the underlying (annualized).
    option : {"call", "put"}
        Option type.

    Returns
    -------
    float or ndarray
        Black–Scholes option price.

    Examples
    --------
    >>> from ItoLab import bs_option_value
    >>> round(float(bs_option_value(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option="call")), 6)
    10.450584
    >>> round(float(bs_option_value(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option="put")), 6)
    5.573526
    """
    S = np.asarray(S)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option.lower() == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    elif option.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    else:
        raise ValueError("option must be 'call' or 'put'")
