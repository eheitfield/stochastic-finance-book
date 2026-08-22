"""European option pricing under the Merton jump-diffusion model."""

import numpy as np
from scipy.special import gammaln

from .bs_option_value import bs_option_value


def merton_option_value(
    S: float | np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lam: float,
    alpha: float,
    delta: float,
    option: str = "call",
    n_terms: int = 50,
) -> float | np.ndarray:
    """
    Price a European option under the Merton jump-diffusion model.

    Parameters
    ----------
    S : float or ndarray
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Diffusion volatility.
    lam : float
        Jump arrival intensity.
    alpha : float
        Mean of log jump size.
    delta : float
        Standard deviation of log jump size.
    option : {"call", "put"}
        Option type.
    n_terms : int, default=50
        Number of Poisson terms used in the approximation.

    Returns
    -------
    float or ndarray
        European option value.
    """

    S = np.asarray(S, dtype=float)

    # Expected proportional jump size
    k = np.exp(alpha + 0.5 * delta**2) - 1.0

    price = np.zeros_like(S, dtype=float)

    for n in range(n_terms):

        # Poisson probability
        log_weight = -lam * T + n * np.log(lam * T) - gammaln(n + 1)
        weight = np.exp(log_weight)

        # Effective spot price
        S_n = S * np.exp(n * alpha + 0.5 * n * delta**2 - lam * k * T)

        # Effective volatility
        sigma_n = np.sqrt(sigma**2 + n * delta**2 / T)

        price += weight * bs_option_value(
            S=S_n,
            K=K,
            T=T,
            r=r,
            sigma=sigma_n,
            option=option,
        )

    return price
