import numpy as np


def gbm_vasicek_futures_price(
    S: float,
    r: float,
    T: float,
    t: float,
    kappa: float,
    theta: float,
    sigma_r: float,
    sigma_S: float,
    rho: float,
) -> float:
    """
    Futures price under a GBM-Vasicek model.

    Under Q:
        dS/S = r dt + sigma_S dW_S
        dr   = kappa(theta - r) dt + sigma_r dW_r

    with
        dW_S dW_r = rho dt.

    Parameters
    ----------
    S : float
        Current asset price S_t.
    r : float
        Current short rate r_t.
    T : float
        Futures expiration.
    t : float
        Current time.
    kappa : float
        Vasicek mean-reversion speed.
    theta : float
        Vasicek long-run mean under Q.
    sigma_r : float
        Volatility of the short rate.
    sigma_S : float
        Volatility of the asset.
    rho : float
        Correlation between the Wiener processes.

    Returns
    -------
    float
        Futures price F_t = E_t^Q[S_T].
    """

    tau: float = T - t

    # Vasicek bond-pricing function
    B: float = (1 - np.exp(-kappa * tau)) / kappa

    # Mean accumulated short rate
    m_I: float = theta * tau + (r - theta) * B

    # Variance of accumulated short rate
    V_I: float = (
        sigma_r**2
        / kappa**2
        * (tau - 2 * B + (1 - np.exp(-2 * kappa * tau)) / (2 * kappa))
    )

    # Covariance between accumulated short rate and stock Wiener shock
    C_I: float = rho * sigma_r / kappa * (tau - B)

    # Futures price
    F: float = S * np.exp(m_I + 0.5 * V_I + sigma_S * C_I)

    return F
