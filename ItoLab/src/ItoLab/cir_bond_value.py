"""Cox–Ingersoll–Ross zero-coupon bond pricer.

Under the CIR model the zero-coupon bond price has the affine form

.. math::
    P(t, T) = A(\\tau)\\, e^{-B(\\tau)\\, r_t}

where :math:`\\tau = T - t` is the time to maturity and the CIR-specific
``A`` and ``B`` coefficients are functions of the model parameters.
"""

import numpy as np


def cir_bond_value(
    r_t: float,
    tau: float,
    kappa_q: float,
    theta_q: float,
    sigma: float,
    as_yield: bool = False,
) -> float:
    """Price a zero-coupon bond under the CIR model.

    Parameters
    ----------
    r_t : float
        Current short rate.
    tau : float
        Time to maturity (T - t), in years.
    kappa_q : float
        Mean reversion speed under the risk-neutral measure.
    theta_q : float
        Long-run mean under the risk-neutral measure.
    sigma : float
        Volatility parameter.
    as_yield : bool, default False
        If ``True``, return the continuously compounded yield instead of
        the price.

    Returns
    -------
    float
        Zero-coupon bond price, or the equivalent yield if
        ``as_yield=True``.

    Examples
    --------
    >>> from ItoLab import cir_bond_value
    >>> float(round(cir_bond_value(r_t=0.05, tau=1.0, kappa_q=0.9, theta_q=0.05, sigma=0.02), 6))
    0.951231
    >>> round(cir_bond_value(r_t=0.05, tau=1.0, kappa_q=0.9, theta_q=0.05, sigma=0.02, as_yield=True), 6)  # doctest: +SKIP
    0.050037
    """
    gamma = np.sqrt(kappa_q**2 + 2.0 * sigma**2)
    exp_gamma_tau = np.exp(gamma * tau)
    denominator = (gamma + kappa_q) * (exp_gamma_tau - 1.0) + 2.0 * gamma
    B = 2.0 * (exp_gamma_tau - 1.0) / denominator
    A = (2.0 * gamma * np.exp((kappa_q + gamma) * tau / 2.0) / denominator) ** (
        2.0 * kappa_q * theta_q / sigma**2
    )
    V = A * np.exp(-B * r_t)
    if not as_yield:
        return V
    else:
        return V ** (-1 / tau) - 1
