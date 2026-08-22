"""Vasicek (Ornstein–Uhlenbeck) zero-coupon bond pricer.

Under the Vasicek model the bond price has the affine form

.. math::
    P(t, T) = A(\\tau)\\, e^{-B(\\tau)\\, r_t}

where :math:`\\tau = T - t` and the Vasicek ``A`` and ``B`` coefficients
are functions of the mean-reversion speed ``kappa``, the long-run mean
``theta_q``, and the volatility ``sigma``.
"""

import numpy as np


def vasicek_bond_value(
    r_t: float,
    tau: float,
    kappa: float,
    theta_q: float,
    sigma: float,
    as_yield: bool = False,
) -> float:
    """Compute the price of a zero-coupon bond under the Vasicek model.

    Parameters
    ----------
    r_t : float
        Current short rate.
    tau : float
        Time to maturity (T - t), in years.
    kappa : float
        Mean reversion speed.
    theta_q : float
        Long-run mean under the risk-neutral measure.
    sigma : float
        Volatility.
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
    >>> from ItoLab import vasicek_bond_value
    >>> float(round(vasicek_bond_value(r_t=0.05, tau=1.0, kappa=0.9, theta_q=0.05, sigma=0.02), 6))
    0.951263
    >>> round(vasicek_bond_value(r_t=0.05, tau=1.0, kappa=0.9, theta_q=0.05, sigma=0.02, as_yield=True), 6)  # doctest: +SKIP
    0.050088
    """
    # Affine coefficient B(t,T)
    B = (1.0 - np.exp(-kappa * tau)) / kappa
    # Affine coefficient A(t,T)
    exponent = (theta_q - sigma**2 / (2.0 * kappa**2)) * (B - tau) - (
        sigma**2 / (4.0 * kappa)
    ) * B**2
    A = np.exp(exponent)
    # Bond price
    V = A * np.exp(-B * r_t)
    if not as_yield:
        return V
    else:
        return -np.log(V) / tau
