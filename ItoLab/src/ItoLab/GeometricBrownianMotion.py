"""Geometric Brownian motion (GBM) simulator.

GBM is the standard model for asset prices in the Black-Scholes framework.
Paths are generated via the closed-form solution:

.. math::
    S_t = S_0 \\exp\\!\\left( \\mu t - \\tfrac{1}{2}\\sigma^2 t + \\sigma W_t \\right)
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess
from .Wiener import Wiener


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian motion process.

    The SDE is :math:`dS_t = \\mu S_t\\, dt + \\sigma S_t\\, dW_t`.
    Simulated directly via the closed-form solution to avoid
    discretization error.

    Parameters
    ----------
    s0 : float, default 1.0
        Initial value :math:`S_0`.
    mu : float, default 0.0
        Drift (expected return) parameter.
    sigma : float, default 1.0
        Volatility parameter.

    Examples
    --------
    >>> from ItoLab import GeometricBrownianMotion
    >>> gbm = GeometricBrownianMotion(s0=100, mu=0.05, sigma=0.2)
    >>> paths = gbm.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
    >>> np.round(paths.at(0.0), 4)
    array([100., 100., 100.])
    """

    def __init__(
        self,
        s0: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
    ):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Simulate sample paths of a GBM process.

        Uses the closed-form solution so that no discretization bias is
        introduced regardless of ``n_inc``.

        Parameters
        ----------
        n_paths : int, default 100
            Number of independent path realizations.
        n_inc : int, default 1000
            Number of time steps.
        max_T : float, default 1.0
            Terminal time.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        SamplePaths
            Simulated GBM paths with ``S_0 = s0`` at every path.

        Examples
        --------
        >>> from ItoLab import GeometricBrownianMotion
        >>> gbm = GeometricBrownianMotion(s0=100, mu=0.05, sigma=0.2)
        >>> sp = gbm.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 4)  # doctest: +SKIP
        array([116.6973,  94.972 , 119.8782])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        w = Wiener().generate_paths(
            n_paths=n_paths,
            n_inc=n_inc,
            max_T=max_T,
            seed=seed,
        )
        y = (self.mu - 0.5 * self.sigma**2) * t.reshape(1, n_inc) + self.sigma * w.paths
        paths = self.s0 * np.exp(y)
        return SamplePaths(paths, t)
