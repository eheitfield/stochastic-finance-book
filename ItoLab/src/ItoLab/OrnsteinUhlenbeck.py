"""Ornstein–Uhlenbeck (OU) process simulator.

The OU process is a mean-reverting Gaussian process governed by the SDE

.. math::
    dx_t = \\kappa(\\theta - x_t)\\,dt + \\sigma\\,dW_t.

It is simulated here using the **exact** transition distribution, so no
Euler discretisation error is introduced.
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess


__all__ = ["OrnsteinUhlenbeck"]


class OrnsteinUhlenbeck(StochasticProcess):
    """Ornstein–Uhlenbeck mean-reverting process.

    Parameters
    ----------
    x0 : float, default 0.05
        Initial value :math:`x_0`.
    theta : float, default 0.05
        Long-run mean level.
    kappa : float, default 0.9
        Mean-reversion speed.
    sigma : float, default 0.02
        Volatility parameter.

    Examples
    --------
    >>> from ItoLab import OrnsteinUhlenbeck
    >>> ou = OrnsteinUhlenbeck(x0=0.05, theta=0.05, kappa=0.9, sigma=0.02)
    >>> paths = ou.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
    >>> np.round(paths.at(0.0), 4)
    array([0.05, 0.05, 0.05])
    """

    def __init__(
        self,
        x0: float = 0.05,
        theta: float = 0.05,
        kappa: float = 0.9,
        sigma: float = 0.02,
    ):
        self.x0 = x0
        self.theta = theta
        self.kappa = kappa
        self.sigma = sigma

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Simulate sample paths of an OU process using the exact transition.

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
            Simulated OU paths.  All paths start at ``x0``.

        Examples
        --------
        >>> from ItoLab import OrnsteinUhlenbeck
        >>> ou = OrnsteinUhlenbeck(x0=0.05, theta=0.05, kappa=0.9, sigma=0.02)
        >>> sp = ou.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 6)  # doctest: +SKIP
        array([0.059436, 0.050179, 0.060306])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        ekdt = np.exp(-self.kappa * dt)
        sv = np.sqrt(
            self.sigma**2 * (1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa)
        )
        z = rng.normal(size=[n_paths, n_inc - 1])
        paths = np.zeros(shape=(n_paths, n_inc))
        paths[:, 0] = np.full(shape=n_paths, fill_value=self.x0)
        for i in range(1, n_inc):
            paths[:, i] = (
                self.theta + (paths[:, i - 1] - self.theta) * ekdt + sv * z[:, i - 1]
            )
        return SamplePaths(paths, t)
