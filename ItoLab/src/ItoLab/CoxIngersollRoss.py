"""Cox–Ingersoll–Ross (CIR) process simulator.

The CIR process is a mean-reverting, square-root diffusion widely used
for interest-rate and volatility modelling.  It is simulated here using
the **exact** transition distribution (scaled non-central chi-squared),
which avoids the discretisation bias of Euler–Maruyama schemes.
"""

from typing import Optional

import numpy as np
from scipy import stats

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess


class CoxIngersollRoss(StochasticProcess):
    """Cox–Ingersoll–Ross (square-root diffusion) process.

    The SDE is :math:`dr_t = \\kappa(\\theta - r_t)\\,dt + \\sigma\\sqrt{r_t}\\,dW_t`.
    Simulation uses the exact conditional distribution of :math:`r_{t+dt}`
    given :math:`r_t`, which is a scaled non-central chi-squared.

    Parameters
    ----------
    r0 : float, default 0.05
        Initial value :math:`r_0`.
    theta : float, default 0.05
        Long-run mean level.
    kappa : float, default 0.9
        Mean-reversion speed.
    sigma : float, default 0.02
        Volatility parameter.

    Examples
    --------
    >>> from ItoLab import CoxIngersollRoss
    >>> cir = CoxIngersollRoss(r0=0.05, theta=0.05, kappa=0.9, sigma=0.02)
    >>> paths = cir.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
    >>> np.round(paths.at(0.0), 4)
    array([0.05, 0.05, 0.05])
    """

    def __init__(self, r0: float = 0.05, theta: float = 0.05,
                 kappa: float = 0.9, sigma: float = 0.02):
        self.r0 = r0
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
        """Simulate sample paths of a CIR process using the exact transition.

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
            Simulated CIR paths.  All paths start at ``r0``.

        Examples
        --------
        >>> from ItoLab import CoxIngersollRoss
        >>> cir = CoxIngersollRoss(r0=0.05, theta=0.05, kappa=0.9, sigma=0.02)
        >>> sp = cir.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 6)  # doctest: +SKIP
        array([0.053906, 0.044929, 0.049137])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        paths = np.zeros(shape=(n_paths, n_inc))
        paths[:, 0] = np.full(shape=n_paths, fill_value=self.r0)
        exp_term = np.exp(-self.kappa * dt)
        c = self.sigma**2 * (1.0 - exp_term) / (4.0 * self.kappa)
        df = 4.0 * self.kappa * self.theta / self.sigma**2
        for i in range(1, n_inc):
            nc = 4.0 * self.kappa * exp_term * paths[:, i - 1] / (
                self.sigma**2 * (1.0 - exp_term)
            )
            paths[:, i] = c * stats.ncx2.rvs(df=df, nc=nc, random_state=rng)
        return SamplePaths(paths, t)
