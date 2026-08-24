"""Wiener process (standard Brownian motion) simulator.

A Wiener process :math:`W_t` is a continuous-time stochastic process
with independent, stationary increments such that :math:`W_t - W_s`
is normally distributed with mean 0 and variance :math:`t - s`.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from .SamplePaths import SamplePaths
from .MultivariateStochasticProcess import MultivariateStochasticProcess


class Wiener(MultivariateStochasticProcess):
    """Correlated Wiener process (Brownian motion).

    The process starts at zero :math:`W_0 = 0` and has independent
    Gaussian increments :math:`dW_t \\sim N(0, dt)`.

    This class is often used internally by other models (e.g.
    :class:`~ItoLab.GeometricBrownianMotion`) but can be instantiated
    on its own for general Brownian-motion simulations.

    """

    def generate_paths(
        self,
        corr: npt.NDArray | float | None = None,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> list[SamplePaths]:
        """Simulate sample paths of a standard Wiener process.

        Parameters
        ----------
        corr : NDArray | float | None
            2-D square correlation matrix.  Should be symmetric, positive
            definite with ones along the main diagonal.  If a float is given
            then assume bivariate with rho=corr.  If None then rho=0.
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
        list[SamplePaths]
            List of simulated Wiener paths for each variable.

        """
        if not corr:
            corr = np.array([1.0, 0.0], [0.0, 1.0])
        elif isinstance(corr, float):
            rho = corr
            corr = np.array([1.0, rho], [rho, 1.0])
        n_var = corr.shape[0]
        # FIX FROM HERE
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        dw = rng.normal(0, np.sqrt(dt), size=[n_paths, n_inc - 1])
        paths = np.zeros(shape=[n_paths, n_inc])
        paths[:, 1:] = np.cumsum(dw, axis=1)
        return SamplePaths(paths, t)
