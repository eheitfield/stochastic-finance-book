"""Wiener process (standard Brownian motion) simulator.

A Wiener process :math:`W_t` is a continuous-time stochastic process
with independent, stationary increments such that :math:`W_t - W_s`
is normally distributed with mean 0 and variance :math:`t - s`.
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess


class Wiener(StochasticProcess):
    """Standard Wiener process (Brownian motion).

    The process starts at zero :math:`W_0 = 0` and has independent
    Gaussian increments :math:`dW_t \\sim N(0, dt)`.

    This class is often used internally by other models (e.g.
    :class:`~ItoLab.GeometricBrownianMotion`) but can be instantiated
    on its own for general Brownian-motion simulations.

    Examples
    --------
    >>> from ItoLab import Wiener
    >>> w = Wiener()
    >>> paths = w.generate_paths(n_paths=5, n_inc=100, max_T=1.0, seed=42917)
    >>> paths.paths[:, 0]  # all paths start at 0
    array([0., 0., 0., 0., 0.])
    >>> paths.n_inc
    99
    """

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Simulate sample paths of a standard Wiener process.

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
            Simulated Wiener paths with ``W_0 = 0``.

        Examples
        --------
        >>> from ItoLab import Wiener
        >>> sp = Wiener().generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 4)  # doctest: +SKIP
        array([-0.0899, -0.4902,  0.4759])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        dw = rng.normal(0, np.sqrt(dt), size=[n_paths, n_inc - 1])
        paths = np.zeros(shape=[n_paths, n_inc])
        paths[:, 1:] = np.cumsum(dw, axis=1)
        return SamplePaths(paths, t)
