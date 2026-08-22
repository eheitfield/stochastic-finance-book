"""Standard Poisson process (counting process) simulator.

A Poisson process :math:`N_t` with intensity :math:`\\lambda` counts
the number of events occurring in the interval :math:`[0, t]`.  The
increments :math:`N_t - N_s` follow a Poisson distribution with
parameter :math:`\\lambda (t - s)`.
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess


class PoissonProcess(StochasticProcess):
    """Standard Poisson counting process.

    Parameters
    ----------
    lam : float, default 1.0
        Jump intensity (expected number of events per unit time).

    Examples
    --------
    >>> from ItoLab import PoissonProcess
    >>> pp = PoissonProcess(lam=2.0)
    >>> paths = pp.generate_paths(n_paths=3, n_inc=50, max_T=1.0, seed=42917)
    >>> paths.at(0.0)  # all paths start at 0
    array([0., 0., 0.])
    >>> paths.n_inc
    49
    """

    def __init__(self, lam: float = 1.0):
        self.lam = lam

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Simulate sample paths of a Poisson counting process.

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
            Simulated Poisson paths with ``N_0 = 0``.

        Examples
        --------
        >>> from ItoLab import PoissonProcess
        >>> pp = PoissonProcess(lam=2.0)
        >>> sp = pp.generate_paths(n_paths=3, n_inc=50, max_T=1.0, seed=42917)
        >>> sp.at(1.0)  # doctest: +SKIP
        array([4., 0., 2.])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        dN = rng.poisson(self.lam * dt, size=(n_paths, n_inc - 1))
        paths = np.zeros(shape=[n_paths, n_inc])
        paths[:, 1:] = np.cumsum(dN, axis=1)
        return SamplePaths(paths, t)
