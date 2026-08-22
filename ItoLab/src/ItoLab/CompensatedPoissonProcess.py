"""Compensated (zero-drift) Poisson process simulator.

A compensated Poisson process :math:`\\tilde{N}_t = N_t - \\lambda t`
subtracts the deterministic mean :math:`\\lambda t` from the standard
Poisson process, yielding a martingale with mean zero.
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess


class CompensatedPoissonProcess(StochasticProcess):
    """Compensated Poisson process (martingale).

    Parameters
    ----------
    lam : float, default 1.0
        Jump intensity (expected number of events per unit time).

    Examples
    --------
    >>> from ItoLab import CompensatedPoissonProcess
    >>> cpp = CompensatedPoissonProcess(lam=2.0)
    >>> paths = cpp.generate_paths(n_paths=3, n_inc=50, max_T=1.0, seed=42917)
    >>> paths.at(0.0)  # at t=0, compensated process is 0
    array([0., 0., 0.])
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
        """Simulate sample paths of a compensated Poisson process.

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
            Simulated compensated Poisson paths.  At each time *t*
            the expected value is zero.

        Examples
        --------
        >>> from ItoLab import CompensatedPoissonProcess
        >>> cpp = CompensatedPoissonProcess(lam=2.0)
        >>> sp = cpp.generate_paths(n_paths=3, n_inc=50, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 4)  # doctest: +SKIP
        array([ 2., -2.,  0.])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        dN = rng.poisson(self.lam * dt, size=(n_paths, n_inc - 1))
        paths = np.zeros(shape=[n_paths, n_inc])
        paths[:, 1:] = np.cumsum(dN, axis=1)
        paths = paths - np.tile(t * self.lam, (n_paths, 1))
        return SamplePaths(paths, t)
