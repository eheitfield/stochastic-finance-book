"""Merton jump-diffusion process simulator.

The Merton (1976) jump-diffusion model combines a continuous
geometric Brownian motion component with a compound Poisson jump
component whose jump sizes are log-normally distributed.  The price
process follows

.. math::
    S_t = S_0 \\exp\\!\\left( \\mu t + \\sigma W_t + J_t \\right)

where :math:`J_t` is a compound Poisson process with log-normal jumps.
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess
from .GeometricBrownianMotion import GeometricBrownianMotion
from .CompoundPoissonProcess import CompoundPoissonProcess
from .JumpType import JumpType


class MertonJumpProcess(StochasticProcess):
    """Merton jump-diffusion process.

    The continuous part is a GBM with drift ``mu`` and volatility
    ``sigma``.  The jump part is a compound Poisson process with
    intensity ``eta`` and normally distributed log-jumps with mean
    ``alpha`` and standard deviation ``delta``.

    Parameters
    ----------
    s0 : float, default 1.0
        Initial asset price :math:`S_0`.
    mu : float, default 0.0
        Drift of the continuous (GBM) component.
    sigma : float, default 1.0
        Volatility of the continuous (GBM) component.
    eta : float, default 1.0
        Jump intensity (expected number of jumps per unit time).
    alpha : float, default 0.0
        Mean of the log-jump size distribution.
    delta : float, default 1.0
        Standard deviation of the log-jump size distribution.

    Examples
    --------
    >>> from ItoLab import MertonJumpProcess
    >>> mjd = MertonJumpProcess(
    ...     s0=100, mu=0.05, sigma=0.2, eta=1.0, alpha=0.0, delta=0.1
    ... )
    >>> paths = mjd.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
    >>> np.round(paths.at(0.0), 4)
    array([100., 100., 100.])
    """

    def __init__(
        self,
        s0: float = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        eta: float = 1.0,
        alpha: float = 0.0,
        delta: float = 1.0,
    ):
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma
        self.eta = eta
        self.alpha = alpha
        self.delta = delta

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Simulate sample paths of a Merton jump-diffusion process.

        The path is constructed as
        ``GBM_paths * exp(jump_paths)``, where the GBM component uses
        ``seed`` and the compound Poisson jump component uses
        ``seed + 123456`` to avoid correlation.

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
            Simulated Merton jump-diffusion paths.  All paths start at
            ``s0``.

        Examples
        --------
        >>> from ItoLab import MertonJumpProcess
        >>> mjd = MertonJumpProcess(
        ...     s0=100, mu=0.05, sigma=0.2, eta=1.0, alpha=0.0, delta=0.1
        ... )
        >>> sp = mjd.generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 4)  # doctest: +SKIP
        array([116.6973,  94.972 , 117.9331])
        """
        seed2 = seed
        if seed2:
            seed2 = seed + 123456
        t = np.linspace(0.0, max_T, num=n_inc)
        # GBM path generated directly from GBM process
        gbm = GeometricBrownianMotion(s0=self.s0, mu=self.mu, sigma=self.sigma)
        gbm_paths = gbm.generate_paths(
            n_paths=n_paths, n_inc=n_inc, max_T=max_T, seed=seed
        )
        # jump multipliers generated from CPP process
        j = CompoundPoissonProcess(
            lam=self.eta,
            jump_distribution=JumpType.NORMAL,
            loc=self.alpha,
            scale=self.delta,
        )
        dJ = j.generate_paths(n_paths=n_paths, n_inc=n_inc, max_T=max_T, seed=seed2)
        j_paths = dJ.paths
        # put the processes together
        paths = gbm_paths.paths * np.exp(j_paths)
        return SamplePaths(paths, t)
