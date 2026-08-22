"""Compound Poisson process simulator.

A compound Poisson process combines a Poisson jump-count process with
random jump sizes drawn from an arbitrary distribution.  The path at
time :math:`t` is the cumulative sum of all realized jumps up to
:math:`t`.
"""

from typing import Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess
from .JumpType import JumpType


class CompoundPoissonProcess(StochasticProcess):
    """Compound Poisson process with configurable jump-size distribution.

    At each time step, the number of jumps is drawn from a Poisson
    distribution with parameter ``lam * dt``.  Each jump's size is then
    sampled from the selected ``jump_distribution``.

    Parameters
    ----------
    lam : float
        Poisson jump intensity parameter (expected number of jumps
        per unit time).
    jump_distribution : JumpType
        Distribution from which jump sizes are drawn.
    **kwargs
        Named parameters forwarded to the corresponding NumPy Generator
        method (e.g. ``loc`` and ``scale`` for ``NORMAL``, ``scale`` for
        ``EXPONENTIAL``, ``a`` and ``scale`` for ``GAMMA``).  See
        https://numpy.org/doc/stable/reference/random/generator.html
        for full parameter documentation.

    Examples
    --------
    >>> from ItoLab import CompoundPoissonProcess, JumpType
    >>> cpp = CompoundPoissonProcess(
    ...     lam=1.0, jump_distribution=JumpType.NORMAL, loc=0.0, scale=0.1
    ... )
    >>> paths = cpp.generate_paths(n_paths=3, n_inc=50, max_T=1.0, seed=42917)
    >>> paths.at(0.0)  # starts at 0
    array([0., 0., 0.])
    """

    def __init__(
        self,
        lam: float,
        jump_distribution: JumpType,
        **kwargs,
    ):
        self.lam = lam
        self.jump_distribution = jump_distribution
        self.jump_parameters = kwargs

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Simulate sample paths of a compound Poisson process.

        Parameters
        ----------
        n_paths : int, default 100
            Number of independent path realizations.
        n_inc : int, default 1000
            Number of time steps.
        max_T : float, default 1.0
            Terminal time.
        seed : int, optional
            Random seed for reproducibility.  A slightly offset seed
            is used internally for the jump-size distribution so that
            jump counts and sizes are not correlated.

        Returns
        -------
        SamplePaths
            Simulated compound Poisson paths.

        Examples
        --------
        >>> from ItoLab import CompoundPoissonProcess, JumpType
        >>> cpp = CompoundPoissonProcess(
        ...     lam=1.0, jump_distribution=JumpType.NORMAL, loc=0.0, scale=0.1
        ... )
        >>> sp = cpp.generate_paths(n_paths=3, n_inc=50, max_T=1.0, seed=42917)
        >>> np.round(sp.at(1.0), 4)  # doctest: +SKIP
        array([-0.0069,  0.0000, -0.0014])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        dt = max_T / n_inc
        rng = np.random.default_rng(seed)
        # create offset seed for jump distribution
        if seed:
            seed_jump = seed + 12345
        else:
            seed_jump = None
        rng_jump = np.random.default_rng(seed_jump)
        dN = np.round(
            rng.poisson(self.lam * dt, size=(n_paths, n_inc - 1))
        ).astype(int)

        def draw_y(size):
            match self.jump_distribution:
                case JumpType.NORMAL:
                    return rng_jump.normal(size=size, **self.jump_parameters)
                case JumpType.GAMMA:
                    return rng_jump.gamma(size=size, **self.jump_parameters)
                case JumpType.LOGNORMAL:
                    return rng_jump.lognormal(size=size, **self.jump_parameters)
                case JumpType.EXPONENTIAL:
                    return rng_jump.exponential(size=size, **self.jump_parameters)
                case _:
                    raise ValueError(
                        f"Unsupported jump distribution: {self.jump_distribution}"
                    )

        paths = np.zeros(shape=(n_paths, n_inc))
        for i in range(1, n_inc):
            y_i = [draw_y(size=n) for n in dN[:, i - 1]]
            x_i = [np.sum(y_vec) for y_vec in y_i]
            paths[:, i] = paths[:, i - 1] + x_i
        return SamplePaths(paths, t)
