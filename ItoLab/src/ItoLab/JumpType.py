"""Enumeration of jump-size distributions for compound processes.

Used by :class:`~ItoLab.CompoundPoissonProcess` to select the random
distribution from which individual jump sizes are drawn.
"""

from enum import Enum, auto


class JumpType(Enum):
    """Jump-size distribution selector.

    Members
    -------
    NORMAL
        Gaussian jumps: ``rng.normal(loc, scale, size)``.
    LOGNORMAL
        Log-normal jumps: ``rng.lognormal(mean, sigma, size)``.
    EXPONENTIAL
        Exponential jumps: ``rng.exponential(scale, size)``.
    GAMMA
        Gamma-distributed jumps: ``rng.gamma(a, scale, size)``.

    Examples
    --------
    >>> from ItoLab import JumpType
    >>> JumpType.NORMAL
    <JumpType.NORMAL: 1>
    >>> JumpType.EXPONENTIAL
    <JumpType.EXPONENTIAL: 3>
    """

    NORMAL = auto()
    LOGNORMAL = auto()
    EXPONENTIAL = auto()
    GAMMA = auto()
