"""Deterministic function wrapper for the stochastic process interface.

A deterministic function :math:`f(t)` can be treated as a degenerate
stochastic process in which every path is identical.  This is useful
for mathematical results that apply to stochastic processes as special
cases, and for using deterministic functions as integrands in
:func:`~ItoLab.ito_integral`.
"""

from typing import Callable, Optional

import numpy as np

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess


class DeterministicFn(StochasticProcess):
    """A non-stochastic function of time ``f(t)``.

    Parameters
    ----------
    f : Callable[[float], float], default ``lambda t: 1.0``
        A function ``f(t)`` evaluated at each time point.

    Examples
    --------
    >>> from ItoLab import DeterministicFn
    >>> import numpy as np
    >>> det = DeterministicFn(f=lambda t: t**2)
    >>> paths = det.generate_paths(n_paths=2, n_inc=5, max_T=1.0, seed=42917)
    >>> np.round(paths.paths[0], 4)
    array([0.    , 0.0625, 0.25  , 0.5625, 1.    ])
    """

    def __init__(self, f: Callable[[float], float] = lambda t: 1.0):
        self.f = f

    def generate_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> SamplePaths:
        """Generate identical paths of the deterministic function.

        Since the process is non-random the ``seed`` parameter has no
        effect on the output.

        Parameters
        ----------
        n_paths : int, default 100
            Number of identical path realizations.
        n_inc : int, default 1000
            Number of time steps.
        max_T : float, default 1.0
            Terminal time.
        seed : int, optional
            Ignored (no randomness involved).

        Returns
        -------
        SamplePaths
            ``n_paths`` identical copies of ``f(t)`` evaluated on a
            uniform grid over ``[0, max_T]``.

        Examples
        --------
        >>> from ItoLab import DeterministicFn
        >>> import numpy as np
        >>> det = DeterministicFn(f=lambda t: t)
        >>> sp = det.generate_paths(n_paths=2, n_inc=5, max_T=1.0)
        >>> sp.paths[0]
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        """
        t = np.linspace(0.0, max_T, num=n_inc)
        f_t = np.array([self.f(time) for time in t])
        paths = np.tile(f_t, reps=(n_paths, 1))
        return SamplePaths(paths, t)
