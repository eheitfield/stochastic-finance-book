"""Abstract base class for all stochastic processes in ItoLab.

Every concrete process model (Wiener, GBM, CIR, etc.) inherits from
:class:`StochasticProcess` and implements :meth:`generate_paths`.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .SamplePaths import SamplePaths


class MultivariateStochasticProcess(ABC):
    """Abstract base class for a simulatable multivariate process.

    Subclasses must implement :meth:`generate_multi_paths`, which produces a
    :class:`~SamplePaths` object containing one or more sample paths
    of the process.

    The canonical signature is::

        generate_paths(self, n_paths, n_inc, max_T, seed) -> list[SamplePaths]

    Attributes
    ----------
    None — subclasses define their own parameters in ``__init__``.

    Examples
    --------
    Concrete subclasses can be instantiated directly:

    >>> from ItoLab import Wiener
    >>> sp = Wiener()
    >>> paths = sp.generate_paths(n_paths=10, n_inc=100, max_T=1.0, seed=42917)
    >>> paths.paths.shape
    (10, 100)
    """

    @abstractmethod
    def generate_multi_paths(
        self,
        n_paths: int = 100,
        n_inc: int = 1000,
        max_T: float = 1.0,
        seed: Optional[int] = None,
    ) -> list[SamplePaths]:
        """Generate simulated sample paths of the stochastic process.

        Parameters
        ----------
        n_paths : int, default 100
            Number of independent path realizations to simulate.
        n_inc : int, default 1000
            Number of time steps (columns in the path array).
        max_T : float, default 1.0
            Terminal time of the simulation.
        seed : int, optional
            Random seed for reproducibility.  If ``None``, results are
            not reproducible.

        Returns
        -------
        List of SamplePaths
            Containers holding the simulated paths and time vector for
            each variable.
        """
        pass
