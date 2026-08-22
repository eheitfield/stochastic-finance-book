"""Container for arrays of simulated stochastic process paths.

The :class:`SamplePaths` class is the standard return type for every
``generate_paths`` method in the package.  It bundles together the path
array, the time vector, and a few convenience methods for slicing,
sampling, and converting to a pandas DataFrame.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Optional, Self


class SamplePaths:
    """Wrapper class for an array of simulated random paths.

    Parameters
    ----------
    paths : npt.NDArray
        ``n_paths x n_points`` array of simulated path values.
        Column 0 is the initial value at ``t = 0``; column
        ``n_points - 1`` is the terminal value at ``t = max_T``.
    times : npt.NDArray
        ``n_points``-length vector of time points corresponding to the
        columns of *paths*.

    Attributes
    ----------
    paths : npt.NDArray
        The raw path array (read / write).
    times : npt.NDArray
        The time vector.
    n_paths : int
        Number of simulated paths.
    n_inc : int
        Number of time *increments* (``n_points - 1``).
    max_T : float
        Terminal time of the paths.

    Examples
    --------
    >>> from ItoLab import Wiener
    >>> sp = Wiener().generate_paths(n_paths=5, n_inc=50, max_T=1.0, seed=42917)
    >>> sp.n_paths
    5
    >>> sp.n_inc
    49
    >>> sp.max_T
    1.0
    >>> sp.paths.shape
    (5, 50)
    """

    def __init__(self, paths: npt.NDArray[np.float64], times: npt.NDArray[np.float64]):
        self.paths = paths
        self.times = times
        self.n_paths = paths.shape[0]
        self.n_inc = paths.shape[1] - 1
        self.max_T = float(np.max(times))

    def at(self, t: Optional[float] = None) -> npt.NDArray[np.float64]:
        """Return all path values at (or just before) time *t*.

        Uses :func:`numpy.searchsorted` which returns the index of the
        first element in ``self.times`` that is ``>= t``.  When ``t`` is
        before the first time point the index is 0; when ``t`` exceeds the
        terminal time the terminal column is returned.

        Parameters
        ----------
        t : float, optional
            Time at which to evaluate the paths.  Defaults to ``max_T``.

        Returns
        -------
        npt.NDArray
            1-D array of length ``n_paths`` — the value of each path
            at the nearest time index ``<= t``.

        Examples
        --------
        >>> from ItoLab import Wiener
        >>> sp = Wiener().generate_paths(n_paths=3, n_inc=100, max_T=1.0, seed=42917)
        >>> vals = sp.at(0.5)
        >>> vals.shape
        (3,)
        >>> vals  # doctest: +SKIP
        array([-0.08993216, -0.49015282,  0.47592897])
        """
        if t is None:
            t = self.max_T
        j = np.searchsorted(self.times, t)
        return self.paths[:, j]

    def slice(self, t0: float = 0.0, t1: Optional[float] = None) -> Self:
        """Return a sub-window of paths over the time interval ``[t0, t1]``.

        Parameters
        ----------
        t0 : float, default 0.0
            Start of the time window.
        t1 : float, optional
            End of the time window.  Defaults to ``max_T``.

        Returns
        -------
        SamplePaths
            A new :class:`SamplePaths` instance containing only the
            columns whose time falls in ``[t0, t1]``.

        Examples
        --------
        >>> from ItoLab import Wiener
        >>> sp = Wiener().generate_paths(n_paths=5, n_inc=100, max_T=1.0, seed=42917)
        >>> sub = sp.slice(0.0, 0.5)
        >>> sub.n_inc
        49
        >>> sub.max_T < sp.max_T
        True
        """
        if t1 is None:
            t1 = self.max_T
        j0 = int(np.clip(round(self.n_inc * t0 / self.max_T), 0, self.n_inc))
        j1 = int(np.clip(round(self.n_inc * t1 / self.max_T), 0, self.n_inc))
        return SamplePaths(self.paths[:, j0:j1], self.times[j0:j1])

    def downsample(self, n_samples: int) -> Self:
        """Reduce the number of time points to *n_samples* via uniform spacing.

        The first and last time points are always retained so that the
        path endpoints are preserved.

        Parameters
        ----------
        n_samples : int
            Number of time points to keep (``>= 2``).

        Returns
        -------
        SamplePaths
            A new :class:`SamplePaths` with ``n_samples`` columns.

        Examples
        --------
        >>> from ItoLab import Wiener
        >>> sp = Wiener().generate_paths(n_paths=5, n_inc=1000, max_T=1.0, seed=42917)
        >>> sub = sp.downsample(20)
        >>> sub.paths.shape[1]
        20
        >>> sub.paths[:, 0]  # first column preserved
        array([0., 0., 0., 0., 0.])
        """
        idxs = np.floor(np.linspace(0, self.n_inc, num=n_samples)).astype(int)
        idxs = np.clip(idxs, 0, self.n_inc)
        return SamplePaths(self.paths[:, idxs], self.times[idxs])

    def as_df(self) -> pd.DataFrame:
        """Convert paths to a :class:`pandas.DataFrame`.

        Each column is a single path (``Path_0``, ``Path_1``, ...);
        the index is the time vector.

        Returns
        -------
        pandas.DataFrame
            DataFrame of shape ``(n_points, n_paths)``.

        Examples
        --------
        >>> from ItoLab import Wiener
        >>> sp = Wiener().generate_paths(n_paths=3, n_inc=10, max_T=1.0, seed=42917)
        >>> df = sp.as_df()
        >>> df.shape
        (10, 3)
        >>> list(df.columns)
        ['Path_0', 'Path_1', 'Path_2']
        """
        return pd.DataFrame(
            self.paths.T,
            index=self.times,
            columns=[f"Path_{i}" for i in range(self.n_paths)],
        )
