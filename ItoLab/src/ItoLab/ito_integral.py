"""Itô integral approximator.

The Itô integral :math:`\\int_0^t X(s)\\, dW(s)` is approximated by a
Monte Carlo left-endpoint Riemann sum:

.. math::
    I(t) \\approx \\sum_{i=1}^{n} X_{i-1}\\, \\Delta W_i

where :math:`\\Delta W_i = W_{i} - W_{i-1}`.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from .StochasticProcess import StochasticProcess
from .Wiener import Wiener


def ito_integral(
    x: StochasticProcess,
    t: float = 1.0,
    n_paths: int = 1000,
    n_inc: int = 1000,
    seed0: Optional[int] = None,
    seed1: Optional[int] = None,
) -> npt.NDArray[np.float64]:
    """Approximate the Itô integral of ``x`` against a Wiener process.

    The integral :math:`I(t) = \\int_0^t X(s)\\, dW(s)` is approximated
    using a left-endpoint Monte Carlo scheme.  Independent Wiener
    increments are generated for the integrator ``W``, and independent
    paths of ``x`` are generated for the integrand.

    Parameters
    ----------
    x : StochasticProcess
        The integrand, any process implementing
        :class:`~ItoLab.StochasticProcess`.
    t : float, default 1.0
        Upper limit of integration.
    n_paths : int, default 1000
        Number of Monte Carlo sample paths.
    n_inc : int, default 1000
        Number of time increments used in the Riemann sum approximation.
    seed0 : int, optional
        Random seed for simulating the integrand ``x``.
    seed1 : int, optional
        Random seed for simulating the Wiener process ``W``.

    Returns
    -------
    npt.NDArray
        1-D array of length ``n_paths`` — simulated realisations of
        :math:`I(t)`.

    Examples
    --------
    For a constant integrand :math:`X_t \\equiv 1`, the Itô integral
    reduces to :math:`W_t`, which is normally distributed with mean 0
    and variance ``t``:

    >>> from ItoLab import DeterministicFn, ito_integral
    >>> import numpy as np
    >>> det = DeterministicFn(f=lambda t: 1.0)
    >>> result = ito_integral(det, t=1.0, n_paths=5000, n_inc=500, seed0=42917, seed1=42918)
    >>> float(np.round(result.mean(), 3))  # mean ~ 0
    0.006
    >>> float(np.round(result.var(), 3))    # variance ~ 1 ( = t )
    0.992
    """
    w_paths = (
        Wiener().generate_paths(n_paths=n_paths, n_inc=n_inc, seed=seed1).slice(0, t)
    )
    x_paths = x.generate_paths(n_paths=n_paths, n_inc=n_inc, seed=seed0)
    X_left = x_paths.slice(0, t).paths[:, :-1]
    dW = np.diff(w_paths.paths, axis=1)
    return np.sum(X_left * dW, axis=1)
