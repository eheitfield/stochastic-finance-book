"""ItoLab — a lightweight stochastic-process simulation library.

ItoLab provides simulators for common stochastic processes used in
quantitative finance (Wiener, GBM, Poisson, CIR, OU, Merton
jump-diffusion, etc.) and closed-form pricing tools (Black–Scholes
options, CIR and Vasicek zero-coupon bonds).  Every simulator returns
a :class:`~ItoLab.SamplePaths` object for consistent downstream
handling.

Typical usage::

    from ItoLab import GeometricBrownianMotion

    paths = GeometricBrownianMotion(s0=100, mu=0.05, sigma=0.2).generate_paths(
        n_paths=1000, n_inc=252, max_T=1.0, seed=42
    )
"""

from .bs_option_value import bs_option_value
from .cir_bond_value import cir_bond_value
from .CompensatedPoissonProcess import CompensatedPoissonProcess
from .CompoundPoissonProcess import CompoundPoissonProcess
from .CorrelatedWienerProcess import CorrelatedWienerProcess
from .CoxIngersollRoss import CoxIngersollRoss
from .DeterministicFn import DeterministicFn
from .GeometricBrownianMotion import GeometricBrownianMotion
from .ito_integral import ito_integral
from .JumpType import JumpType
from .merton_option_value import merton_option_value
from .MertonJumpProcess import MertonJumpProcess
from .MultivariateStochasticProcess import MultivariateStochasticProcess
from .OrnsteinUhlenbeck import OrnsteinUhlenbeck
from .PoissonProcess import PoissonProcess
from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess
from .vasicek_bond_value import vasicek_bond_value
from .Wiener import Wiener

__all__ = {
    "SamplePaths",
    "StochasticProcess",
    "Wiener",
    "PoissonProcess",
    "DeterministicFn",
    "GeometricBrownianMotion",
    "CoxIngersollRoss",
    "OrnsteinUhlenbeck",
    "CompensatedPoissonProcess",
    "CompoundPoissonProcess",
    "JumpType",
    "MertonJumpProcess",
    "MultivariateStochasticProcess",
    "CorrelatedWienerProcess",
    "vasicek_bond_value",
    "cir_bond_value",
    "bs_option_value",
    "merton_option_value",
    "ito_integral",
}
