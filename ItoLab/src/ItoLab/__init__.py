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

from .SamplePaths import SamplePaths
from .StochasticProcess import StochasticProcess
from .Wiener import Wiener
from .PoissonProcess import PoissonProcess
from .DeterministicFn import DeterministicFn
from .GeometricBrownianMotion import GeometricBrownianMotion
from .CoxIngersollRoss import CoxIngersollRoss
from .OrnsteinUhlenbeck import OrnsteinUhlenbeck
from .CompensatedPoissonProcess import CompensatedPoissonProcess
from .CompoundPoissonProcess import CompoundPoissonProcess
from .JumpType import JumpType
from .MertonJumpProcess import MertonJumpProcess
from .vasicek_bond_value import vasicek_bond_value
from .cir_bond_value import cir_bond_value
from .bs_option_value import bs_option_value
from .ito_integral import ito_integral
from .merton_option_value import merton_option_value

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
    "vasicek_bond_value",
    "cir_bond_value",
    "bs_option_value",
    "merton_option_value",
    "ito_integral",
}
