# ItoLab

A lightweight stochastic-process simulation library accompanying the
*Stochastic Finance* book.

## Installation

Install in development mode from the repository root:

```bash
pip install -e ItoLab
```

Or install the package and its dependencies (numpy, scipy, pandas,
matplotlib) manually.

## Quick start

```python
from ItoLab import GeometricBrownianMotion, bs_option_value

# Simulate 1000 GBM paths
paths = GeometricBrownianMotion(s0=100, mu=0.05, sigma=0.2).generate_paths(
    n_paths=1000, n_inc=252, max_T=1.0, seed=42
)

# Price a European call with Black–Scholes
price = bs_option_value(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option="call")
```

## Available components

| Category | Components |
|---|---|
| **Processes** | `Wiener`, `PoissonProcess`, `CompensatedPoissonProcess`, `CompoundPoissonProcess`, `GeometricBrownianMotion`, `CoxIngersollRoss`, `OrnsteinUhlenbeck` (alias `OrnsteinUhlenbec`), `MertonJumpProcess`, `DeterministicFn` |
| **Integration** | `ito_integral` — Monte Carlo Itô integral approximator |
| **Pricing** | `bs_option_value`, `cir_bond_value`, `vasicek_bond_value` |
| **Utilities** | `SamplePaths` (path container), `JumpType` (enum) |

All process classes implement the `StochasticProcess` interface with a
`generate_paths(n_paths, n_inc, max_T, seed)` method that returns a
`SamplePaths` object.

## License

See the parent project for license information.
