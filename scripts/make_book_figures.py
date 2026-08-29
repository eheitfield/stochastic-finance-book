"""
Generate figures for the Stochastic Finance book.

This script generates one or more figures used throughout the book.

Usage
-----
From the project root:

    python scripts/<script_name>.py

All figures are written to

    figures/

Existing files with the same names are overwritten.
"""

from datetime import datetime
from numpy import character
from pathlib import Path
import datetime
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Optional

import ItoLab as ito

MASTER_SEED = 12345


# =============================================================================
# Project directories
# =============================================================================

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Output directory for all book figures
FIGURE_DIR = ROOT / "figures"

# Directory for working data
DATA_DIR = ROOT / "scripts" / "data"


# =============================================================================
# Figure generation functions
# =============================================================================


# =============================================================================
# Path-simulation figures
# =============================================================================


def figure_poisson_paths(seed: Optional[int] = None) -> None:
    lam = 5.0
    pois = ito.PoissonProcess(lam=lam)
    fig, ax = paths_plot(process=pois, y_label=rf"$N_t$", max_T=10.0, seed=seed)
    fig.savefig(FIGURE_DIR / "poisson_paths.svg")
    plt.close(fig)


def figure_comp_pois_paths(seed: Optional[int] = None) -> None:
    lam = 5.0
    cp = ito.CompensatedPoissonProcess(lam=lam)
    fig, ax = paths_plot(process=cp, y_label=rf"$M_t$", max_T=10.0, seed=seed)
    fig.savefig(FIGURE_DIR / "comp_pois_paths.svg")
    plt.close(fig)


def figure_wiener_paths(seed: Optional[int] = None) -> None:
    w = ito.Wiener()
    fig, ax = paths_plot(process=w, y_label=r"$X_t$", max_T=10, seed=seed)
    fig.savefig(FIGURE_DIR / "wiener_paths.svg")
    plt.close(fig)


def figure_gbm_paths(seed: Optional[int] = None) -> None:
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    gbm = ito.GeometricBrownianMotion(s0=s0, mu=mu, sigma=sigma)
    fig, ax = paths_plot(process=gbm, y_label=rf"$S_t$", max_T=10.0, seed=seed)
    fig.savefig(FIGURE_DIR / "gbm_paths.svg")
    plt.close(fig)


def figure_ou_paths(seed: Optional[int] = None) -> None:
    x0 = 0.06
    theta = 0.03
    kappa = 0.6
    sigma = 0.015
    ou = ito.OrnsteinUhlenbeck(x0=x0, theta=theta, kappa=kappa, sigma=sigma)
    fig, ax = paths_plot(process=ou, y_label=rf"$M_t$", max_T=10.0, seed=seed)
    ax.axhline(y=theta, color="salmon", linestyle="--", label=r"$\theta$")
    fig.savefig(FIGURE_DIR / "ou_paths.svg")
    plt.close(fig)


def figure_cir_paths(seed: Optional[int] = None) -> None:
    x0 = 0.06
    theta = 0.03
    kappa = 0.6
    sigma = 0.015 / np.sqrt(theta)
    cir = ito.CoxIngersollRoss(r0=x0, theta=theta, kappa=kappa, sigma=sigma)
    fig, ax = paths_plot(cir, y_label=rf"$X_t$", max_T=10.0, seed=seed)
    ax.axhline(y=theta, color="salmon", linestyle="--", label=r"$\theta$")
    fig.savefig(FIGURE_DIR / "cir_paths.svg")
    plt.close(fig)


def figure_cpp_paths(seed: Optional[int] = None) -> None:
    lam = 5.0
    mu = 0.0
    sigma = 1.0
    cpp = ito.CompoundPoissonProcess(
        lam=lam, jump_distribution=ito.JumpType.NORMAL, loc=mu, scale=sigma
    )
    fig, ax = paths_plot(cpp, y_label=rf"$J_t$", max_T=10.0, seed=seed)
    fig.savefig(FIGURE_DIR / "cpp_paths.svg")
    plt.close(fig)


def figure_gbm_paths_nasdaq(seed: Optional[int] = None) -> None:
    with open(DATA_DIR / "nasdaq_params.json") as f:
        params = json.load(f)
    mu = params["mu_observed"]
    sigma = params["sigma"]
    s0 = 1000
    gbm = ito.GeometricBrownianMotion(s0=s0, mu=mu, sigma=sigma)
    fig, ax = paths_plot(
        process=gbm, y_label=rf"$S_t$", max_T=25.0, n_paths=10, seed=seed
    )
    t = np.linspace(0, 25, num=100)
    exp_fit = s0 * np.exp(mu * t)
    ax.plot(
        t,
        exp_fit,
        label="Exponential Growth",
        color="salmon",
        linewidth=1.5,
        linestyle="--",
    )
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f}")
    ax.set_ylim(0, 20000)
    ax.legend(loc="upper left")
    fig.savefig(FIGURE_DIR / "gbm_paths_nasdaq.svg")
    plt.close(fig)


def figure_merton_paths(seed: Optional[int] = None) -> None:
    with open(DATA_DIR / "nasdaq_params.json") as f:
        params = json.load(f)
    mu = params["mu_sde"]
    sigma = params["sigma"]
    eta = params["lambda"]
    alpha = params["alpha"]
    delta = params["delta"]
    mu_observed = params["mu_observed"]
    s0 = 1000
    jd = ito.MertonJumpProcess(
        s0=s0, mu=mu, sigma=sigma, eta=eta, alpha=alpha, delta=delta
    )
    fig, ax = paths_plot(
        process=jd, y_label=rf"$S_t$", max_T=25.0, n_paths=10, seed=seed
    )
    t = np.linspace(0, 25, num=100)
    exp_fit = s0 * np.exp(mu_observed * t)
    ax.plot(
        t,
        exp_fit,
        label="Exponential Growth",
        color="salmon",
        linewidth=1.5,
        linestyle="--",
    )
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f}")
    ax.set_ylim(0, 20000)
    ax.legend(loc="upper left")
    fig.savefig(FIGURE_DIR / "mertaon_paths.svg")
    plt.close(fig)


# =============================================================================
# Process distribution figures
# =============================================================================


def figure_poisson_dist(seed: Optional[int] = None) -> None:
    lam = 5.0
    pois = ito.PoissonProcess(lam=lam)
    fig, ax = ridge_plot(pois, seed=seed)
    fig.savefig(FIGURE_DIR / "poisson_dist.svg")
    plt.close(fig)


def figure_wiener_dist(seed: Optional[int] = None) -> None:
    w = ito.Wiener()
    fig, ax = ridge_plot(process=w, max_T=10.0, seed=seed)
    fig.savefig(FIGURE_DIR / "wiener_dist.svg")
    plt.close(fig)


def figure_gbm_dist(seed: Optional[int] = None) -> None:
    s0 = 100.0
    mu = 0.08
    sigma = 0.2
    gbm = ito.GeometricBrownianMotion(s0=s0, mu=mu, sigma=sigma)
    fig, ax = ridge_plot(process=gbm, seed=seed)
    ax[0].set_xlim(0, 1000)
    fig.savefig(FIGURE_DIR / "gbm_dist.svg")
    plt.close(fig)


def figure_ou_dist(seed: Optional[int] = None) -> None:
    x0 = 0.06
    theta = 0.03
    kappa = 0.6
    sigma = 0.015
    ou = ito.OrnsteinUhlenbeck(x0=x0, theta=theta, kappa=kappa, sigma=sigma)
    fig, axs = ridge_plot(ou, max_T=10, seed=seed)
    for ax in axs:
        ax.axvline(x=theta, color="salmon", linestyle="--", label=r"$\theta$")
        ax.axvline(x=x0, color="cadetblue", linestyle="--", label=r"$X_0$")
    axs[0].legend()
    fig.savefig(FIGURE_DIR / "ou_dist.svg")
    plt.close(fig)


def figure_cir_dist(seed: Optional[int] = None) -> None:
    x0 = 0.06
    theta = 0.03
    kappa = 0.6
    sigma = 0.015 / np.sqrt(theta)
    cir = ito.CoxIngersollRoss(r0=x0, theta=theta, kappa=kappa, sigma=sigma)
    fig, axs = ridge_plot(cir, max_T=10, seed=seed)
    for ax in axs:
        ax.axvline(x=theta, color="salmon", linestyle="--", label=r"$\theta$")
        ax.axvline(x=x0, color="cadetblue", linestyle="--", label=r"$X_0$")
    axs[0].legend()
    axs[0].set_xlim(0, 0.1)
    fig.savefig(FIGURE_DIR / "cir_dist.svg")
    plt.close(fig)


def figure_gbm_dist_nasdaq(seed: Optional[int] = None) -> None:
    with open(DATA_DIR / "nasdaq_params.json") as f:
        params = json.load(f)
    mu = params["mu_observed"]
    sigma = params["sigma"]
    s0 = 1000
    gbm = ito.GeometricBrownianMotion(s0=s0, mu=mu, sigma=sigma)
    fig, axs = ridge_plot(gbm, max_T=20.0, process_lbl=r"$S_t$", seed=seed)
    axs[0].set_xlim(0, 30000)
    axs[-1].xaxis.set_major_formatter(lambda x, pos: f"{x:,.0f}")
    fig.savefig(FIGURE_DIR / "gbm_dist_nasdaq.svg")
    plt.close(fig)


# =============================================================================
# Data figures
# =============================================================================


def figure_nasdaq():
    nasdaq = pd.read_csv(DATA_DIR / "nasdaq.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(nasdaq["date"], nasdaq["index"])
    with open(DATA_DIR / "nasdaq_params.json") as f:
        params = json.load(f)
    mu = params["mu_observed"]
    sigma = params["sigma"]

    s0 = nasdaq["index"].iloc[range(250)].mean()
    d0 = nasdaq["date"].iloc[0]
    print(s0)
    t = (nasdaq["date"] - nasdaq["date"].iloc[0]) / pd.Timedelta(days=365.25)
    exp_fit = [s0 * np.exp(mu * tt) for tt in t]

    ax.plot(
        nasdaq["date"],
        exp_fit,
        label="Exponential Growth",
        color="salmon",
        linewidth=1.5,
        linestyle="--",
    )
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:,.0f}")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "nasdaq.svg")
    plt.close(fig)


def figure_effr() -> None:
    ust = pd.read_csv(DATA_DIR / "effr.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ust["date"], ust["effr"])
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    fig.savefig(FIGURE_DIR / "effr.svg")
    plt.close(fig)


def figure_ust10() -> None:
    ust = pd.read_csv(DATA_DIR / "ust_10_yr.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ust["date"], ust["ust_10_yr"])
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    fig.savefig(FIGURE_DIR / "ust_10_yr.svg")
    plt.close(fig)


def figure_yield_curve_ex() -> None:
    zero_ylds = pd.read_csv(DATA_DIR / "zero_yields.csv", parse_dates=["date"])
    zero_ylds.set_index("date", inplace=True)
    ttm = range(1, 11)
    date_lst = ["May 13, 2008", "March 3, 2023", "July 23, 2025"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for dt in date_lst:
        ax.plot(ttm, zero_ylds.loc[dt], label=dt)
    ax.legend(loc="lower right")
    ax.set_xlabel("Time to Maturity (Years)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    fig.savefig(FIGURE_DIR / "yield_curve_ex.svg")
    plt.close(fig)


# =============================================================================
# Daily return figures
# =============================================================================


def figure_nas_daily_ret() -> None:
    nasdaq = pd.read_csv(DATA_DIR / "nasdaq.csv", parse_dates=["date"])
    fig, ax = returns_plot(nasdaq["date"], nasdaq["daily_return"])
    fig.savefig(FIGURE_DIR / "nas_daily_ret.svg")
    plt.close(fig)


def figure_gbm_daily_return(seed: Optional[int] = None) -> None:
    with open(DATA_DIR / "nasdaq_params.json") as f:
        params = json.load(f)
    mu = params["mu_observed"]
    sigma = params["sigma"]
    s0 = 1000.0
    gbm = ito.GeometricBrownianMotion(s0=s0, mu=mu, sigma=sigma)
    path = gbm.generate_paths(n_paths=1, n_inc=252 * 35, max_T=35, seed=seed)
    ret = np.diff(np.log(path.paths), axis=1)
    dates = path.times[1:]
    fig, ax = returns_plot(dates, ret.T)
    fig.savefig(FIGURE_DIR / "gbm_daily_ret.svg")
    plt.close(fig)


def figure_mjd_daily_return(seed: Optional[int] = None) -> None:
    with open(DATA_DIR / "nasdaq_params.json") as f:
        params = json.load(f)
    mu = params["mu_sde"]
    sigma = params["sigma"]
    eta = params["lambda"]
    alpha = params["alpha"]
    delta = params["delta"]
    s0 = 1000.0
    mjd = ito.MertonJumpProcess(
        s0=s0, mu=mu, sigma=sigma, eta=eta, alpha=alpha, delta=delta
    )
    path = mjd.generate_paths(n_paths=1, n_inc=252 * 35, max_T=35, seed=seed)
    ret = np.diff(np.log(path.paths), axis=1)
    dates = path.times[1:]
    fig, ax = returns_plot(dates, ret.T)
    fig.savefig(FIGURE_DIR / "mjd_daily_ret.svg")
    plt.close(fig)


# =============================================================================
# Other figure functions
# =============================================================================


def figure_matingale_ex(seed: Optional[int] = None) -> None:
    if not seed:
        seed = 1234
    lam = 5.0
    cp = ito.CompensatedPoissonProcess(lam=lam)
    main_path = cp.generate_paths(n_paths=1, max_T=10.0, seed=seed)
    start_times = [2.5, 5.0, 7.5]
    fig, axs = plt.subplots(nrows=3, figsize=(6, 8), sharey=True)
    for i, st in enumerate(start_times):
        mps = main_path.slice(0.0, st)
        x_offset = mps.paths[:, -1]
        t_offset = mps.times[-1]
        alt_paths = cp.generate_paths(n_paths=10, max_T=10.0 - t_offset, seed=seed + i)
        axs[i].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
        axs[i].axhline(y=x_offset, color="salmon", linestyle="--", linewidth=0.75)
        axs[i].plot(mps.times, mps.paths.T, color="steelblue")
        axs[i].plot(
            alt_paths.times + t_offset,
            alt_paths.paths.T + x_offset,
            color="steelblue",
            alpha=0.75,
        )
        axs[i].set_title(f"t = {st:.1f}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "martingale_ex.svg")
    plt.close(fig)


def figure_wiener_limit(seed: Optional[int] = None) -> None:
    wiener = ito.Wiener()
    path1000 = wiener.generate_paths(n_paths=1, n_inc=1001, max_T=10.0, seed=seed)
    path100 = path1000.downsample(101)
    path10 = path1000.downsample(11)
    fig, axs = plt.subplots(nrows=3, figsize=(6, 8))
    axs[0].set_title(rf"$N = 10$")
    axs[0].step(path10.times, path10.paths.T, where="post")
    axs[0].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
    axs[1].set_title(rf"$N = 100$")
    axs[1].step(path100.times, path100.paths.T, where="post")
    axs[1].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
    axs[2].set_title(rf"$N = 1000$")
    axs[2].step(path1000.times, path1000.paths.T, where="post")
    axs[2].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "wiener_limit.svg")
    plt.close(fig)


def figure_weiner_frac(seed: Optional[int] = None) -> None:
    wiener = ito.Wiener()
    fn = ito.DeterministicFn(lambda x: np.sin(x * np.pi / 2))
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 8))
    axs[0, 0].set_title(rf"$\sin(t\,\pi\,/\,10)$")
    f_paths = fn.generate_paths(n_paths=1, n_inc=100_000, max_T=10.0)
    axs[0, 0].plot(f_paths.times, f_paths.paths.T)
    f_paths = f_paths.slice(0.0, 1.0)
    axs[1, 0].plot(f_paths.times, f_paths.paths.T)
    f_paths = f_paths.slice(0.0, 0.1)
    axs[2, 0].plot(f_paths.times, f_paths.paths.T)
    axs[0, 1].set_title(r"$W_t$")
    w_paths = wiener.generate_paths(n_paths=1, n_inc=100_000, max_T=10.0, seed=seed)
    axs[0, 1].plot(w_paths.times, w_paths.paths.T)
    axs[0, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
    w_paths = w_paths.slice(0.0, 1.0)
    axs[1, 1].plot(w_paths.times, w_paths.paths.T)
    axs[1, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
    w_paths = w_paths.slice(0.0, 0.1)
    axs[2, 1].plot(w_paths.times, w_paths.paths.T)
    axs[2, 1].axhline(y=0, color="black", linestyle="-", linewidth=0.75)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "wiener_frac.svg")
    plt.close(fig)


def figure_bs_put_payoff() -> None:
    s0 = np.linspace(75.0, 125.0, num=100)
    r = 0.05
    sigma = 0.2
    K = 100.0
    T = np.linspace(0.0, 1.0, num=5)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel(r"Option Value ($V^{\text{Put}}_0$)")
    ax.set_xlabel(r"Stock Price ($S_0$)")
    colors = sns.light_palette("steelblue", n_colors=T.shape[0], reverse=True)
    for i, t in enumerate(T):
        payoffs = ito.bs_option_value(S=s0, K=K, T=t, r=r, sigma=sigma, option="put")
        ax.plot(s0, payoffs, label=rf"$T={t:.2f}$", color=colors[i])
    ax.legend()
    ax.set_box_aspect(0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bs_put_payoff.svg")
    plt.close(fig)


def figure_bs_put_call() -> None:
    s0 = np.linspace(75.0, 125.0, num=100)
    r = 0.05
    sigma = 0.2
    K = 100.0
    T = 1.0
    payoffs_put = ito.bs_option_value(S=s0, K=K, T=T, r=r, sigma=sigma, option="put")
    payoffs_call = ito.bs_option_value(S=s0, K=K, T=T, r=r, sigma=sigma, option="call")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel(r"Option Value ($V_0$)")
    ax.set_xlabel(r"Stock Price ($S_0$)")
    ax.plot(s0[payoffs_put < 25], payoffs_put[payoffs_put < 25], label=rf"Put Option")
    ax.plot(
        s0[payoffs_call < 25], payoffs_call[payoffs_call < 25], label=rf"Call Option"
    )
    ax.set_box_aspect(0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bs_put_call.svg")
    plt.close(fig)


def figure_d_and_c() -> None:
    r0 = 0.05
    t_max = 10
    c = 0.05
    f = 1.0
    dy = np.linspace(-0.02, 0.02, num=100)

    def bond_price(delta_y):
        r = r0 + delta_y
        return (c / r) * (1 - np.exp(-r * t_max)) + f * np.exp(-r * t_max)

    dur = -(1 / bond_price(0)) * (
        (-c / r0**2) * (1 - np.exp(-r0 * t_max))
        + (c / r0) * (t_max) * np.exp(-r0 * t_max)
        + f * (-t_max) * np.exp(-r0 * t_max)
    )
    con = (1 / bond_price(0)) * (
        (2 * c / r0**3) * (1 - np.exp(-r0 * t_max))
        + (c / r0) * (t_max * (-t_max) * np.exp(-r0 * t_max))
        + f * (-t_max) * (-t_max) * np.exp(-r0 * t_max)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dy, bond_price(dy) / bond_price(0) - 1, label="Actual")
    ax.plot(
        dy,
        -dur * dy,
        label="1st Order Approximation",
    )
    ax.plot(
        dy,
        -dur * dy + 0.5 * con * (dy**2),
        label="2nd Order Approximation",
    )
    ax.axhline(0, color="k")
    ax.axvline(0, color="k")
    ax.set_xlabel(r"$\Delta y$")
    ax.set_ylabel("$\Delta B_t / B_t$")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.legend()
    fig.savefig(FIGURE_DIR / "d_and_c.svg")
    plt.close()


def figure_vasicek_term_structure() -> None:
    lam = 0.1
    kappa = 0.6
    sigma = 0.015
    theta_p = 0.03
    theta_q = theta_p - (sigma / kappa) * lam
    taus = np.linspace(0.01, 10.0, num=50)
    r_low = 0.01
    r_hi = 0.06
    yields_r_low = [
        ito.vasicek_bond_value(
            r_t=r_low, tau=tau, kappa=kappa, theta_q=theta_q, sigma=sigma, as_yield=True
        )
        for tau in taus
    ]
    yields_r_hi = [
        ito.vasicek_bond_value(
            r_t=r_hi, tau=tau, kappa=kappa, theta_q=theta_q, sigma=sigma, as_yield=True
        )
        for tau in taus
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel(r"Time to Maturity ($\tau$)")
    ax.plot(taus, yields_r_low, label=f"$r_0$ = {r_low:.1%}")
    ax.plot(taus, yields_r_hi, label=f"$r_0$ = {r_hi:.1%}")
    ax.axhline(y=theta_q, color="salmon", linestyle="--", label=r"$\theta_Q$")
    ax.legend(loc="upper right")
    ax.set_ylim(0.0, 0.07)
    ax.set_ylabel(r"Yield ($Y_{\tau}$)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "vasicek_term_structure.svg")
    plt.close(fig)


def figure_vasicek_short_rates() -> None:
    lam = 0.1
    kappa = 0.6
    sigma = 0.015
    theta_p = 0.03
    theta_q = theta_p - (sigma / kappa) * lam
    taus = np.linspace(0.01, 10.0, num=50)
    r0 = 0.01

    def expected_r(tau, r0, theta, kappa):
        return np.exp(-kappa * tau) * r0 + theta * (1 - np.exp(-kappa * tau))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, [expected_r(t, r0, theta_q, kappa) for t in taus], label="Q Measure")
    ax.plot(taus, [expected_r(t, r0, theta_p, kappa) for t in taus], label="P Measure")
    ax.set_ylabel(r"Expected Short Rate ($\mathrm{E}[r_t]$)")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylim(0, 0.04)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.legend(loc="lower right")
    fig.get_tight_layout()
    fig.savefig(FIGURE_DIR / "vasicek_short_rates.svg")
    plt.close(fig)


def figure_cir_term_structure() -> None:
    lam_vas = 0.1
    sigma_vas = 0.015
    kappa_p = 0.6
    theta_p = 0.03
    sigma = 0.015 / np.sqrt(theta_p)
    theta_q = theta_p - (sigma_vas / kappa_p) * lam_vas
    kappa_q = kappa_p * theta_p / theta_q
    sigma = sigma_vas / np.sqrt(theta_p)
    taus = np.linspace(0.01, 10.0, num=50)
    r_low = 0.01
    r_hi = 0.06
    yields_r_low = [
        ito.cir_bond_value(
            r_t=r_low,
            tau=tau,
            kappa_q=kappa_q,
            theta_q=theta_q,
            sigma=sigma,
            as_yield=True,
        )
        for tau in taus
    ]
    yields_r_hi = [
        ito.cir_bond_value(
            r_t=r_hi,
            tau=tau,
            kappa_q=kappa_q,
            theta_q=theta_q,
            sigma=sigma,
            as_yield=True,
        )
        for tau in taus
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel(r"Time to Maturity ($\tau$)")
    ax.plot(taus, yields_r_low, label=rf"$r_0 = {r_low}$")
    ax.plot(taus, yields_r_hi, label=rf"$r_0 = {r_hi}$")
    ax.axhline(y=theta_q, color="salmon", linestyle="--", label=r"$\theta_Q$")
    ax.legend(loc="upper right")
    ax.set_ylim(0.0, 0.07)
    ax.set_ylabel(r"Yield ($Y_{\tau}$)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cir_term_structure.svg")
    plt.close(fig)


def figure_cir_vol() -> None:
    lam_vas = 0.1
    sigma_vas = 0.015
    kappa_p = 0.6
    theta_p = 0.03
    sigma = 0.015 / np.sqrt(theta_p)
    theta_q = theta_p - (sigma_vas / kappa_p) * lam_vas
    kappa_q = kappa_p * theta_p / theta_q
    sigma = sigma_vas / np.sqrt(theta_p)
    taus = np.linspace(0.01, 10.0, num=50)
    rs = [0.01, theta_p, 0.06]

    def vol_vas(tau):
        return sigma_vas * (1 - np.exp(-kappa_p * tau)) / kappa_p

    def vol_cir(tau, r):
        gamma = np.sqrt(kappa_q**2 + 2 * sigma**2)
        B = (
            2
            * (np.exp(gamma * tau) - 1)
            / ((kappa_q + gamma) * (np.exp(gamma * tau) - 1) + 2 * gamma)
        )
        return sigma * np.sqrt(r) * B

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(taus, [vol_vas(t) for t in taus], linestyle="--", label="Vasicek")
    for r in rs:
        plt.plot(taus, [vol_cir(t, r) for t in taus], label=f"CIR: r={r:.1%}")
    ax.set_xlabel(r"Time to Maturity ($\tau$)")
    ax.set_ylabel(r"Bond Return Volatility ($\sigma_Q$)")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cir_vol.svg")
    plt.close(fig)


def figure_bond_vol() -> None:
    ust = pd.read_csv(DATA_DIR / "ust_10_yr.csv", parse_dates=["date"])
    ust["ln_price"] = ust["ust_10_yr"] * 7.5
    ust["daily_ret"] = ust["ln_price"].diff()
    ust["ann_ret"] = ust["daily_ret"].rolling(window=252, min_periods=1).mean() * 252
    ust["ann_vol"] = ust["daily_ret"].rolling(
        window=252, min_periods=1
    ).std() * np.sqrt(252)
    ust["ann_yld"] = ust["ust_10_yr"].rolling(window=252, min_periods=1).mean()
    ust_yearly = ust.groupby(ust["date"].dt.year)[["ann_yld", "ann_vol"]].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ust_yearly["ann_yld"], ust_yearly["ann_vol"], s=5)
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1%}")
    ax.set_xlabel("10-Year Treasiry Yield")
    ax.set_ylabel("10-Year Treasury Volatility")
    fig.savefig(FIGURE_DIR / "bond_vol.svg")
    plt.close(fig)


def figure_mjd_vs_bs_put_payoff() -> None:
    s0 = np.linspace(50.0, 150.0, num=100)
    r = 0.05
    sigma = 0.2
    K = 100.0
    alpha = -0.01
    delta = 0.07
    eta = 1.0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_ylabel(r"Option Value ($V^{\text{Put}}_0$)")
    ax.set_xlabel(r"Stock Price ($S_0$)")
    v_gbm = ito.bs_option_value(S=s0, K=K, T=0.5, r=r, sigma=sigma, option="put")
    ax.plot(s0, v_gbm, label=rf"GBM")
    v_jd = ito.merton_option_value(
        S=s0,
        K=K,
        T=1,
        r=r,
        sigma=sigma,
        lam=eta,
        alpha=alpha,
        delta=delta,
        option="put",
    )
    ax.plot(s0, v_jd, label=r"Jump-Diffusion")
    ax.axvline(K, linestyle="--", label="Strik Price")
    ax.legend()
    ax.set_box_aspect(0.5)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "mjd_vs_bs_put_payoff.svg")
    plt.close(fig)


def figure_biv_wiener(seed: Optional[int] = None) -> None:
    rho = 0.75
    n_paths = 20
    bw0 = CorrelatedWienerProcess(corr=0.0).generate_multi_paths(
        n_paths=n_paths, max_T=10.0, seed=seed
    )
    bw1 = CorrelatedWienerProcess(corr=rho).generate_multi_paths(
        n_paths=n_paths, max_T=10.0, seed=seed + 20
    )

    def plot_paths(bw, ax):
        for p in range(n_paths):
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.axvline(x=0, color="black", linewidth=0.5)
            ax.plot(bw[0].paths[p, :], bw[1].paths[p, :], color="steelblue", alpha=0.5)
            ax.set_ylim(-8, 8)
            ax.set_xlim(-8, 8)
            ax.set_aspect("equal")

    fig, axs = plt.subplots(nrows=2, figsize=(4, 8))

    plot_paths(bw0, axs[0])
    axs[0].set_title(r"$\rho=0.0$")
    plot_paths(bw1, axs[1])
    axs[1].set_title(rf"$\rho={{{rho:.2}}}$")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "biv_wiener.svg")
    plt.close(fig)


# =============================================================================
# Plot helper functions
# =============================================================================


def paths_plot(
    process: ito.StochasticProcess,
    y_label: str = rf"$X_t$",
    n_paths: int = 5,
    max_T: float = 10.0,
    seed: Optional[int] = None,
) -> tuple[plt.Figure, plt.Axes]:
    paths = process.generate_paths(n_paths=n_paths, max_T=max_T, seed=seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    if np.max(paths.paths) > 0 and np.min(paths.paths) < 0:
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.plot(paths.times, paths.paths.T, color="steelblue", alpha=0.75)
    ax.set_xlabel(rf"$t$")
    ax.set_ylabel(y_label)
    ax.plot(0.0, paths.paths[0, 0], "ko", ms=3)
    fig.tight_layout()
    return fig, ax


def three_panel_dens_plot(
    process: ito.StochasticProcess,
    times: NDArray = [1.0, 5.0, 10.0],
    n_inc: int = 500,
    n_paths: int = 10000,
    max_T: float = 10.0,
    seed: Optional[int] = None,
) -> tuple[plt.Figure, plt.Axes]:
    paths = process.generate_paths(n_paths=n_paths, n_inc=n_inc, max_T=max_T, seed=seed)
    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
    for i, t in enumerate(times):
        # ax[i].hist(paths.at(t), bins = 50, density=True)
        sns.kdeplot(paths.at(t), fill=True, ax=ax[i])
        ax[i].set_title(f"t = {t}")
        ax[i].set_ylabel(rf"Density")
    ax[0].legend()
    fig.tight_layout()
    return fig, ax


def returns_plot(dates: NDArray, returns: NDArray) -> tuple[plt.Figure, plt.Axes]:
    min_dt = np.min(dates)
    max_dt = np.max(dates)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(dates, returns, color="steelblue")
    ax.hlines(
        y=np.mean(returns), xmin=min_dt, xmax=max_dt, color="dimgray", label="Mean"
    )
    ax.hlines(
        y=[np.std(returns), -np.std(returns)],
        xmin=min_dt,
        xmax=max_dt,
        color="firebrick",
        label="1 S.D.",
    )
    ax.hlines(
        y=[5 * np.std(returns), -5 * np.std(returns)],
        xmin=min_dt,
        xmax=max_dt,
        color="lightsalmon",
        label="5 S.D.",
    )
    ax.legend(loc="lower left")
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.margins(x=0)
    ax.set_ylim(-0.18, 0.18)
    fig.tight_layout()
    return fig, ax


def ridge_plot(
    process: ito.StochasticProcess,
    times: Optional[NDArray] = None,
    n_inc: int = 500,
    n_paths: int = 10000,
    max_T: float = 10.0,
    process_lbl: character = r"$X_t$",
    seed: Optional[int] = None,
) -> tuple[plt.Figure, np.ndarray]:
    if not times:
        times = np.linspace(max_T / 10, max_T, num=10)
    nt = times.shape[0]
    paths = process.generate_paths(n_paths=n_paths, n_inc=n_inc, max_T=max_T, seed=seed)

    fig, axs = plt.subplots(nrows=nt, ncols=1, sharex=True, figsize=(6, 8))
    for i, t in enumerate(times):
        sns.kdeplot(paths.at(t), fill=True, linewidth=0, ax=axs[i])
        axs[i].text(
            1.0,
            0.02,
            f"t = {t:.1f}",
            transform=axs[i].transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=10,
        )
        axs[i].spines["left"].set_visible(False)
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].set_yticks([])
        axs[i].set_ylabel("")
        if i < nt - 1:
            axs[i].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        else:
            axs[i].set_xlabel(process_lbl)
    fig.tight_layout()
    return fig, axs


def figure_bs_option_path(seed: Optional[int] = None) -> None:
    s0 = 90
    r = 0.05
    mu = 0.08
    sigma = 0.20
    k = 100
    max_T = 1.0
    n_inc = 1000

    s_path = ito.GeometricBrownianMotion(s0=s0, mu=mu, sigma=sigma).generate_paths(
        n_paths=1, n_inc=n_inc, max_T=max_T, seed=seed
    )
    v_put = []
    v_call = []
    for i in range(n_inc):
        ttm = max_T - s_path.times[i]
        v_s = s_path.paths[0, i]
        v_put.append(
            ito.bs_option_value(S=v_s, K=k, T=ttm, r=r, sigma=sigma, option="put")
        )
        v_call.append(
            ito.bs_option_value(S=v_s, K=k, T=ttm, r=r, sigma=sigma, option="call")
        )

    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))
    axs[0].plot(s_path.times, s_path.paths.T)
    axs[0].set_title("Stock Price")
    axs[0].axhline(y=k, label="Strike Price", color="salmon", linestyle="--")
    axs[0].legend()
    axs[1].plot(s_path.times, v_put, label="Put")
    axs[1].plot(s_path.times, v_call, label="Call")
    axs[1].set_title("Option Value")
    axs[1].legend()
    axs[1].set_xlabel("t")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "bs_option_path.svg")
    plt.close(fig)


# =============================================================================
# Figure orchestration
# =============================================================================


def all_figures() -> None:
    # Chapter 1
    figure_poisson_paths(MASTER_SEED)
    figure_poisson_dist(MASTER_SEED + 10)
    figure_comp_pois_paths(MASTER_SEED + 110)
    figure_matingale_ex(MASTER_SEED + 130)
    # Chapter 2
    figure_wiener_limit(MASTER_SEED + 20)
    figure_weiner_frac(MASTER_SEED + 30)
    figure_wiener_paths(MASTER_SEED + 40)
    figure_wiener_dist(MASTER_SEED + 50)
    # Chapter 3
    figure_nasdaq()
    figure_gbm_paths_nasdaq(seed=MASTER_SEED + 15)
    figure_gbm_dist_nasdaq(seed=MASTER_SEED + 35)
    # Chapter 4
    figure_bs_put_payoff()
    figure_bs_put_call()
    # Chapter 5
    figure_effr()
    figure_yield_curve_ex()
    figure_d_and_c()
    # Chapter 6
    figure_ust10()
    figure_ou_paths(MASTER_SEED + 80)
    figure_ou_dist(MASTER_SEED + 90)
    figure_vasicek_term_structure()
    figure_vasicek_short_rates()
    figure_bond_vol()
    figure_cir_paths(MASTER_SEED + 100)
    figure_cir_dist(MASTER_SEED + 95)
    figure_cir_term_structure()
    figure_cir_vol()
    figure_biv_wiener(MASTER_SEED + 250)
    # Chapter 8
    figure_nas_daily_ret()
    figure_cpp_paths(seed=MASTER_SEED + 150)
    figure_gbm_daily_return(seed=MASTER_SEED + 210)
    figure_merton_paths(seed=MASTER_SEED + 200)
    figure_mjd_daily_return(seed=MASTER_SEED + 200)
    figure_mjd_vs_bs_put_payoff()


def test_figures() -> None:
    figure_bs_option_path(seed=MASTER_SEED + 250)


def main() -> None:

    # Configure master style parameters
    mpl.rcdefaults()
    # Define custom color palette: muted professional colors
    # with steelblue and salmon as the first two
    colors = [
        "steelblue",
        "salmon",
        "darkseagreen",
        "dimgray",
        "cadetblue",
        "lightsalmon",
    ]
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=colors)
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["lines.linewidth"] = 1.0
    mpl.rcParams["mathtext.fontset"] = "cm"
    mpl.rcParams["legend.frameon"] = False

    # Choose one
    test_figures()
    # all_figures()


if __name__ == "__main__":
    main()
