"""
plotting.py — Reusable figure functions for the RL pricing paper.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .evaluation import ClusterMetrics


# ---------------------------------------------------------------------------
# Tick formatter shared across plots
# ---------------------------------------------------------------------------

_BASELINE_PRICE = 0.25  # EUR/kWh flat-rate reference


def _price_tick_formatter(x, pos):
    """Label the flat-rate baseline as π^b; format everything else to 2 dp."""
    if np.isclose(x, _BASELINE_PRICE, rtol=0.02):
        return r"$\pi^b$"
    return f"{x:.2f}"


def _configure_time_axis(ax: plt.Axes, index: pd.DatetimeIndex) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%I %p"))
    ax.set_xlim(index.min(), index.max())


# ---------------------------------------------------------------------------
# Main result figure
# ---------------------------------------------------------------------------


def plot_pricing_comparison(
    tou_prices_eur: pd.Series,
    rl_prices_list: list[pd.Series],
    flat_metrics: ClusterMetrics,
    tou_metrics: ClusterMetrics,
    rl_metrics: ClusterMetrics,
    surcharge_threshold: float = 230.0,
    y_lim_load: tuple[float, float] = (80, 300),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel figure: (top) price trajectories, (bottom) aggregate load curves.

    Parameters
    ----------
    tou_prices_eur : pd.Series
        TOU price schedule in EUR/kWh.
    rl_prices_list : list of pd.Series
        One price trajectory per consumer (EUR/kWh) from the RL agent.
    flat_metrics, tou_metrics, rl_metrics : ClusterMetrics
        Results from the three pricing schemes.
    surcharge_threshold : float
        Threshold above which the shaded penalty region is drawn on the load panel.
    y_lim_load : tuple
        Y-axis limits for the aggregate load panel (kW).
    save_path : str, optional
        If provided, save the figure to this path at 400 dpi.

    Returns
    -------
    matplotlib Figure
    """
    index = tou_prices_eur.index
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5), layout="constrained")

    for ax in axes:
        _configure_time_axis(ax, index)

    # --- Top panel: prices ---
    ax0 = axes[0]
    ax0.grid(alpha=0.4)
    for p in rl_prices_list:
        ax0.step(p.index, p, alpha=0.07, color="C2", where="post")
    ax0.step(index, tou_prices_eur, color="C0", where="post", label="TOU")
    ax0.axhline(_BASELINE_PRICE, linestyle="--", color="k", linewidth=0.9)
    ax0.yaxis.set_major_formatter(mticker.FuncFormatter(_price_tick_formatter))
    ax0.set(ylabel=r"$\pi_t$ (EUR/kWh)")

    # --- Bottom panel: aggregate load ---
    ax1 = axes[1]
    ax1.grid(alpha=0.4)
    ax1.fill_between(
        index,
        surcharge_threshold,
        surcharge_threshold * 1.5,
        alpha=0.2,
        color="r",
    )
    ax1.text(0.05, 0.8, r"$G_t > G^{peak}$", transform=ax1.transAxes, fontsize=8)

    (l_base,) = ax1.plot(index, flat_metrics.cumul_pgrid, color="k", alpha=0.8, linestyle="--")
    (l_tou,)  = ax1.plot(index, tou_metrics.cumul_pgrid, color="C0")
    (l_rl,)   = ax1.plot(index, rl_metrics.cumul_pgrid,  color="C2")

    ax1.set(ylabel=r"$G_t$ (kW)", xlabel=r"Time ($t$)", ylim=y_lim_load)

    fig.autofmt_xdate()
    fig.legend(
        (l_base, l_tou, l_rl),
        ("Baseline", "TOU", "RL"),
        loc="outside lower center",
        ncol=3,
    )

    if save_path:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Auxiliary diagnostics
# ---------------------------------------------------------------------------


def plot_demand_samples(
    generated_profiles: list[pd.Series],
    label: str = "Sampled",
    color: str = "C9",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Overlay multiple stochastic demand trajectories to visualise KDE spread.

    Parameters
    ----------
    generated_profiles : list of pd.Series
        Demand trajectories (kW), one per sample.
    label : str
        Label prefix shown in the legend.
    color : str
        Matplotlib colour spec.
    ax : plt.Axes, optional
        Axes to draw on; creates a new figure if None.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    for profile in generated_profiles:
        ax.plot(profile.index, profile, alpha=0.15, color=color)
    ax.set(xlabel="Time", ylabel="Demand (kW)", title=f"{label} demand trajectories")
    return ax


def plot_price_distribution(
    price_series_list: list[pd.Series],
    color: str = "C3",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Step-plot all consumer price trajectories from the RL agent.

    Parameters
    ----------
    price_series_list : list of pd.Series
        Per-consumer price trajectories (EUR/kWh).
    color : str
    ax : plt.Axes, optional

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    for p in price_series_list:
        ax.step(p.index, p, alpha=0.1, color=color, where="post")
    mean_price = np.array([p.values for p in price_series_list]).mean()
    ax.axhline(mean_price, color="g", linestyle="--", alpha=0.6, label=f"Mean = {mean_price:.3f}")
    ax.set(xlabel="Time", ylabel="Price (EUR/kWh)", title="RL price distribution")
    ax.legend()
    return ax
