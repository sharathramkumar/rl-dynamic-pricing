"""
evaluation.py — Utility functions for benchmarking the RL pricing agent.

Provides:
  - ClusterMetrics: dataclass for collecting PAR, bill, and profit results
  - evaluate_flat_pricing:   baseline (flat-rate) benchmark
  - evaluate_price_sequence: arbitrary hourly price signal (e.g. TOU)
  - evaluate_rl_agent:       run a trained SB3 PPO agent across a cluster
  - build_tou_prices:        construct a typical time-of-use price schedule
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from .environment import ResponsiveBuildingEnv


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ClusterMetrics:
    """Aggregate statistics for one pricing scheme applied to a cluster."""

    scheme: str
    par: float                   # peak-to-average ratio of total cluster load
    bill_mean: float             # mean consumer bill (EUR)
    bill_std: float
    bill_min: float
    bill_max: float
    agg_profit: float            # total service-provider profit (EUR)
    cumul_pgrid: pd.Series = field(repr=False)  # hourly aggregate load curve

    def summary(self) -> str:
        lines = [
            f"=== {self.scheme} ===",
            f"  Peak-to-Average Ratio : {self.par:.3f}",
            f"  Consumer Bill (EUR)   : mean={self.bill_mean:.2f}  "
            f"std={self.bill_std:.2f}  "
            f"[{self.bill_min:.2f}, {self.bill_max:.2f}]",
            f"  Aggregate Profit (EUR): {self.agg_profit:.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def evaluate_flat_pricing(
    envs: list[ResponsiveBuildingEnv],
    w_prices: pd.Series,
    flat_price: float = 0.25,
    surcharge_threshold: float = 230.0,
    surcharge_factor: float = 0.02,
) -> ClusterMetrics:
    """
    Compute cluster-level metrics under a fixed flat retail price.

    Parameters
    ----------
    envs : list of ResponsiveBuildingEnv
        One environment per consumer (each holding its load profile).
    w_prices : pd.Series
        Hourly wholesale prices aligned with the load profiles (EUR/kWh).
    flat_price : float
        Retail price charged to all consumers (EUR/kWh).
    surcharge_threshold : float
        Peak power level above which the grid operator incurs a quadratic surcharge (kW).
    surcharge_factor : float
        Coefficient for the quadratic surcharge penalty (EUR/kW²).

    Returns
    -------
    ClusterMetrics
    """
    cumul = sum(t.pgrid_hourly for t in envs)
    par = _par(cumul)
    bills = np.array([float((t.pgrid_hourly * flat_price).sum()) for t in envs])
    profit = _aggregate_profit(envs, w_prices, flat_price, cumul, surcharge_threshold, surcharge_factor)

    return ClusterMetrics(
        scheme=f"Flat ({flat_price} EUR/kWh)",
        par=par,
        bill_mean=bills.mean(),
        bill_std=bills.std(),
        bill_min=bills.min(),
        bill_max=bills.max(),
        agg_profit=profit,
        cumul_pgrid=cumul,
    )


def evaluate_price_sequence(
    envs: list[ResponsiveBuildingEnv],
    price_sequence: pd.Series,
    w_prices: pd.Series,
    surcharge_threshold: float = 230.0,
    surcharge_factor: float = 0.02,
) -> ClusterMetrics:
    """
    Compute cluster-level metrics when all consumers face the same hourly price signal.

    The price_sequence should be expressed in the normalised [-1, 1] action space;
    each environment's act_to_price mapping converts it to EUR/kWh internally.

    Parameters
    ----------
    envs : list of ResponsiveBuildingEnv
    price_sequence : pd.Series
        Normalised hourly price actions in [-1, 1].
    w_prices : pd.Series
        Hourly wholesale prices (EUR/kWh).
    surcharge_threshold, surcharge_factor : float
        Grid operator penalty parameters.

    Returns
    -------
    ClusterMetrics
    """
    for t in envs:
        t.m_pgrid, _disutility, _baseline = t.evaluate_price_sequence(price_sequence)

    cumul = sum(t.m_pgrid for t in envs)
    par = _par(cumul)
    bills = np.array(
        [float((t.m_pgrid * t.act_to_price(price_sequence)).sum()) for t in envs]
    )
    profit = _aggregate_profit_modified(
        envs, w_prices, cumul, surcharge_threshold, surcharge_factor
    )

    return ClusterMetrics(
        scheme="Scheduled (TOU / custom)",
        par=par,
        bill_mean=bills.mean(),
        bill_std=bills.std(),
        bill_min=bills.min(),
        bill_max=bills.max(),
        agg_profit=profit,
        cumul_pgrid=cumul,
    )


def evaluate_rl_agent(
    envs: list[ResponsiveBuildingEnv],
    models: list[PPO],
    w_prices: pd.Series,
    surcharge_threshold: float = 230.0,
    surcharge_factor: float = 0.02,
) -> ClusterMetrics:
    """
    Run each consumer's environment through a trained PPO agent and collect metrics.

    Parameters
    ----------
    envs : list of ResponsiveBuildingEnv
    models : list of PPO
        One trained model per consumer (can repeat the same model for all consumers).
    w_prices : pd.Series
        Hourly wholesale prices (EUR/kWh).

    Returns
    -------
    ClusterMetrics
    """
    for t, m in zip(envs, models):
        _run_agent(t, m)

    cumul = sum(t.m_pgrid for t in envs)
    par = _par(cumul)
    bills = np.array([float((t.m_pgrid * t.prices).sum()) for t in envs])
    profit = _aggregate_profit_modified(
        envs, w_prices, cumul, surcharge_threshold, surcharge_factor
    )

    return ClusterMetrics(
        scheme="RL (PPO)",
        par=par,
        bill_mean=bills.mean(),
        bill_std=bills.std(),
        bill_min=bills.min(),
        bill_max=bills.max(),
        agg_profit=profit,
        cumul_pgrid=cumul,
    )


# ---------------------------------------------------------------------------
# Pricing scheme constructors
# ---------------------------------------------------------------------------


def build_tou_prices(index: pd.DatetimeIndex) -> pd.Series:
    """
    Return a typical two-block time-of-use price schedule in normalised [-1, 1] space.

    Peak hours: 08:00–10:00 and 18:00–20:00 → action = +0.5
    Off-peak   : all other hours              → action = -0.16

    Parameters
    ----------
    index : pd.DatetimeIndex
        Datetime index for a 24-hour period.

    Returns
    -------
    pd.Series with values in [-1, 1].
    """
    actions = [-0.16] * 8 + [0.5] * 3 + [-0.16] * 6 + [0.5] * 3 + [-0.16] * 4
    return pd.Series(actions, index=index)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _par(series: pd.Series) -> float:
    """Peak-to-average ratio."""
    return float(series.max() / series.mean())


def _aggregate_profit(
    envs, w_prices, retail_price, cumul_pgrid, threshold, factor
) -> float:
    t1 = sum(float((t.pgrid_hourly * (retail_price - w_prices)).sum()) for t in envs)
    excess = (cumul_pgrid.to_numpy().clip(min=threshold) - threshold) ** 2
    t2 = (excess * factor).sum()
    return t1 - t2


def _aggregate_profit_modified(envs, w_prices, cumul_pgrid, threshold, factor) -> float:
    t1 = sum(
        float((t.pgrid_hourly * (t.act_to_price(getattr(t, "prices", t.m_pgrid) * 0) - w_prices)).sum())
        for t in envs
    )
    # Re-derive properly using stored prices where available
    t1 = 0.0
    for t in envs:
        prices = getattr(t, "prices", None)
        if prices is None:
            continue
        t1 += float((t.pgrid_hourly * (prices - w_prices)).sum())
    excess = (cumul_pgrid.to_numpy().clip(min=threshold) - threshold) ** 2
    t2 = (excess * factor).sum()
    return t1 - t2


def _run_agent(env: ResponsiveBuildingEnv, model: PPO) -> None:
    """Step through one episode using the RL agent (deterministic policy)."""
    prices = []
    obs, _ = env.reset()
    for _ in range(len(env.pgrid_hourly)):
        action, _ = model.predict(obs, deterministic=True)
        prices.append(float(action.item()))
        obs, _rew, done, _term, _info = env.step(action)
        if done:
            break
    env.m_pgrid = pd.Series(env.ep_pgrid_profile, index=env.pgrid_hourly.index)
    env.prices  = pd.Series(env.act_to_price(np.array(prices)), index=env.pgrid_hourly.index)
