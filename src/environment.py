"""
ResponsiveBuildingEnv: A Gymnasium environment for RL-based real-time electricity pricing.

The environment models a responsive building (or cluster of buildings) whose electricity
demand reacts to retail price signals. An RL agent learns to set hourly prices that:
  - Maximise the service provider's profit margin over wholesale prices
  - Reduce the peak-to-average ratio (PAR) of the aggregate load curve
  - Keep consumer bills close to the flat-rate baseline

Consumer behaviour is captured by two parameters:
  p_lambda  : load-shedding sensitivity — fraction of demand that can be shed when prices are high
  p_epsilon : load-shifting sensitivity — fraction of demand that can be shifted forward in time

A KDE fitted on historical hourly consumption is used to generate stochastic demand
trajectories during training. At test time the recorded load profile can be replayed
directly (use_pgrid_directly=True).
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Box
from sklearn.neighbors import KernelDensity


class ResponsiveBuildingEnv(gym.Env):
    """
    Gymnasium environment for RL-based real-time electricity pricing.

    Parameters
    ----------
    pgrid_hourly : pd.Series
        Historical hourly grid power consumption (kW), used to fit the KDE demand model.
        Index must be a DatetimeIndex with hourly frequency.
    w_prices_hourly : pd.Series
        Hourly day-ahead wholesale electricity prices (EUR/kWh).
    consumer_prices_min_max_tuple : tuple[float, float]
        (min_price, max_price) bounds for the retail price the agent can set (EUR/kWh).
    p_lambda : float
        Load-shedding sensitivity in [0, 1]. Fraction of baseline demand that can be
        reduced at each step when a high price is set.
    p_alpha : float
        Backlog recovery rate in [0, 1]. Controls how quickly deferred demand returns.
    p_epsilon : float
        Load-shifting sensitivity in [0, 1]. Fraction of future demand drawn forward
        when a low price is set.
    demand_t_minus_1 : float, optional
        Last observed demand value before the episode starts (kW). Defaults to the
        historical mean if not provided.
    kde_kwargs : dict
        Keyword arguments forwarded to sklearn KernelDensity (default: Scott's rule,
        Gaussian kernel).
    use_pgrid_directly : bool
        If True, replay pgrid_hourly verbatim instead of sampling from the KDE.
        Intended for evaluation only.
    prev_pgrid_window_stats : tuple[float, float], optional
        (mean, std) of the training-period load, used to normalise observations during
        evaluation so that scaling is consistent with training.
    prev_wprices_window_stats : tuple[float, float], optional
        (mean, std) of the training-period wholesale prices, used similarly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pgrid_hourly: pd.Series,
        w_prices_hourly: pd.Series,
        consumer_prices_min_max_tuple: tuple[float, float],
        p_lambda: float,
        p_alpha: float,
        p_epsilon: float,
        demand_t_minus_1: float = None,
        kde_kwargs: dict = None,
        use_pgrid_directly: bool = False,
        prev_pgrid_window_stats: tuple[float, float] = None,
        prev_wprices_window_stats: tuple[float, float] = None,
    ):
        if kde_kwargs is None:
            kde_kwargs = {"bandwidth": "scott", "kernel": "gaussian"}

        # --- Demand model ---
        self.kernels, self.prev_lh_stats = self._fit_kernels(pgrid_hourly, kde_kwargs)
        # Override scaling stats if provided (keeps eval consistent with training)
        if prev_pgrid_window_stats:
            self.prev_lh_stats = prev_pgrid_window_stats

        self.pgrid_hourly = pgrid_hourly
        self.use_pgrid_directly = use_pgrid_directly

        # --- Wholesale price handling ---
        self.w_prices_hourly = w_prices_hourly
        if prev_wprices_window_stats:
            self.prev_lh_wprices_stats = prev_wprices_window_stats
        else:
            self.prev_lh_wprices_stats = (w_prices_hourly.mean(), w_prices_hourly.std())

        # --- Initial demand ---
        self.demand_t_minus_1 = (
            demand_t_minus_1 if demand_t_minus_1 is not None else pgrid_hourly.mean()
        )

        # --- Behavioural parameters ---
        # f_lambda: load-shedding applied when action > 0 (high price)
        self.f_lambda = lambda x: np.clip(x, 0, 1) * p_lambda
        # f_epsilon: load-shifting applied when action < 0 (low price)
        self.f_epsilon = lambda x: -np.clip(x, -1, 0) * p_epsilon
        self.p_alpha = p_alpha  # backlog recovery rate
        self.p_mix = 0.4  # KDE smoothing coefficient

        # --- Episode horizon ---
        self.l_w = 4  # observation window length (previous steps)
        self.l_u = 24  # episode length (hours per day)

        # --- Gym spaces ---
        # Observation: [t/24, avg_price_so_far, pgrid_hist × l_w, scaled_next_wprice]
        obs_low = np.array([0.0, -1.0] + [-5.0] * self.l_w + [-5.0], dtype=np.float32)
        obs_high = np.array([1.0, 1.0] + [5.0] * self.l_w + [5.0], dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        # Action: normalised price signal in [-1, 1]; mapped to [min_price, max_price]
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Price mapping ---
        self.price_limits = consumer_prices_min_max_tuple
        self.act_to_price = np.poly1d(
            np.polyfit([-1, 1], list(consumer_prices_min_max_tuple), deg=1)
        )

        # Normalisation baseline for profit reward shaping
        self.avg_step_profit_estimate = pgrid_hourly.mean() * (
            sum(consumer_prices_min_max_tuple) / 2
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        """Reset the environment to the start of a new episode."""
        super().reset(seed=seed)
        self.t_ix = 0
        self.w_price_day_idx = 0  # day-ahead prices are known; use index 0 for the day

        # Generate (or load) a baseline demand trajectory for the episode
        if not self.use_pgrid_directly:
            self.bsl_dem = self._get_reference_trajectory(
                self.w_prices_hourly, self.demand_t_minus_1
            )
        else:
            self.bsl_dem = self.pgrid_hourly.to_list()

        self.bsl_std_dev = np.array(self.bsl_dem).std()
        self.backlog_t = 0.0
        self.pgrid_hist = [0.0] * self.l_w
        self.price_hist = [0.0] * self.l_w
        self.avg_price_so_far = 0.0
        self.all_prices_hist = []
        self.ep_profit = 0.0
        self.ep_pgrid_profile = []

        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        """
        Advance the environment by one hour.

        Parameters
        ----------
        action : np.ndarray, shape (1,)
            Normalised price signal in [-1, 1].

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        act = float(action.item())

        # Update running average price (used as an observation feature)
        self.avg_price_so_far = ((self.avg_price_so_far * self.t_ix) + act) / (
            self.t_ix + 1
        )

        wprice_t = self.w_prices_hourly.iloc[24 * self.w_price_day_idx + self.t_ix]
        self.price_hist = self.price_hist[1:] + [act]

        # --- Demand response model ---
        lb = self.f_lambda(act)  # shedding fraction
        ep = self.f_epsilon(act)  # shifting fraction

        # Baseline demand at t, augmented by unrecovered backlog
        b_dem_t = self.bsl_dem[self.t_ix] + self.p_alpha * self.backlog_t
        self.backlog_t -= self.p_alpha * self.backlog_t  # decay backlog

        # Load-shifting: pull future demand earlier when price is low
        ep_impact = 0.0
        if ep > 0.0:
            for i in range(self.l_u - self.t_ix - 1):
                step_impact = (ep ** (i + 1)) * self.bsl_dem[self.t_ix + i + 1]
                ep_impact += step_impact
                self.bsl_dem[self.t_ix + i + 1] -= step_impact

        # Realised demand after shedding and shifting
        pdem_m = b_dem_t * (1 - lb) + ep_impact
        self.backlog_t = max(0.0, self.backlog_t + b_dem_t - pdem_m)
        self.ep_pgrid_profile.append(pdem_m)

        # --- Reward ---
        act_prices = np.array([self.act_to_price(a) for a in self.price_hist])
        self.all_prices_hist.append(act_prices[-1])
        step_profit = pdem_m * (act_prices[-1] - wprice_t)
        self.ep_profit += step_profit

        reward = 100.0 * (step_profit / self.avg_step_profit_estimate)

        # --- State transition ---
        self.t_ix += 1
        scaled_pgrid = (pdem_m - self.prev_lh_stats[0]) / self.prev_lh_stats[1]
        self.pgrid_hist = self.pgrid_hist[1:] + [float(scaled_pgrid)]

        wprice_next = self.w_prices_hourly.iloc[
            24 * self.w_price_day_idx + (self.t_ix + 1) % 24
        ]
        scaled_wprice = (
            wprice_next - self.prev_lh_wprices_stats[0]
        ) / self.prev_lh_wprices_stats[1]

        obs = np.array(
            [self.t_ix / 24, self.avg_price_so_far] + self.pgrid_hist + [scaled_wprice],
            dtype=np.float32,
        )

        # Terminal conditions
        terminated = self.t_ix == (self.l_u)  # - 1
        if terminated:
            aph = np.array(self.all_prices_hist)
            final_load = np.array(self.ep_pgrid_profile)
            # Bonus for keeping the mean retail price close to the flat-rate baseline
            if np.isclose(aph.mean(), 0.25, rtol=0.05):
                reward += 300.0
                # Further bonus for flattening the load curve relative to baseline
                if final_load.std() < self.bsl_std_dev:
                    reward += (self.bsl_std_dev / final_load.std()) * 150.0
            else:
                reward -= 10_000.0

        info = {
            "step_profit": step_profit,
            "pdem_m": pdem_m,
            "retail_price": act_prices[-1],
            "wholesale_price": wprice_t,
        }
        return obs, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def evaluate_price_sequence(
        self,
        price_sequence: pd.Series,
        backlog_t_minus_1: float = 0.0,
        demand_t_minus_1: float = 0.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Apply a fixed price sequence to the demand model and return the resulting
        consumption, dissatisfaction, and (modified) baseline trajectories.

        This is used to benchmark TOU and flat pricing without running the full
        Gym loop.

        Parameters
        ----------
        price_sequence : pd.Series
            Hourly price signals in [-1, 1] (normalised action space), indexed by
            DatetimeIndex.
        backlog_t_minus_1 : float
            Initial backlog value (kWh deferred from before the window).
        demand_t_minus_1 : float
            Last demand observation before the window (kW).

        Returns
        -------
        transformed : pd.Series   — realised demand after price response (kW)
        disutilities : pd.Series  — cumulative backlog at each step (kW)
        altered_baseline : pd.Series — shifted baseline trajectory (kW)
        """
        if self.use_pgrid_directly:
            altered_baseline = self.pgrid_hourly.to_list()[: len(price_sequence)]
        else:
            altered_baseline = self._get_reference_trajectory(
                price_sequence, demand_t_minus_1
            )

        transformed, disutilities = [], []
        d_tm1 = backlog_t_minus_1

        for t in range(len(price_sequence)):
            lb = self.f_lambda(price_sequence.iloc[t])
            ep = self.f_epsilon(price_sequence.iloc[t])

            b_dem_t = altered_baseline[t] + self.p_alpha * d_tm1
            d_tm1 -= self.p_alpha * d_tm1

            ep_impact = 0.0
            if ep > 0.0:
                for i in range(len(price_sequence) - t - 1):
                    step_impact = (ep ** (i + 1)) * altered_baseline[t + i + 1]
                    ep_impact += step_impact
                    altered_baseline[t + i + 1] -= step_impact

            dem_t = b_dem_t * (1 - lb) + ep_impact
            d_tm1 = max(0.0, d_tm1 + b_dem_t - dem_t)
            transformed.append(dem_t)
            disutilities.append(d_tm1)

        return (
            pd.Series(transformed, index=price_sequence.index),
            pd.Series(disutilities, index=price_sequence.index),
            pd.Series(altered_baseline, index=price_sequence.index),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fit_kernels(
        self, pgrid_hourly: pd.Series, kde_kwargs: dict
    ) -> tuple[dict, tuple[float, float]]:
        """Fit one KDE per hour-of-day from the historical load series."""
        kernels = {}
        for hh in range(24):
            data = pgrid_hourly[pgrid_hourly.index.hour == hh].to_numpy().reshape(-1, 1)
            kernels[hh] = KernelDensity(**kde_kwargs).fit(data)
        stats = (pgrid_hourly.mean(), pgrid_hourly.std())
        return kernels, stats

    def _get_reference_trajectory(
        self, price_sequence: pd.Series, demand_t_minus_1: float
    ) -> list[float]:
        """Sample a smooth demand trajectory from the hourly KDEs."""
        profile = []
        for ix, t in enumerate(price_sequence.index):
            k = self.kernels[t.hour].sample().item()
            if ix == 0:
                profile.append(self.p_mix * k + (1 - self.p_mix) * demand_t_minus_1)
            else:
                profile.append(self.p_mix * k + (1 - self.p_mix) * profile[-1])
        return profile
