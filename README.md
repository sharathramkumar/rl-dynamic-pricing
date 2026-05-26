# RL-Based Real-Time Electricity Pricing

Code accompanying the paper:

> **Real-time Retail Electricity Pricing Using Offline Reinforcement Learning** — Sharath Ram Kumar, Arvind Easwaran, Benoit Delinchant, Rémy Rigo-Mariani, eEnergy '24, Singapore  
> DOI: [https://doi.org/10.1145/3632775.3661964](https://doi.org/10.1145/3632775.3661964)

---

## Overview

An RL agent (PPO) learns to set **hourly retail electricity prices** for a cluster
of N = 100 residential buildings, with the joint objective of:

- Maximising the service provider's profit margin over day-ahead wholesale prices
- Reducing the aggregate **peak-to-average ratio (PAR)**
- Keeping the mean consumer bill close to the flat-rate baseline (0.25 EUR/kWh)

Consumer demand response is modelled by the `ResponsiveBuildingEnv` Gymnasium
environment, which captures two behavioural mechanisms based on a price elasticity model. 
Consumers **redistribute** a part of their energy demand in response to real-time prices,
while conserving their overall energy consumption. More details in the [paper](https://doi.org/10.1145/3632775.3661964).


A KDE fitted on one month of historical consumption generates stochastic demand
trajectories during training; real recorded profiles are replayed at evaluation time.

---

## Repository layout

```
rl-dynamic-pricing/
├── src/
│   ├── __init__.py
│   ├── environment.py   # ResponsiveBuildingEnv (Gymnasium)
│   ├── evaluation.py    # ClusterMetrics, benchmark helpers
│   └── plotting.py      # Figure utilities
├── data/
│   └── README.md        # Expected input format and data sources
├── outputs/
│   ├── figures/         # Generated PNG figures (git-ignored)
│   └── tb_logs/         # TensorBoard training logs (git-ignored)
├── experiment.ipynb     # End-to-end reproducible notebook
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/sharathramkumar/rl-dynamic-pricing.git
cd rl-dynamic-pricing

# 2. Create a virtual environment (Python ≥ 3.10)
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate                             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data
#    See data/README.md for the expected CSV format and data sources.

# 5. Run the notebook
jupyter lab experiment.ipynb
```

---

## Environment API

`ResponsiveBuildingEnv` follows the standard [Gymnasium](https://gymnasium.farama.org/) API.

```python
from src import ResponsiveBuildingEnv
import pandas as pd

env = ResponsiveBuildingEnv(
    pgrid_hourly              = demand_series,          # pd.Series, kW, DatetimeIndex
    w_prices_hourly           = wholesale_price_series, # pd.Series, EUR/kWh
    consumer_prices_min_max_tuple = (0.20, 0.30),       # EUR/kWh bounds
    p_lambda  = 0.40,   # load-shedding sensitivity
    p_alpha   = 0.25,   # backlog recovery rate
    p_epsilon = 0.40,   # load-shifting sensitivity
)

obs, info = env.reset()
for step in range(24):
    action = env.action_space.sample()   # or use a trained agent
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

**Observation space** (7-dimensional `float32`):

| Index | Feature |
|-------|---------|
| 0 | Normalised time-of-day `t / 24` |
| 1 | Running average price action |
| 2–5 | Last 4 hours of normalised demand |
| 6 | Next-hour normalised wholesale price |

**Action space**: scalar in `[-1, 1]`, linearly mapped to `[price_min, price_max]`.

---

## Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from src import ResponsiveBuildingEnv

env = ResponsiveBuildingEnv(...)
model = PPO('MlpPolicy', env, gamma=0.958, verbose=1)
model.learn(total_timesteps=100_000, progress_bar=True)
model.save('outputs/my_model')
```

If you have `tensorboard` installed, run the following snippet to visualize the training progress:

```bash
tensorboard --logdir tb_logs
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{kumarRealtimeRetailElectricity2024,
  title = {Real-Time {{Retail Electricity Pricing Using Offline Reinforcement Learning}}},
  booktitle = {Proceedings of the 15th {{ACM International Conference}} on {{Future}} and {{Sustainable Energy Systems}}},
  author = {Kumar, Sharath Ram and Easwaran, Arvind and Delinchant, Benoit and {Rigo-Mariani}, Remy},
  year = 2024,
  pages = {454--458},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3632775.3661964},
}
```