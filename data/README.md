# Data

This directory holds the input files required to run `experiment.ipynb`.

## Required files

| File | Description |
|------|-------------|
| `demand_type_a.csv` | Hourly net grid consumption for consumer type A (kW) |
| `demand_type_b.csv` | Hourly net grid consumption for consumer type B (kW) |
| `wholesale_prices.csv` | Hourly day-ahead wholesale electricity prices |

---

## Schema

### `demand_type_a.csv` and `demand_type_b.csv`

```
datetime,demand_kw
2018-07-01 00:00:00,12.34
2018-07-01 01:00:00,11.87
...
```

- `datetime` — ISO 8601, hourly resolution, local time
- `demand_kw` — net grid power consumption in kW (positive = import from grid)

### `wholesale_prices.csv`

```
Datetime (Local),Price (EUR/MWhe)
2018-07-01 00:00:00,45.2
2018-07-01 01:00:00,41.8
...
```

This matches the format exported by the
[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/).  
The notebook converts MWh prices to kWh by multiplying by `1e-3`.

---

## Obtaining wholesale prices

French day-ahead prices for 2018 can be downloaded directly from ENTSO-E:

1. Visit https://transparency.entsoe.eu/
2. Select **Day-ahead Prices** → area: **FR** → 2018
3. Export as CSV