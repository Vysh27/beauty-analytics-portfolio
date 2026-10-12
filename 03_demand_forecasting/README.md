# 03 — Demand Forecasting (Time-Series)

Forecast daily category demand so planning, inventory and promo teams can staff
and stock ahead — with an honest, back-tested accuracy number.

## Problem

Beauty demand has strong weekly rhythm, a festive annual lift, and sharp spikes
on sale events (e.g. Pink Sale). The business wants a 60-day forecast per
category and a credible read on how accurate it is.

## Approach

1. **Data** — `generate_data.py` builds ~2 years of daily sales for four
   categories, each with trend + weekly seasonality + annual festive lift +
   sale-event spikes + noise.
2. **Back-test** — hold out the last 60 days. Compare three models against a
   **seasonal-naive baseline** (the bar any real model must beat):
   **Holt-Winters** (additive trend + weekly seasonality) and **SARIMA**
   `(1,1,1)(1,0,1)_7`.
3. **Score** — report **MAPE** and **RMSE** on the hold-out, pick the winner,
   refit on full history and forecast forward with a confidence interval.

## Results (synthetic, Skincare, seed 42)

| Model | MAPE | RMSE |
|-------|------|------|
| Seasonal-naive (baseline) | 13.67% | 109.4 |
| Holt-Winters | 9.64% | 82.2 |
| **SARIMA** | **9.31%** | **69.1** |

SARIMA wins and is refit on all 760 days to produce the 60-day forecast. The
point is the honest framing: the model is reported against a baseline, not in
isolation.

## Run it

```bash
python generate_data.py                          # -> data/daily_sales.csv
python forecast.py --category Skincare --horizon 60
# try other categories: Makeup, Haircare, Fragrance
```

## Outputs

| File | What it is |
|------|------------|
| `outputs/<category>_accuracy.csv` | MAPE / RMSE for every model on the hold-out |
| `outputs/<category>_forecast.csv` | The forward forecast (units/day) |
| `outputs/<category>_forecast.png` | Actual vs forecast with an 80% interval |

## Caveats worth stating

Synthetic seasonality is cleaner than reality. On real data you would add
holiday/promo regressors (SARIMAX with exogenous features), handle stock-outs
masking true demand, and refresh the forecast on a schedule.
